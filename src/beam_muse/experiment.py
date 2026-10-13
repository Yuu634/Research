"""共通データによる学習・val選定・固定後のholdout評価。trial完了単位で再開する。"""
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
from time import perf_counter
import numpy as np
import torch

from data.pairdata.io import file_hash, write_json
from src.beam_muse.data import collate, load_cache, load_metadata, source_hashes
from src.common.comparison import evaluate_output, sequence_costs, summarize_outputs
from src.musemorphose.generation import Grammar, generate, score_generation
from src.common.losses import LossConfig, policy_loss, reconstruction_kl
from src.common.loss_logging import MetricMeans, TrialLossLog
from src.common.provenance import validate_sources
from src.beam_muse.model import BeamMuseMorphose
from src.beam_muse.configuration import validate_data_config
from src.musemorphose.conditional import ConditionalMuseMorphose, ModelConfig


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(variant, config, vocabulary):
    if variant not in ('beam_muse', 'musemorphose'):
        raise ValueError('ニューラル方式名が不正')
    if config.use_skeleton != (variant == 'beam_muse'):
        raise ValueError('単体は骨格なし、ハイブリッドは骨格ありを明示する')
    ids = vocabulary['token2idx']
    cls = BeamMuseMorphose if variant == 'beam_muse' else ConditionalMuseMorphose
    return cls(len(ids), ids['<PAD>'], ids['<BAR>'], config)


def expand_trials(config):
    if set(config) - {'data'} != {'schema_version', 'model', 'loss', 'training', 'trials'} or type(config['schema_version']) is not int or config['schema_version'] != 1:
        raise ValueError('実験設定の形式が不正')
    validate_data_config(config)
    result, names = [], set()
    for trial in config['trials']:
        if set(trial)-{'name', 'variant', 'model', 'loss', 'training'}:
            raise ValueError('未知のtrial設定')
        name, variant = trial['name'], trial['variant']
        if not isinstance(name, str) or not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in name) or name in names:
            raise ValueError('trial名が不正または重複')
        names.add(name)
        model = ModelConfig(**(config['model'] | trial.get('model', {}))).validate()
        if variant not in ('musemorphose', 'beam_muse') or model.use_skeleton != (variant == 'beam_muse'):
            raise ValueError('方式と骨格設定が不一致')
        loss = LossConfig(**(config['loss'] | trial.get('loss', {}))).validate()
        train = config['training'] | trial.get('training', {})
        if set(train) != {'epochs', 'batch_size', 'learning_rate', 'weight_decay', 'seed', 'risk_interval', 'risk_examples', 'grad_clip'}:
            raise ValueError('training設定のキーが不正')
        for key in ('epochs', 'batch_size', 'risk_interval', 'risk_examples'):
            if type(train[key]) is not int or train[key] < 1:
                raise ValueError(key+'は正整数')
        if type(train['seed']) is not int or not 0 <= train['seed'] < 2**32:
            raise ValueError('seedは0以上2**32未満の整数')
        for key in ('learning_rate', 'weight_decay', 'grad_clip'):
            if type(train[key]) not in (float, int) or not math.isfinite(train[key]) or train[key] < 0:
                raise ValueError(key+'は有限・非負')
        if train['learning_rate'] == 0 or train['grad_clip'] == 0:
            raise ValueError('学習率と勾配上限は正数')
        result.append(dict(name=name, variant=variant, model=asdict(model), loss=asdict(loss), training=train))
    if not result:
        raise ValueError('ニューラル方式の候補が1件以上必要')
    return result


def selection_key(summary, name=''):
    return (-summary['source_macro']['joint_success_rate']['mean'],
            summary['source_macro']['failure_adjusted_mae']['mean'],
            summary.get('validation_reconstruction', float('inf')), name)


def evaluate(model, examples, vocabulary, scorer, variant, device, evaluation, *, loss_log=None, phase='val', epoch=None, step=None):
    model.eval()
    rows, rec_sum, rec_tokens = [], 0., 0
    loss_means, beam_means = MetricMeans(), MetricMeans()
    grammar = Grammar(vocabulary)
    with torch.no_grad():
        for example in examples:
            started = perf_counter()
            batch = collate([example], vocabulary, scorer, variant, device, training=False)
            result = generate(model, batch, example.prepared, vocabulary, grammar=grammar)
            elapsed = perf_counter()-started + (example.beam_seconds if variant == 'beam_muse' else 0.)
            rows.append(evaluate_output(example.prepared, result.candidate, scorer,
                        case_id=example.case_id, source_id=example.source_id, elapsed_seconds=elapsed,
                        rule_count=len(example.history) if variant == 'beam_muse' else 0,
                        failure_reason=result.failure_reason, config=evaluation))
            teacher_batch = collate([example], vocabulary, scorer, variant, device)
            out = model(teacher_batch, sample=False)
            loss = reconstruction_kl(out, teacher_batch['decoder_target'], model.pad_id)
            n = int(teacher_batch['decoder_target'].ne(model.pad_id).sum())
            rec_sum += float(loss['reconstruction'])*n
            rec_tokens += n
            if loss_log is not None:
                loss_means.add({k: float(loss[k]) for k in ('kl', 'kl_raw')})
                if variant == 'beam_muse':
                    score = scorer.score(example.prepared.original, example.intermediate, example.prepared.target,
                                         previous=example.prepared.previous, guards=example.prepared.guards,
                                         rule_count=len(example.history))
                    beam_means.add(asdict(score))
    summary = summarize_outputs(rows)
    summary['validation_reconstruction'] = rec_sum/rec_tokens
    if loss_log is not None:
        means = loss_means.means()
        means['reconstruction'] = summary['validation_reconstruction']
        means['total'] = (means['reconstruction'] + LossConfig().beta * means['kl']
                          if means['kl'] is not None else None)
        loss_log.write(phase, event='evaluation_summary', epoch=epoch, step=step,
                       n=len(rows), tokens=rec_tokens, musemorphose=means,
                       beamsearch=beam_means.means() if variant == 'beam_muse' else None,
                       beta=LossConfig().beta, free_bits=LossConfig().free_bits, evaluation=summary)
    return summary, rows


def train_trial(trial, train_rows, val_rows, vocabulary, scorer, evaluation, identity, directory, device, *, log_interval=1):
    directory = Path(directory)
    receipt = directory/'completed.json'
    log_config = dict(trial=trial, provenance=identity, evaluation=asdict(evaluation), beamsearch_score=asdict(scorer.config),
                      logging=dict(format='JSON Lines', log_interval=log_interval,
                                   total='rec + beta * KL（補助損失を含まない）',
                                   beamsearch='入力ごとの採用中間譜面を再採点。単体では該当なし',
                                   evaluation_loss='評価1回につき1行。再構成はトークン数、KLは件数で平均。LossConfig既定値を使用'))
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['identity'] != identity or saved['trial'] != trial or file_hash(directory/'best.pt') != saved['checkpoint_sha256']:
            raise ValueError('完了trialの設定・checkpointが変更された')
        loss_log = TrialLossLog(directory, log_config, log_interval=log_interval)
        loss_log.plot()
        return saved
    directory.mkdir(parents=True, exist_ok=True)
    loss_log = TrialLossLog(directory, log_config, reset=True, log_interval=log_interval)
    t = trial['training']
    seed_all(t['seed'])
    model = build_model(trial['variant'], ModelConfig(**trial['model']), vocabulary).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t['learning_rate'], weight_decay=t['weight_decay'])
    loss_config = LossConfig(**trial['loss'])
    grammar = Grammar(vocabulary)
    history, best, step = [], None, 0
    for epoch in range(t['epochs']):
        order = list(range(len(train_rows)))
        random.Random(t['seed']+epoch).shuffle(order)
        losses, risks = [], []
        for start in range(0, len(order), t['batch_size']):
            examples = [train_rows[i] for i in order[start:start+t['batch_size']]]
            batch = collate(examples, vocabulary, scorer, trial['variant'], device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = reconstruction_kl(output, batch['decoder_target'], model.pad_id, loss_config)
            loss_log.write('train', event='base_loss', epoch=epoch+1, step=step+1,
                           case_ids=[e.case_id for e in examples],
                           musemorphose={k: float(v.detach()) for k, v in loss.items()}, beamsearch=None,
                           beta=loss_config.beta, free_bits=loss_config.free_bits,
                           learning_rate=optimizer.param_groups[0]['lr'])
            loss_log.intermediate(examples, scorer, trial['variant'], 'train', epoch+1, step+1)
            if not torch.isfinite(loss['total']):
                raise ValueError('再構成・KL損失が非有限')
            loss['total'].backward()
            losses.append({k: float(v.detach()) for k, v in loss.items()})
            if step % t['risk_interval'] == 0:
                # dropoutを無効にし、sampleとlog p再計算を同じ方策にする。
                model.eval()
                risk_examples = examples[:t['risk_examples']]
                for example in risk_examples:
                    inputs = collate([example], vocabulary, scorer, trial['variant'], device, training=False)
                    with torch.no_grad():
                        greedy = generate(model, inputs, example.prepared, vocabulary, grammar=grammar)
                        sampled = generate(model, inputs, example.prepared, vocabulary, sample=True, grammar=grammar)
                    sampled_cost = sequence_costs(example.prepared, sampled.candidate, scorer)
                    greedy_cost = sequence_costs(example.prepared, greedy.candidate, scorer)
                    logp = score_generation(model, inputs, sampled, vocabulary, grammar).unsqueeze(0)
                    policy = policy_loss(logp, [sampled_cost], [greedy_cost], loss_config)
                    scale = t['risk_interval']/len(risk_examples)
                    loss_log.write('train', event='auxiliary_loss', epoch=epoch+1, step=step+1,
                                   case_id=example.case_id, musemorphose=None, beamsearch=None,
                                   auxiliary=dict(parts={k: float(v.detach()) for k, v in policy.items()},
                                                  scale=scale, scaled_total=float((policy['total']*t['risk_interval']/len(risk_examples)).detach()),
                                                  sampled=sampled_cost, greedy=greedy_cost))
                    # interval回に1度なので期待される更新係数を補正する。
                    (policy['total']*t['risk_interval']/len(risk_examples)).backward()
                    risks.append(dict(sampled=sampled_cost, greedy=greedy_cost, policy=float(policy['total'].detach())))
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            step += 1
        summary, rows = evaluate(model, val_rows, vocabulary, scorer, trial['variant'], device, evaluation,
                                 loss_log=loss_log, epoch=epoch+1, step=step)
        record = dict(epoch=epoch+1, steps=step, training_means={k: float(np.mean([r[k] for r in losses])) for k in losses[0]},
                       risks=risks, validation=summary, last_gradient_norm=float(norm))
        history.append(record)
        write_json(directory/'history.json', history)
        if best is None or selection_key(summary) < selection_key(best['validation']):
            best = dict(epoch=epoch+1, validation=summary)
            checkpoint = dict(schema_version=1, trial=trial, identity=identity, epoch=epoch+1,
                              state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            temporary = directory/'best.pt.tmp'
            torch.save(checkpoint, temporary)
            temporary.replace(directory/'best.pt')
            write_json(directory/'val_rows.json', rows)
        loss_log.plot(best_epoch=best['epoch'])
        print(f'{trial["name"]}: epoch={epoch+1}, val同時達成={summary["joint_success_rate"]["value"]:.4f}, '
              f'適格率={summary["valid_rate"]["value"]:.4f}', flush=True)
    completed = dict(identity=identity, trial=trial, **best, checkpoint_sha256=file_hash(directory/'best.pt'))
    write_json(receipt, completed)
    return completed


def load_checkpoint(path, vocabulary, identity=None, device='cpu'):
    # ローカルで自分が学習したcheckpointのみを読み込む。
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['schema_version'] != 1 or (identity is not None and checkpoint['identity'] != identity):
        raise ValueError('checkpoint形式または学習時の入力・設定が異なる')
    validate_sources(checkpoint['identity']['sources'], source_hashes())
    trial = checkpoint['trial']
    model = build_model(trial['variant'], ModelConfig(**trial['model']), vocabulary)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.to(device).eval(), checkpoint


def run_experiment(config, cache, meta, selected_model, output, device, *, phase='select', smoke=False, log_interval=1):
    trials = expand_trials(config)
    variants = tuple(v for v in ('musemorphose', 'beam_muse') if any(t['variant'] == v for t in trials))
    scorer, _, _, evaluation, _, vocabulary = load_metadata(meta, selected_model)
    train_rows, manifest = load_cache(cache, 'train', meta, selected_model)
    val_rows, _ = load_cache(cache, 'val', meta, selected_model)
    if 'beam_muse' in variants and manifest['identity'].get('intermediate_mode', 'beamsearch') != 'beamsearch':
        raise ValueError('ハイブリッドにはBeamSearch中間譜面付きキャッシュが必要')
    if bool(manifest['identity']['smoke_limit']) != smoke:
        raise ValueError('smoke用キャッシュと本実験を混在させない')
    if phase == 'test' and smoke:
        raise ValueError('smokeではholdoutを評価しない')
    if any(t['model']['max_tokens'] != manifest['identity']['max_tokens'] for t in trials):
        raise ValueError('候補間とキャッシュで同じトークン上限を指定する')
    identity = dict(cache_sha256=file_hash(Path(cache)/'manifest.json'), config=config,
                    sources=source_hashes(), smoke=smoke,
                    vocabulary=manifest['identity']['vocabulary'], bounds=manifest['identity']['bounds'],
                    beam_model=manifest['identity']['beam_model'],
                    runtime=dict(device=str(device), torch=str(torch.__version__), cuda=torch.version.cuda,
                                 threads=torch.get_num_threads()))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    record = output/'run.json'
    if record.exists():
        saved_identity = json.loads(record.read_text())['identity']
        if saved_identity != identity:
            if phase != 'test' or {k: v for k, v in saved_identity.items() if k != 'sources'} != {
                    k: v for k, v in identity.items() if k != 'sources'}:
                raise ValueError('実験の設定・入力が変更された。別の出力先が必要')
            # 検証済みの評価指標追加では、学習時のidentityと重みを保持して再評価する。
            validate_sources(saved_identity['sources'], identity['sources'])
            identity = saved_identity
    if not record.exists():
        write_json(record, dict(identity=identity, device=str(device), torch=str(torch.__version__),
                                cuda=torch.version.cuda, threads=torch.get_num_threads()))
    selection_path = output/'selection.json'
    if phase == 'select':
        results = [train_trial(t, train_rows, val_rows, vocabulary, scorer, evaluation, identity, output/t['name'], device,
                               log_interval=log_interval) for t in trials]
        winners = {variant: min((r for r in results if r['trial']['variant'] == variant),
                               key=lambda r: selection_key(r['validation'], r['trial']['name'])) for variant in variants}
        selected = dict(selected_on='val', identity=identity,
                        winners={variant: dict(name=r['trial']['name'], checkpoint_sha256=r['checkpoint_sha256'],
                                               validation=r['validation'], epoch=r['epoch']) for variant, r in winners.items()})
        if selection_path.exists() and json.loads(selection_path.read_text()) != selected:
            raise ValueError('確定したval選定結果を変更しない')
        write_json(selection_path, selected)
        summaries = {variant: r['validation'] for variant, r in winners.items()}
        if 'beam_muse' in variants:
            beam_rows = [evaluate_output(e.prepared, e.intermediate, scorer, case_id=e.case_id, source_id=e.source_id,
                         elapsed_seconds=e.beam_seconds, rule_count=len(e.history), config=evaluation) for e in val_rows]
            summaries = dict(beamsearch=summarize_outputs(beam_rows), **summaries)
        write_json(output/'val_comparison.json', summaries)
        return selected
    if phase != 'test':
        raise ValueError('phaseはselectまたはtest')
    selected = json.loads(selection_path.read_text())
    if selected['selected_on'] != 'val' or selected['identity'] != identity or set(selected['winners']) != set(variants):
        raise ValueError('valで確定した選定記録が必要')
    for split in ('test', 'itest'):
        examples, _ = load_cache(cache, split, meta, selected_model)
        summaries = {}
        if 'beam_muse' in variants:
            beam_rows = [evaluate_output(e.prepared, e.intermediate, scorer, case_id=e.case_id, source_id=e.source_id,
                         elapsed_seconds=e.beam_seconds, rule_count=len(e.history), config=evaluation) for e in examples]
            summaries['beamsearch'] = summarize_outputs(beam_rows)
            write_json(output/f'{split}_beamsearch_rows.json', beam_rows)
        for variant, winner in selected['winners'].items():
            path = output/winner['name']/'best.pt'
            if file_hash(path) != winner['checkpoint_sha256']:
                raise ValueError('選定後に重みが変更された')
            model, checkpoint = load_checkpoint(path, vocabulary, identity, device)
            config_path = output/winner['name']/'config.json'
            log_config = (json.loads(config_path.read_text()) if config_path.exists() else
                          dict(trial=checkpoint['trial'], provenance=identity, evaluation=asdict(evaluation),
                               beamsearch_score=asdict(scorer.config), logging=dict(format='JSON Lines')))
            loss_log = TrialLossLog(output/winner['name'], log_config)
            summaries[variant], rows = evaluate(model, examples, vocabulary, scorer, variant, device, evaluation,
                                                loss_log=loss_log, phase=split, epoch=winner['epoch'])
            loss_log.plot()
            write_json(output/f'{split}_{variant}_rows.json', rows)
        write_json(output/f'{split}_comparison.json', summaries)
    return selected
