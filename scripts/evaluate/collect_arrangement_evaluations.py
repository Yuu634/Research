"""追加実験の全チェックポイントを評価し、既存結果とvalid/invalid別に統合する。"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path
import shutil
from time import perf_counter

import torch

from data.pairdata.io import file_hash, write_json
from scripts.evaluate.evaluate_including_invalid import (
    SCOPES, csv_row, measure_output, read_json, summarize, verify_valid_summary, write_csv,
)
from src.beam_muse.data import collate, load_cache, load_metadata
from src.beam_muse.experiment import load_checkpoint
from src.common.comparison import evaluate_output, summarize_outputs
from src.common.provenance import implementation_hashes, validate_sources
from src.musemorphose.generation import Grammar, generate


WORKER = {}


def evaluate_batch(start, end, path):
    model, vocabulary, scorer = (WORKER[k] for k in ('model', 'vocabulary', 'scorer'))
    grammar = Grammar(vocabulary)
    records = []
    with torch.no_grad():
        for example in WORKER['examples'][start:end]:
            started = perf_counter()
            batch = collate([example], vocabulary, scorer, WORKER['variant'], 'cpu', training=False)
            generated = generate(model, batch, example.prepared, vocabulary, grammar=grammar)
            elapsed = perf_counter()-started
            hybrid = WORKER['variant'] == 'beam_muse'
            row = evaluate_output(example.prepared, generated.candidate, scorer,
                case_id=example.case_id, source_id=example.source_id,
                elapsed_seconds=elapsed+(example.beam_seconds if hybrid else 0.),
                rule_count=len(example.history) if hybrid else 0,
                failure_reason=generated.failure_reason, config=WORKER['evaluation'])
            measured = measure_output(example.prepared, generated.candidate, scorer, row)
            measured.update(candidate=asdict(generated.candidate) if generated.candidate is not None else None,
                            tokens=generated.tokens)
            records.append(dict(evaluation=row, including_invalid=measured))
    write_json(path, records)
    write_json(path.with_suffix('.receipt.json'), dict(start=start, end=end, sha256=file_hash(path)))
    return len(records)


def case_contract(rows):
    result = {r['case_id']: (r['source_id'], r['target']) for r in rows}
    if len(result) != len(rows):
        raise ValueError('評価IDが重複')
    return result


def evaluate_checkpoint(args, checkpoint, split, examples, scorer, evaluation, vocabulary, manifest):
    run = args.run
    completed = read_json(checkpoint.parent/'completed.json')
    identity = read_json(run/'run.json')['identity']
    digest = file_hash(checkpoint)
    if completed['identity'] != identity or completed['checkpoint_sha256'] != digest:
        raise ValueError('学習完了記録・重み・実験IDが不一致')
    model, state = load_checkpoint(checkpoint, vocabulary, identity, 'cpu')
    model_id = f'{run.name}/{checkpoint.parent.name}'
    provenance = dict(checkpoint_sha256=digest, input_sha256=manifest['splits'][split]['sha256'],
        cache_manifest_sha256=file_hash(run/'cache/manifest.json'),
        selection_sha256=file_hash(run/'selection.json'), sources=implementation_hashes('beam_muse'),
        script_sha256=file_hash(__file__), measurement_script_sha256=file_hash(
            Path(__file__).with_name('evaluate_including_invalid.py')),
        torch=str(torch.__version__), device='cpu', threads=1, decoding='greedy',
        latent_sampling=False, batch_size=args.batch_size, epoch=state['epoch'],
        trial=state['trial'], evaluation_config=asdict(evaluation),
        timing='CPU生成時間。複合モデルは保存済みBeamSearch時間を加算。実行環境差のある速度比較には使わない。')
    cache = args.output/'evaluation_cache'/model_id/split
    cache.mkdir(parents=True, exist_ok=True)
    if (cache/'manifest.json').exists() and read_json(cache/'manifest.json') != provenance:
        raise ValueError('再開キャッシュの重み・入力・実装・設定が不一致')
    write_json(cache/'manifest.json', provenance)
    WORKER.clear()
    WORKER.update(model=model, vocabulary=vocabulary, scorer=scorer, evaluation=evaluation,
                  examples=examples, variant=state['trial']['variant'])
    paths, pending = [], []
    for start in range(0, len(examples), args.batch_size):
        end = min(start+args.batch_size, len(examples))
        path = cache/f'batch_{start:06d}.json'
        receipt = path.with_suffix('.receipt.json')
        paths.append(path)
        if path.exists() and receipt.exists():
            if read_json(receipt) != dict(start=start, end=end, sha256=file_hash(path)):
                raise ValueError('評価キャッシュが破損: '+str(path))
        else:
            pending.append((start, end, path))
    done = len(examples)-sum(end-start for start, end, _ in pending)
    print(f'{model_id} {split}: {done}/{len(examples)}件を再利用', flush=True)
    started = perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('fork')) as pool:
            futures = [pool.submit(evaluate_batch, *task) for task in pending]
            for future in as_completed(futures):
                done += future.result()
                print(f'{model_id} {split}: {done}/{len(examples)}件 ({perf_counter()-started:.0f}秒)', flush=True)
    records = [r for path in paths for r in read_json(path)]
    if [r['evaluation']['case_id'] for r in records] != [e.case_id for e in examples]:
        raise ValueError('出力と入力のID・順序が不一致')
    rows = [r['evaluation'] for r in records]
    measured = [r['including_invalid'] for r in records]
    regular = summarize_outputs(rows)
    inclusive = summarize(measured, evaluation.target_tolerance)
    verify_valid_summary(inclusive['valid_only'], regular)
    folder = args.output/'models'/model_id
    write_json(folder/f'{split}_rows.json', rows)
    write_json(folder/f'{split}_including_invalid_rows.json', measured)
    write_json(folder/f'{split}_evaluation.json', provenance)
    return regular, inclusive, provenance, measured


def read_legacy(args, split):
    regular_path = args.previous/f'{split}_summary.json'
    inclusive_path = args.previous/f'{split}_including_invalid_summary.json'
    regular, inclusive = read_json(regular_path), read_json(inclusive_path)
    receipt = read_json(args.previous/f'{split}_including_invalid_manifest.json')
    if file_hash(regular_path) != receipt['reference_sha256'] or regular.keys() != inclusive.keys():
        raise ValueError('既存の通常評価・追加評価が不一致')
    provenance, selected, contracts = {}, {}, None
    for model_id, summary in regular.items():
        details = inclusive[model_id]
        verify_valid_summary(details['valid_only'], summary)
        source = args.previous/'models'/model_id/f'{split}_including_invalid_rows.json'
        rows = read_json(source)
        current = case_contract(rows)
        if contracts is not None and contracts != current:
            raise ValueError('既存モデル間の入力ID・目標が不一致')
        contracts = current
        if summarize(rows, summary['evaluation_config']['target_tolerance']) != {s: details[s] for s in SCOPES}:
            raise ValueError('既存の詳細行と集計が不一致: '+model_id)
        identity = receipt['models'][model_id]
        if 'sources' in identity:
            validate_sources(identity['sources'], implementation_hashes('beam_muse'))
            if file_hash(args.root/model_id/'best.pt') != identity['checkpoint_sha256']:
                raise ValueError('既存ニューラル重みのハッシュ不一致')
        folder = args.output/'models'/model_id
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, folder/source.name)
        provenance[model_id] = dict(mode='verified_previous_results', previous=str(args.previous),
            regular_summary_sha256=file_hash(regular_path), inclusive_summary_sha256=file_hash(inclusive_path),
            rows_sha256=file_hash(source), original_evaluation=identity)
        selected[model_id] = details['selected_on_val']
    return regular, inclusive, provenance, selected, contracts


def table_row(model_id, selected, scope, inclusive, regular):
    row = csv_row(model_id, selected, scope, inclusive[scope])
    if scope == 'valid_only':
        # 従来のsummary.csvと同様、n/statusは全入力、各指標のnは測定対象。
        row.update(n=regular['n'], **regular['status_counts'], valid_rate=regular['valid_rate']['value'])
    row.update(target_success_rate=regular['target_success_rate']['value'],
               joint_success_rate=regular['joint_success_rate']['value'],
               failure_adjusted_mae=regular['failure_adjusted_mae']['mean'])
    return row


def save_tables(args, split, regular, inclusive, provenance, selected, complete):
    for model_id in regular:
        s = inclusive[model_id]
        s.update(evaluation_config=regular[model_id]['evaluation_config'], selected_on_val=selected[model_id],
            original_target_success_rate=regular[model_id]['target_success_rate'],
            original_joint_success_rate=regular[model_id]['joint_success_rate'])
        folder = args.output/'models'/model_id
        write_json(folder/f'{split}_summary.json', regular[model_id])
        write_json(folder/f'{split}_including_invalid_summary.json', s)
    write_json(args.output/f'{split}_summary.json', regular)
    write_json(args.output/f'{split}_including_invalid_summary.json', inclusive)
    rows = [table_row(m, selected[m], scope, inclusive[m], regular[m]) for m in regular for scope in SCOPES]
    write_csv(args.output/f'{split}_summary.csv', [r for r in rows if r['scope'] == 'valid_only'])
    write_csv(args.output/f'{split}_including_invalid_summary.csv', [r for r in rows if r['scope'] == 'all_outputs'])
    # 比較CSVでは、valid_onlyのn/statusもその集合自身の件数にする。
    comparison = [csv_row(m, selected[m], scope, inclusive[m][scope]) for m in regular for scope in SCOPES]
    write_csv(args.output/f'{split}_valid_invalid_comparison.csv', comparison)
    write_csv(args.output/f'{split}_selected_comparison.csv', [r for r in rows if r['selected_on_val'] and r['scope'] != 'invalid_only'])
    write_json(args.output/f'{split}_evaluation_manifest.json', dict(
        schema_version=1, status='complete' if complete else 'running',
        updated_at_utc=datetime.now(timezone.utc).isoformat(), model_count=len(regular),
        selected_model_ids_on_val=[m for m in regular if selected[m]], models=provenance))


def validate_inputs(examples, previous_examples):
    if len(examples) != len(previous_examples):
        raise ValueError('新旧入力件数が不一致')
    old = {e.case_id: e for e in previous_examples}
    if len(old) != len(previous_examples) or len({e.case_id for e in examples}) != len(examples):
        raise ValueError('新旧入力IDが重複')
    for e in examples:
        if e.case_id not in old or e.source_id != old[e.case_id].source_id or e.prepared != old[e.case_id].prepared:
            raise ValueError('新旧の原譜・文脈・目標・制約が不一致: '+e.case_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('trained_model'))
    parser.add_argument('--run', type=Path, default=Path('trained_model/beam-muse_v1'))
    parser.add_argument('--previous', type=Path, default=Path('output/eval-result'))
    parser.add_argument('--output', type=Path, default=Path('output/eval_result'))
    parser.add_argument('--splits', nargs='+', choices=('test', 'itest'), default=['itest', 'test'])
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if min(args.workers, args.batch_size) < 1 or args.previous.resolve() == args.output.resolve():
        parser.error('workers/batch-sizeは正整数。既存結果と出力先は分けてください')
    torch.set_num_threads(1)
    manifest = read_json(args.run/'cache/manifest.json')
    paths = manifest['resolved_paths']
    scorer, _, _, evaluation, _, vocabulary = load_metadata(paths['meta'], paths['beam_model'])
    selection = read_json(args.run/'selection.json')
    run_identity = read_json(args.run/'run.json')['identity']
    if selection['selected_on'] != 'val' or selection['identity'] != run_identity:
        raise ValueError('選定記録と実験IDが不一致')
    if run_identity['cache_sha256'] != file_hash(args.run/'cache/manifest.json'):
        raise ValueError('学習時キャッシュのハッシュ不一致')
    validate_sources(run_identity['sources'], implementation_hashes('beam_muse'))
    checkpoints = sorted(args.run.glob('*/best.pt'))
    if not checkpoints:
        raise ValueError('評価可能なチェックポイントなし')
    winners = {v['name'] for v in selection['winners'].values()}
    for winner in selection['winners'].values():
        if file_hash(args.run/winner['name']/'best.pt') != winner['checkpoint_sha256']:
            raise ValueError('val選定時の重みと不一致')
    args.output.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        regular, inclusive, provenance, selected, contracts = read_legacy(args, split)
        examples, _ = load_cache(args.run/'cache', split, paths['meta'], paths['beam_model'])
        previous_examples, _ = load_cache(args.root/'musemorphose_v1/cache', split, paths['meta'], paths['beam_model'])
        validate_inputs(examples, previous_examples)
        if contracts != {e.case_id: (e.source_id, e.prepared.target) for e in examples}:
            raise ValueError('既存評価と新規入力のID・目標が不一致')
        if any(s['evaluation_config'] != asdict(evaluation) for s in regular.values()):
            raise ValueError('既存評価と閾値が不一致')
        save_tables(args, split, regular, inclusive, provenance, selected, False)
        for checkpoint in checkpoints:
            model_id = f'{args.run.name}/{checkpoint.parent.name}'
            s, inc, identity, measured = evaluate_checkpoint(
                args, checkpoint, split, examples, scorer, evaluation, vocabulary, manifest)
            if case_contract(measured) != contracts:
                raise ValueError('新規評価と既存評価のID・目標が不一致')
            regular[model_id], inclusive[model_id], provenance[model_id] = s, inc, identity
            selected[model_id] = checkpoint.parent.name in winners
            save_tables(args, split, regular, inclusive, provenance, selected, False)
            print(f'{model_id} {split}: 集計完了 valid={s["status_counts"]["valid"]}/{s["n"]}', flush=True)
        save_tables(args, split, regular, inclusive, provenance, selected, True)
    write_json(args.output/'manifest.json', dict(schema_version=2, status='complete',
        created_at_utc=datetime.now(timezone.utc).isoformat(), previous=str(args.previous),
        added_run=str(args.run), splits=args.splits, added_models=[p.parent.name for p in checkpoints],
        notes=['既存10モデルは詳細行を再集計して一致を確認した上で再利用。',
               '追加実験の全チェックポイントは同じCPU・greedy生成で評価。学習・val選定は変更しない。',
               'summary.csvの難易度・類似度はvalidのみ。n/status/valid_rateは全入力。',
               'including_invalid_summary.csvは制約違反も含め、測定可能な全出力を集計。',
               'valid_invalid_comparison.csvのn/statusは各scope内の件数。',
               'target_success_ignoring_validityは制約を無視した目標達成率。無出力・算出不能も分母に残す。',
               '通常のtarget_success_rate/joint_success_rateは制約違反を不成功とする従来定義。',
               '両方空の発音集合は類似度の欠測。各指標のnを併記。',
               '選定済みモデルのGPU評価はtrained_model内に保持。今回の追加結果はCPU評価として別保存。']))


if __name__ == '__main__':
    main()
