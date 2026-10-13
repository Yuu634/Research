"""保存済みtest/itestを、編曲制約違反の出力も含めて追加評価する。"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
import math
import multiprocessing
from pathlib import Path
from statistics import mean

import torch

from data.pairdata.io import file_hash, read_pickle, write_json
from scripts.evaluate.compare_arrangement_models import read_evaluation_batch
from src.beam_muse.data import collate, load_cache, load_metadata
from src.beam_muse.experiment import load_checkpoint
from src.common.comparison import evaluate_output
from src.common.difficulty import DIFFICULTY_KEYS
from src.common.evaluation import EPSILON, summarize_control_metrics
from src.common.metrics import onset_similarities
from src.common.provenance import implementation_hashes, validate_sources
from src.musemorphose.generation import Grammar, generate


WORKER = {}
SCOPES = ('all_outputs', 'valid_only', 'invalid_only')


def read_json(path):
    return json.loads(Path(path).read_text())


def measure_output(prepared, candidate, scorer, row):
    """制約の成否と独立に測定。無出力・算出不能を0点や成功値で補完しない。"""
    result = {key: row[key] for key in ('case_id', 'source_id', 'status', 'reasons', 'target')}
    result.update(metrics=None, measurement_error=None)
    if candidate is None:
        result['measurement_error'] = 'no_output'
        return result
    try:
        difficulty = scorer.difficulty(candidate, prepared.previous)
        result['metrics'] = dict(difficulty=asdict(difficulty),
                                 **onset_similarities(prepared.original, candidate))
    except ValueError as error:
        result['measurement_error'] = str(error)
    return result


def assert_reproduced(saved, actual):
    """計測時間以外の従来評価が一致しない場合は混合集計を防ぐ。"""
    before = {k: v for k, v in saved.items() if k != 'elapsed_seconds'}
    after = {k: v for k, v in actual.items() if k != 'elapsed_seconds'}
    if before != after:
        differences = [k for k in before.keys() | after.keys() if before.get(k) != after.get(k)]
        raise ValueError(f"再生成が従来評価と不一致: {saved['case_id']} {differences}")


def neural_batch(start, end, path):
    model, vocabulary, scorer = (WORKER[k] for k in ('model', 'vocabulary', 'scorer'))
    grammar = Grammar(vocabulary)
    rows = []
    with torch.no_grad():
        for example, saved in zip(WORKER['examples'][start:end], WORKER['saved'][start:end]):
            batch = collate([example], vocabulary, scorer, WORKER['variant'], 'cpu', training=False)
            generated = generate(model, batch, example.prepared, vocabulary, grammar=grammar)
            actual = evaluate_output(example.prepared, generated.candidate, scorer,
                case_id=example.case_id, source_id=example.source_id,
                rule_count=len(example.history) if WORKER['variant'] == 'beam_muse' else 0,
                failure_reason=generated.failure_reason, config=WORKER['evaluation'])
            assert_reproduced(saved, actual)
            row = measure_output(example.prepared, generated.candidate, scorer, actual)
            row['candidate'] = asdict(generated.candidate) if generated.candidate is not None else None
            row['tokens'] = generated.tokens
            rows.append(row)
    write_json(path, rows)
    write_json(path.with_suffix('.receipt.json'), dict(start=start, end=end, sha256=file_hash(path)))
    return len(rows)


def summarize(rows, tolerance):
    """元のstatusを保ち、適格性とは独立した測定対象集合を明示する。"""
    result = {}
    for scope in SCOPES:
        subset = [r for r in rows if scope == 'all_outputs' or
                  r['status'] == ('valid' if scope == 'valid_only' else 'invalid')]
        measured = [r for r in subset if r['metrics'] is not None]
        # 既存の数式を再利用するための一時ビュー。保存するstatusは変更しない。
        metrics = summarize_control_metrics([dict(r, status='valid') for r in measured])
        errors = [max(abs(r['metrics']['difficulty']['normalized'][k] - r['target'][k])
                      for k in DIFFICULTY_KEYS) for r in measured]
        successes = sum(e <= tolerance + EPSILON for e in errors)
        within_one = sum(e <= 1 / 7 + EPSILON for e in errors)
        metrics.update(n=len(subset), measured_n=len(measured), unmeasurable_n=len(subset)-len(measured),
            status_counts={s: sum(r['status'] == s for r in subset) for s in ('valid', 'invalid', 'failed')},
            valid_rate=sum(r['status'] == 'valid' for r in subset)/len(subset) if subset else None,
            measurement_error_counts=dict(Counter(r['measurement_error'] for r in subset if r['measurement_error'])),
            target_success_ignoring_validity=dict(numerator=successes, denominator=len(subset),
                value=successes/len(subset) if subset else None),
            within_one_class_ignoring_validity=dict(numerator=within_one, denominator=len(subset),
                value=within_one/len(subset) if subset else None))
        result[scope] = metrics
    return result


def verify_valid_summary(actual, expected):
    for key in ('attribute_spearman', 'attribute_label_distance', 'label_distance',
                'rhythm_similarity', 'hihat_similarity'):
        if actual[key] != expected[key]:
            raise ValueError('従来のvalid集計と不一致: ' + key)
    if actual['status_counts']['valid'] != expected['status_counts']['valid']:
        raise ValueError('従来のvalid件数と不一致')


def neural_rows(root, model_id, split, output, workers, batch_size):
    family, name = model_id.split('/')
    directory = root/family
    checkpoint = directory/name/'best.pt'
    saved_path = directory/name/f'{split}_rows.json'
    saved = read_json(saved_path)
    manifest = read_json(directory/'cache/manifest.json')
    paths = manifest['resolved_paths']
    scorer, _, _, evaluation, _, vocabulary = load_metadata(paths['meta'], paths['beam_model'])
    examples, _ = load_cache(directory/'cache', split, paths['meta'], paths['beam_model'])
    if [r['case_id'] for r in saved] != [e.case_id for e in examples]:
        raise ValueError('保存済み評価と入力IDが不一致')
    old_identity = read_json(directory/name/f'{split}_evaluation.json')
    validate_sources(old_identity['sources'], implementation_hashes('beam_muse'))
    checkpoint_hash = file_hash(checkpoint)
    if old_identity['checkpoint_sha256'] != checkpoint_hash or old_identity['input_sha256'] != manifest['splits'][split]['sha256']:
        raise ValueError('従来評価の重み・入力ハッシュが不一致')
    if old_identity['device'] != 'cpu' or old_identity['torch'] != str(torch.__version__):
        raise ValueError('従来評価とCPU/PyTorch実行条件が不一致')
    model, state = load_checkpoint(checkpoint, vocabulary)
    cache = output/'including_invalid_cache'/model_id/split
    cache.mkdir(parents=True, exist_ok=True)
    identity = dict(checkpoint_sha256=checkpoint_hash, input_sha256=old_identity['input_sha256'],
        saved_rows_sha256=file_hash(saved_path), sources=implementation_hashes('beam_muse'),
        script_sha256=file_hash(__file__), batch_size=batch_size, device='cpu', threads=1,
        torch=str(torch.__version__), evaluation_config=asdict(evaluation))
    manifest_path = cache/'manifest.json'
    if manifest_path.exists() and read_json(manifest_path) != identity:
        raise ValueError('追加評価キャッシュの実装・入力が不一致: ' + str(cache))
    write_json(manifest_path, identity)
    WORKER.clear()
    WORKER.update(model=model, vocabulary=vocabulary, scorer=scorer, examples=examples,
                  saved=saved, evaluation=evaluation, variant=state['trial']['variant'])
    pending, batches = [], []
    for start in range(0, len(saved), batch_size):
        end = min(start+batch_size, len(saved))
        path = cache/f'batch_{start:06d}.json'
        receipt = path.with_suffix('.receipt.json')
        batches.append(path)
        if path.exists() and receipt.exists():
            if read_json(receipt) != dict(start=start, end=end, sha256=file_hash(path)):
                raise ValueError('追加評価バッチが破損: ' + str(path))
        else:
            pending.append((start, end, path))
    done = len(saved)-sum(end-start for start, end, _ in pending)
    print(f'{model_id} {split}: {done}/{len(saved)}件を再利用', flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork')) as pool:
            futures = [pool.submit(neural_batch, *args) for args in pending]
            for future in as_completed(futures):
                done += future.result()
                print(f'{model_id} {split}: {done}/{len(saved)}件', flush=True)
    rows = [r for path in batches for r in read_json(path)]
    if [r['case_id'] for r in rows] != [r['case_id'] for r in saved]:
        raise ValueError('再集計したID・順序が不一致')
    return rows, identity


def beam_rows(root, model_id, split, expected):
    family, name = model_id.split('/')
    if expected['status_counts'] != dict(valid=expected['n'], invalid=0, failed=0):
        raise ValueError('BeamSearchの不適格出力は候補からの追加測定が必要')
    paths = sorted((root/family/split/name).glob('batch_*.pkl.gz'))
    rows, inputs, hashes = [], None, {}
    for path in paths:
        digest = file_hash(path)
        if read_json(path.with_suffix('.json'))['sha256'] != digest:
            raise ValueError('BeamSearch保存バッチのハッシュ不一致')
        batch, inputs = read_evaluation_batch(path, inputs)
        hashes[str(path)] = digest
        for r in batch:
            rows.append({k: r[k] for k in ('case_id', 'source_id', 'status', 'reasons', 'target', 'metrics')} |
                        dict(measurement_error=None))
    if len(rows) != expected['n'] or len({r['case_id'] for r in rows}) != len(rows):
        raise ValueError('BeamSearch保存バッチの件数・IDが不一致')
    return rows, dict(mode='saved_valid_outputs', batch_sha256=hashes)


def csv_row(model_id, selected, scope, summary):
    row = dict(model_id=model_id, selected_on_val=selected, scope=scope, n=summary['n'],
               **summary['status_counts'], valid_rate=summary['valid_rate'],
               measured_n=summary['measured_n'], unmeasurable_n=summary['unmeasurable_n'])
    for key in DIFFICULTY_KEYS:
        corr = summary['attribute_spearman'][key]
        distance = summary['attribute_label_distance'][key]
        row.update({f'spearman_{key}': corr['coefficient'], f'spearman_{key}_n': corr['n'],
                    f'spearman_{key}_reason': corr['reason'], f'label_distance_{key}': distance['mean'],
                    f'label_distance_{key}_n': distance['n']})
    for key in ('label_distance', 'rhythm_similarity', 'hihat_similarity'):
        row[key], row[key+'_n'] = summary[key]['mean'], summary[key]['n']
    for key in ('target_success_ignoring_validity', 'within_one_class_ignoring_validity'):
        row[key] = summary[key]['value']
        row[key+'_numerator'] = summary[key]['numerator']
        row[key+'_denominator'] = summary[key]['denominator']
    return row


def write_csv(path, rows):
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('trained_model'))
    parser.add_argument('--output', type=Path, default=Path('output/eval-result'))
    parser.add_argument('--splits', nargs='+', choices=('test', 'itest'), default=['test', 'itest'])
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if min(args.workers, args.batch_size) < 1:
        parser.error('workersとbatch-sizeは正整数')
    torch.set_num_threads(1)
    selected = read_json(args.output/'manifest.json')['selected_models_on_val']
    for split in args.splits:
        reference_path = args.output/f'{split}_summary.json'
        reference = read_json(reference_path)
        summaries, provenance, csv_rows, all_ids = {}, {}, [], None
        for model_id, expected in reference.items():
            if model_id.startswith('beamsearch_v1/'):
                rows, identity = beam_rows(args.root, model_id, split, expected)
            else:
                rows, identity = neural_rows(args.root, model_id, split, args.output, args.workers, args.batch_size)
            ids = {r['case_id'] for r in rows}
            if len(ids) != len(rows) or (all_ids is not None and ids != all_ids):
                raise ValueError('モデル間で評価IDが不一致')
            all_ids = ids
            summary = summarize(rows, expected['evaluation_config']['target_tolerance'])
            verify_valid_summary(summary['valid_only'], expected)
            if summary['all_outputs']['status_counts'] != expected['status_counts']:
                raise ValueError('従来評価とstatus件数が不一致')
            family, name = model_id.split('/')
            summary.update(evaluation_config=expected['evaluation_config'],
                           selected_on_val=selected.get(family) == name,
                           original_target_success_rate=expected['target_success_rate'],
                           original_joint_success_rate=expected['joint_success_rate'])
            summaries[model_id], provenance[model_id] = summary, identity
            for scope in SCOPES:
                csv_rows.append(csv_row(model_id, summary['selected_on_val'], scope, summary[scope]))
            folder = args.output/'models'/model_id
            write_json(folder/f'{split}_including_invalid_rows.json', rows)
            write_json(folder/f'{split}_including_invalid_summary.json', summary)
            print(f'{model_id} {split}: 集計完了、算出可能{summary["all_outputs"]["measured_n"]}件', flush=True)
        write_json(args.output/f'{split}_including_invalid_summary.json', summaries)
        write_csv(args.output/f'{split}_including_invalid_summary.csv', [r for r in csv_rows if r['scope'] == 'all_outputs'])
        write_csv(args.output/f'{split}_valid_invalid_comparison.csv', csv_rows)
        write_json(args.output/f'{split}_including_invalid_manifest.json', dict(
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            reference_sha256=file_hash(reference_path), script_sha256=file_hash(__file__),
            definitions=dict(all_outputs='適格・不適格を問わず、難易度を算出できる出力の指標。算出不能の件数を別記。',
                valid_only='従来のvalid出力のみ。従来集計との一致を検証済み。',
                invalid_only='従来のinvalid出力のみ。適格状態は変更しない。',
                target_success_ignoring_validity='制約違反を不成功にせず難易度誤差だけで判定。無出力・算出不能は分母に残す。',
                regeneration='ニューラルは同じCPU・重み・入力でgreedy再生成し、時間以外の従来評価との一致を全件検証。',
                similarity='原譜・出力がともに空の発音集合は欠測。各指標のnを参照。'),
            models=provenance))


if __name__ == '__main__':
    main()
