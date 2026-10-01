"""valでBeamSearch候補を選び、固定後にtest/itestを評価する。バッチ単位で再開可能。"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing
from pathlib import Path
import pickle
from time import perf_counter

from data.pairdata.io import file_hash, iter_pairs, read_pickle, write_json, write_pickle
from scripts.evaluate.review_arrangement_metrics import fixed_rules
from src.arrangement.beam_search import SearchConfig, beam_search, standard_actions
from src.arrangement.evaluation import EvaluationConfig, evaluate_result, summarize_results
from src.arrangement.inputs import prepare_input, request_from_pair
from src.arrangement.scoring import ScoreConfig, ScoringModel

WORKER = {}
SOURCE_FILES = [f'src/arrangement/{name}.py' for name in (
    'representation', 'rules', 'safety', 'scoring', 'inputs', 'evaluation', 'beam_search')]
SOURCE_FILES += ['src/data/drum_difficulty_v2.py', 'src/data/drum_difficulty_calculator.py',
                 'scripts/evaluate/review_arrangement_metrics.py', __file__]
CONTROLS = [dict(name='no_edit', control=True), dict(name='hh_then_ghost', control=True)]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def load_candidates(path):
    content = json.loads(Path(path).read_text())
    if set(content) != {'schema_version', 'action_set', 'evaluation', 'candidates'} or content['schema_version'] != 1:
        raise ValueError('比較設定の形式が不正')
    if content['action_set'] != 'standard_v1' or not content['candidates']:
        raise ValueError('未対応のアクション集合または候補なし')
    EvaluationConfig(**content['evaluation']).validate()
    names = {item['name'] for item in CONTROLS}
    for candidate in content['candidates']:
        if set(candidate) != {'name', 'score', 'search'}:
            raise ValueError('候補の形式が不正')
        name = candidate['name']
        if not isinstance(name, str) or not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in name) or name in names:
            raise ValueError('候補名が不正・重複')
        names.add(name)
        ScoreConfig(**candidate['score']).validate()
        SearchConfig(**candidate['search']).validate()
    return content


def initialize(meta, input_path, evaluation):
    meta = Path(meta)
    vocab = pickle.loads((meta/'drum_vocab.pkl').read_bytes())
    WORKER.update(bounds=json.loads((meta/'bounds.json').read_text())['boundaries'],
                  vocabulary=vocab['token2idx'], inputs=read_pickle(input_path),
                  evaluation=EvaluationConfig(**evaluation).validate(), actions=standard_actions())


def run_batch(method, start, end, path):
    model = ScoringModel(WORKER['bounds'], WORKER['vocabulary'], ScoreConfig(**method.get('score', {})))
    rows = []
    for item in WORKER['inputs'][start:end]:
        prepared = item['prepared']
        started = perf_counter()
        if method.get('control'):
            names = () if method['name'] == 'no_edit' else ('hh_16_to_8', 'remove_ghost_notes')
            candidate, count, trace = fixed_rules(prepared, names)
            elapsed = perf_counter()-started
            history = [dict(name=entry['rule'], parameters={}) for entry in trace if entry['status'] == 'applied']
            statistics = dict(trace=trace)
        else:
            result = beam_search(prepared, model, SearchConfig(**method['search']), WORKER['actions'])
            candidate, count, elapsed = result.candidate, len(result.history), result.elapsed_seconds
            history, statistics = [action.as_dict() for action in result.history], result.statistics
        row = evaluate_result(prepared, candidate, model, case_id=item['case_id'], source_id=item['source_id'],
                              rule_count=count, elapsed_seconds=elapsed, config=WORKER['evaluation'])
        row.update(history=history, search_statistics=statistics, candidate=asdict(candidate))
        rows.append(row)
    write_pickle(path, rows)
    receipt = dict(method=method['name'], start=start, end=end, n=len(rows), sha256=file_hash(path))
    write_json(Path(path).with_suffix('.json'), receipt)
    return receipt


def prepare_split(args, split, model, vocab, limit):
    items, exclusions, identities = [], [], set()
    for i, pair in enumerate(iter_pairs(args.pairdata, split)):
        if limit is not None and i >= limit:
            break
        case_id = pair['pair_id']
        if case_id in identities or pair['source']['split'] != split:
            raise ValueError('ペアID重複または分割不一致')
        identities.add(case_id)
        source_id = pair['source']['dataset']+':'+pair['source']['id']
        try:
            request = request_from_pair(pair, vocab['idx2token'])
            prepared = prepare_input(request, model)
        except ValueError as error:
            exclusions.append(dict(case_id=case_id, source_id=source_id, reason=str(error)))
            continue
        items.append(dict(case_id=case_id, source_id=source_id, input_sha256=fingerprint(request), prepared=prepared))
    if not items:
        raise ValueError('適格な評価入力がない')
    directory = args.output/split
    directory.mkdir(parents=True, exist_ok=True)
    write_pickle(directory/'inputs.pkl.gz', items)
    manifest = dict(split=split, considered=len(identities), accepted=len(items), excluded=len(exclusions),
                    inputs_sha256=file_hash(directory/'inputs.pkl.gz'),
                    exclusion_reasons=dict(Counter(row['reason'] for row in exclusions)), exclusions=exclusions,
                    inputs=[{key: row[key] for key in ('case_id', 'source_id', 'input_sha256')} for row in items])
    write_json(directory/'manifest.json', manifest)
    return manifest


def evaluate_split(args, split, methods, config, model, vocab, limit=None):
    directory = args.output/split
    input_path = directory/'inputs.pkl.gz'
    if (directory/'manifest.json').exists() and input_path.exists():
        manifest = json.loads((directory/'manifest.json').read_text())
        if file_hash(input_path) != manifest['inputs_sha256']:
            raise ValueError('再開入力のチェックサム不一致')
    else:
        manifest = prepare_split(args, split, model, vocab, limit)
    n = manifest['accepted']
    print(f'{split}: {n}/{manifest["considered"]}入力、除外{manifest["excluded"]}、{len(methods)}方式', flush=True)
    tasks, cached = [], []
    for method in methods:
        folder = directory/method['name']
        folder.mkdir(parents=True, exist_ok=True)
        for start in range(0, n, args.batch_size):
            end = min(start+args.batch_size, n)
            path = folder/f'batch_{start:06d}.pkl.gz'
            receipt_path = path.with_suffix('.json')
            if path.exists() and receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if (receipt['method'], receipt['start'], receipt['end'], receipt['sha256']) != (method['name'], start, end, file_hash(path)):
                    raise ValueError('再開バッチの整合性が不一致')
                cached.append(receipt)
            else:
                tasks.append((method, start, end, path))
    if tasks:
        # forkは数値ライブラリのスレッド数を1にしたCLI実行を想定する。
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('fork'),
                                 initializer=initialize, initargs=(args.meta, input_path, config['evaluation'])) as pool:
            futures = [pool.submit(run_batch, *task) for task in tasks]
            completed = 0
            for future in as_completed(futures):
                future.result()
                completed += 1
                if completed % 20 == 0 or completed == len(tasks):
                    print(f'{split}: {completed}/{len(tasks)}バッチ完了（再利用{len(cached)}）', flush=True)
    summaries = {}
    for method in methods:
        rows = []
        for start in range(0, n, args.batch_size):
            rows.extend(read_pickle(directory/method['name']/f'batch_{start:06d}.pkl.gz'))
        if [r['case_id'] for r in rows] != [r['case_id'] for r in manifest['inputs']]:
            raise ValueError('集計対象が共通入力集合と一致しない')
        summaries[method['name']] = summarize_results(rows)
    write_json(directory/'summaries.json', summaries)
    return summaries


def selection_key(name, summary):
    macro = summary['source_macro']
    return (-macro['joint_success_rate']['mean'], macro['failure_adjusted_mae']['mean'],
            summary['valid_output_means']['edit_distance']['mean'] if summary['valid_output_means']['edit_distance']['mean'] is not None else float('inf'),
            summary['rule_count_valid']['mean'] if summary['rule_count_valid']['mean'] is not None else float('inf'),
            summary['runtime_seconds']['mean'] if summary['runtime_seconds']['mean'] is not None else float('inf'), name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('smoke', 'select', 'test', 'all'), required=True)
    parser.add_argument('--config', type=Path, default=Path('config/beamsearch_candidates.json'))
    parser.add_argument('--meta', type=Path, default=Path('data/metadata/drum_collection_difficulty_v2'))
    parser.add_argument('--pairdata', type=Path, default=Path('data/processed/drum_pairdata_v1'))
    parser.add_argument('--output', type=Path, default=Path('output/beamsearch_step5'))
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--limit', type=int, default=16, help='smokeのtrain件数のみ。本評価では全件使用')
    args = parser.parse_args()
    try:
        if min(args.workers, args.batch_size, args.limit) < 1:
            raise ValueError('件数・並列数・バッチサイズは正整数')
        config = load_candidates(args.config)
        args.output.mkdir(parents=True, exist_ok=True)
        vocab = pickle.loads((args.meta/'drum_vocab.pkl').read_bytes())
        if vocab['tokenization_method'] != 'standard':
            raise ValueError('standard語彙が必要')
        model = ScoringModel(json.loads((args.meta/'bounds.json').read_text())['boundaries'], vocab['token2idx'])
        # 選定設定・実装・入力ファイルが異なる実行結果を再利用しない。
        index = json.loads((args.pairdata/'index.json').read_text())
        for shard in index['shards']:
            if file_hash(args.pairdata/shard['path']) != shard['sha256']:
                raise ValueError('ペアデータのチェックサム不一致')
        receipt = dict(config=config, sources={str(Path(p).relative_to(Path.cwd())) if Path(p).is_absolute() else p: file_hash(p) for p in SOURCE_FILES},
                       bounds_sha256=file_hash(args.meta/'bounds.json'), vocabulary_sha256=file_hash(args.meta/'drum_vocab.pkl'),
                       pair_index_sha256=file_hash(args.pairdata/'index.json'), batch_size=args.batch_size,
                       actions=[action.as_dict() for action in standard_actions()],
                       smoke_limit=args.limit if args.phase == 'smoke' else None)
        receipt_path = args.output/'run_manifest.json'
        if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
            raise ValueError('実装・設定・入力が以前の実行と異なる。別の出力先を指定してください')
        write_json(receipt_path, receipt)
        methods = CONTROLS+config['candidates']
        if args.phase == 'smoke':
            evaluate_split(args, 'train', methods, config, model, vocab, args.limit)
            return
        selected_path = args.output/'selected_model.json'
        if args.phase in ('select', 'all'):
            summaries = evaluate_split(args, 'val', methods, config, model, vocab)
            ranked = sorted(config['candidates'], key=lambda c: selection_key(c['name'], summaries[c['name']]))
            selected = dict(schema_version=1, selected_on='val', candidate=ranked[0],
                            evaluation=config['evaluation'], action_set=config['action_set'],
                            actions=receipt['actions'], bounds_sha256=receipt['bounds_sha256'],
                            vocabulary_sha256=receipt['vocabulary_sha256'], run_manifest_sha256=file_hash(receipt_path),
                            ranking=[c['name'] for c in ranked])
            if selected_path.exists() and json.loads(selected_path.read_text()) != selected:
                raise ValueError('確定済みの選定結果を変更しない')
            write_json(selected_path, selected)
            print('valで選定: '+selected['candidate']['name'], flush=True)
        if args.phase in ('test', 'all'):
            selected = json.loads(selected_path.read_text())
            if selected['run_manifest_sha256'] != file_hash(receipt_path) or selected['selected_on'] != 'val':
                raise ValueError('valで固定したモデルと実行記録が一致しない')
            for split in ('test', 'itest'):
                evaluate_split(args, split, CONTROLS+[selected['candidate']], config, model, vocab)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
