"""trained_modelの保存済みモデルを再学習せずtest/itest評価する。CPU並列・バッチ再開対応。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch

from data.pairdata.io import file_hash, read_pickle, write_json, write_pickle
from scripts.evaluate.compare_arrangement_models import CONTROLS, evaluate_split
from src.beam_muse.data import load_cache, load_metadata
from src.beam_muse.experiment import evaluate, load_checkpoint
from src.common.comparison import summarize_outputs
from src.common.loss_logging import MetricMeans, append_loss
from src.common.provenance import implementation_hashes, validate_sources

WORKER = {}


class CaptureLog:
    def write(self, phase, **record):
        self.record = dict(phase=phase, **record)


def neural_batch(start, end, path):
    log = CaptureLog()
    examples = WORKER['examples'][start:end]
    _, rows = evaluate(WORKER['model'], examples, WORKER['vocabulary'], WORKER['scorer'],
                       WORKER['variant'], 'cpu', WORKER['evaluation'], loss_log=log,
                       phase=WORKER['split'], epoch=WORKER['epoch'])
    write_pickle(path, dict(rows=rows, log=log.record))
    write_json(path.with_suffix('.json'), dict(start=start, end=end, sha256=file_hash(path)))
    return len(rows)


def aggregate_neural(batches):
    rows, kl, beam = [], MetricMeans(), MetricMeans()
    rec_sum, tokens = 0., 0
    for batch in batches:
        rows.extend(batch['rows'])
        record = batch['log']
        tokens += record['tokens']
        rec_sum += record['musemorphose']['reconstruction'] * record['tokens']
        kl.add({key: record['musemorphose'][key] for key in ('kl', 'kl_raw')}, record['n'])
        if record['beamsearch'] is not None:
            beam.add(record['beamsearch'], record['n'])
    summary = summarize_outputs(rows)
    summary['validation_reconstruction'] = rec_sum / tokens
    means = kl.means() | dict(reconstruction=rec_sum / tokens)
    means['total'] = means['reconstruction'] + .01 * means['kl'] if means['kl'] is not None else None
    return summary, rows, dict(musemorphose=means, beamsearch=beam.means() or None, tokens=tokens)


def evaluate_checkpoint(root, checkpoint, split, workers, batch_size):
    record = json.loads((root/'run.json').read_text())
    identity = record['identity']
    validate_sources(identity['sources'], implementation_hashes('beam_muse'))
    manifest = json.loads((root/'cache/manifest.json').read_text())
    paths = manifest['resolved_paths']
    scorer, _, _, evaluation, _, vocabulary = load_metadata(paths['meta'], paths['beam_model'])
    examples, _ = load_cache(root/'cache', split, paths['meta'], paths['beam_model'])
    completed = json.loads((checkpoint.parent/'completed.json').read_text())
    if completed['identity'] != identity or completed['checkpoint_sha256'] != file_hash(checkpoint):
        raise ValueError('学習完了記録と重みが不一致: ' + str(checkpoint))
    cache = checkpoint.parent/'evaluation_cache'/split
    cache.mkdir(parents=True, exist_ok=True)
    evaluation_identity = dict(checkpoint_sha256=file_hash(checkpoint), input_sha256=manifest['splits'][split]['sha256'],
                               sources=implementation_hashes('beam_muse'), batch_size=batch_size,
                               torch=str(torch.__version__), device='cpu', threads=1)
    cache_manifest = cache/'manifest.json'
    if cache_manifest.exists() and json.loads(cache_manifest.read_text()) != evaluation_identity:
        raise ValueError('再評価キャッシュの実装・重み・入力が変更された: ' + str(cache))
    write_json(cache_manifest, evaluation_identity)
    model, saved = load_checkpoint(checkpoint, vocabulary, identity, 'cpu')
    WORKER.clear()
    WORKER.update(model=model, vocabulary=vocabulary, scorer=scorer, evaluation=evaluation,
                  examples=examples, variant=saved['trial']['variant'], split=split, epoch=saved['epoch'])
    pending, paths, cached = [], [], 0
    for start in range(0, len(examples), batch_size):
        end = min(start + batch_size, len(examples))
        path = cache/f'batch_{start:06d}.pkl.gz'
        receipt = path.with_suffix('.json')
        paths.append(path)
        if path.exists() and receipt.exists():
            if json.loads(receipt.read_text()) != dict(start=start, end=end, sha256=file_hash(path)):
                raise ValueError('再評価バッチが破損: ' + str(path))
            cached += end-start
        else:
            pending.append((start, end, path))
    print(f'{checkpoint.parent}: {split} {len(examples)}件、再利用{cached}件', flush=True)
    started, done = perf_counter(), cached
    if pending:
        # CUDAを初期化せず、CPUモデルと入力をforkで共有。各プロセスの数値演算は1スレッド。
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork')) as pool:
            futures = [pool.submit(neural_batch, *task) for task in pending]
            for future in as_completed(futures):
                done += future.result()
                print(f'{checkpoint.parent.name} {split}: {done}/{len(examples)}件 '
                      f'({perf_counter()-started:.0f}秒)', flush=True)
    summary, rows, losses = aggregate_neural(read_pickle(path) for path in paths)
    if [row['case_id'] for row in rows] != [e.case_id for e in examples]:
        raise ValueError('評価対象と保存結果のIDが不一致')
    write_json(checkpoint.parent/f'{split}_summary.json', summary)
    write_json(checkpoint.parent/f'{split}_rows.json', rows)
    write_json(checkpoint.parent/f'{split}_evaluation.json', evaluation_identity | dict(n=len(rows)))
    append_loss(checkpoint.parent/'eval_loss.log', dict(phase=split, event='evaluation_summary',
                epoch=saved['epoch'], step=None, n=len(rows), beta=.01, free_bits=0., evaluation=summary, **losses))
    return summary, rows


def refresh_neural(root, args, status):
    selection_path = root/'selection.json'
    selected = json.loads(selection_path.read_text())['winners'] if selection_path.exists() else {}
    checkpoints = sorted(root.glob('*/best.pt'))
    if args.scope == 'selected':
        checkpoints = [root/winner['name']/'best.pt' for winner in selected.values()]
    variant = getattr(args, 'variant', None)
    if variant is not None:
        checkpoints = [path for path in checkpoints if (path.parent/'completed.json').exists()
                       and json.loads((path.parent/'completed.json').read_text())['trial']['variant'] == variant]
    if not checkpoints:
        status[str(root)] = dict(status='unavailable', reason='評価可能なチェックポイント・選定結果なし')
        return
    for split in args.splits:
        all_path = root/f'{split}_all_models.json'
        all_summaries = json.loads(all_path.read_text()) if all_path.exists() else {}
        comparison_path = root/f'{split}_comparison.json'
        comparison = json.loads(comparison_path.read_text()) if comparison_path.exists() else {}
        for checkpoint in checkpoints:
            if not (checkpoint.parent/'completed.json').exists():
                status[str(checkpoint.parent)] = dict(status='unavailable', reason='学習完了記録なし')
                continue
            summary, rows = evaluate_checkpoint(root, checkpoint, split, args.workers, args.batch_size)
            all_summaries[checkpoint.parent.name] = summary
            write_json(all_path, all_summaries)
            for variant, winner in selected.items():
                if checkpoint.parent.name == winner['name']:
                    comparison[variant] = summary
                    write_json(root/f'{split}_{variant}_rows.json', rows)
            write_json(comparison_path, comparison)
            status[f'{checkpoint.parent}/{split}'] = dict(status='complete', n=summary['n'],
                result=str(checkpoint.parent/f'{split}_summary.json'))
            write_json(args.root/'test_evaluation_status.json', status)


def refresh_beam(root, args, status):
    record = json.loads((root/'run_manifest.json').read_text())
    validate_sources(record['sources'], implementation_hashes('beamsearch'))
    selected = json.loads((root/'selected_model.json').read_text())
    methods = CONTROLS + (record['config']['candidates'] if args.scope == 'all' else [selected['candidate']])
    settings = json.loads((root/selected['candidate']['name']/'config.json').read_text())
    meta, pairdata = Path(settings['meta']), Path(settings['pairdata'])
    for key, path in (('bounds_sha256', meta/'bounds.json'), ('vocabulary_sha256', meta/'drum_vocab.pkl'),
                      ('pair_index_sha256', pairdata/'index.json')):
        if file_hash(path) != record[key]:
            raise ValueError('BeamSearch評価入力のハッシュ不一致: ' + key)
    from src.beamsearch.scoring import ScoringModel
    import pickle
    vocab = pickle.loads((meta/'drum_vocab.pkl').read_bytes())
    scorer = ScoringModel(json.loads((meta/'bounds.json').read_text())['boundaries'], vocab['token2idx'])
    options = SimpleNamespace(output=root, meta=meta, pairdata=pairdata, batch_size=record['batch_size'],
                              workers=args.workers, log_interval=500)
    for split in args.splits:
        summaries = evaluate_split(options, split, methods, record['config'], scorer, vocab)
        for name, summary in summaries.items():
            path = root/name/f'{split}_summary.json'
            write_json(path, summary)
            status[f'{root/name}/{split}'] = dict(status='complete', n=summary['n'], result=str(path))
        write_json(args.root/'test_evaluation_status.json', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('trained_model'))
    parser.add_argument('--scope', choices=('all', 'selected'), default='all', help='全候補、またはval選定済みモデル')
    parser.add_argument('--splits', nargs='+', choices=('test', 'itest'), default=['test', 'itest'])
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=16, help='ニューラル再評価の保存単位')
    args = parser.parse_args()
    if min(args.workers, args.batch_size) < 1:
        parser.error('workersとbatch-sizeは正整数')
    torch.set_num_threads(1)
    status_path = args.root/'test_evaluation_status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    for root in sorted(args.root.iterdir(), key=lambda p: ((p/'run_manifest.json').exists(), p.name)):
        if not root.is_dir():
            continue
        if (root/'run_manifest.json').exists():
            refresh_beam(root, args, status)
        elif (root/'run.json').exists():
            refresh_neural(root, args, status)
        write_json(status_path, status)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
