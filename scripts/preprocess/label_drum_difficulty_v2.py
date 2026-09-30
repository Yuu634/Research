"""承認済み難易度v2を収集済みstandard pklへ付与する（再開可能）。"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import shutil
import sqlite3
import tempfile
import time
import zlib

import numpy as np

from src.data.drum_difficulty_v2 import (
    ATTRS, AUX, VERSION, decode_bars, discretize, score_piece, weighted_boundaries,
)
from src.utils.tokenization_mode import write_dataset_tokenization_metadata

SPLITS = ('train', 'val', 'test', 'itest')
ROOT = Path(__file__).resolve().parents[2]
WORKER = {}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def packed(value):
    return zlib.compress(pickle.dumps(value, protocol=5), level=1)


def unpacked(value):
    return pickle.loads(zlib.decompress(value))


def input_records(meta):
    chosen = {}
    for split in SPLITS:
        for filename in pickle.loads((meta/f'{split}_split.pkl').read_bytes()):
            if Path(filename).name != filename or not filename.endswith('.pkl'):
                raise ValueError('分割リストに不正なファイル名')
            key = filename[:-4]
            if key in chosen:
                raise ValueError('分割に重複ID')
            chosen[key] = split
    records = {}
    with (meta/'manifest.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            if r['id'] not in chosen:
                continue
            if r['token_status'] != 'ok' or r.get('split_exclusion') or r['split'] != chosen[r['id']]:
                raise ValueError(f'分割とmanifestが不一致: {r["id"]}')
            records[r['id']] = dict(id=r['id'], dataset=r['dataset'], split=r['split'],
                                    source_sha256=r['source_sha256'], token_sha256=r['token_sha256'],
                                    bars=r['bars'], source=r['source'], license=r['license'])
    if set(records) != set(chosen):
        raise ValueError('分割IDのmanifest記録が不足')
    return records


def semantic_hash(bar_positions, tokens):
    return digest(json.dumps([bar_positions, tokens], separators=(',', ':')).encode())


def initialize(input_dir, vocab, output_dir, boundaries):
    WORKER.update(input=Path(input_dir), output=Path(output_dir),
                  vocab=pickle.loads(Path(vocab).read_bytes())['idx2token'], boundaries=boundaries)


def score_worker(record):
    key = record['id']
    result = dict(id=key, dataset=record['dataset'], split=record['split'], status='error', bars=0)
    try:
        data = (WORKER['input']/f'{key}.pkl').read_bytes()
        value = pickle.loads(data)
        if len(value) != 2:
            raise ValueError('入力は未ラベルの2要素pklに限定')
        starts, indices = value
        result['input_sha256'] = digest(data)
        result['semantic_sha256'] = semantic_hash(starts, indices)
        if digest(np.asarray(indices, dtype='<i4').tobytes()) != record['token_sha256']:
            raise ValueError('収集時とトークン内容が不一致')
        bars = decode_bars(starts, indices, WORKER['vocab'])
        if len(bars) != record['bars']:
            raise ValueError('manifestと実小節数が不一致')
        scores, detail, flags = score_piece(bars)
        result.update(bars=len(bars), flags=flags)
        invalid = not np.isfinite(scores).all()
        result['status'] = 'excluded' if invalid else 'ok'
        result['reason'] = 'movement_infeasible' if invalid else None
        return result, packed((scores, detail))
    except Exception as e:
        result['reason'] = f'{type(e).__name__}: {e}'
        return result, None


def label_worker(task):
    info, blob = task
    key = info['id']
    data = (WORKER['input']/f'{key}.pkl').read_bytes()
    if digest(data) != info['input_sha256']:
        raise ValueError(f'算出後に入力変更: {key}')
    starts, tokens = pickle.loads(data)
    scores, _ = unpacked(blob)
    classes = discretize(scores, WORKER['boundaries'])
    labels = [dict(zip(ATTRS, map(int, row))) for row in classes]
    payload = pickle.dumps((starts, tokens, labels), protocol=4)
    path = WORKER['output']/f'{key}.pkl'
    tmp = path.with_suffix('.pkl.tmp')
    tmp.write_bytes(payload)
    tmp.replace(path)
    return key, digest(payload)


def validate_worker(task):
    info, blob, expected_hash = task
    key = info['id']
    payload = (WORKER['output']/f'{key}.pkl').read_bytes()
    if digest(payload) != expected_hash:
        raise ValueError(f'出力チェックサム不一致: {key}')
    starts, tokens, labels = pickle.loads(payload)
    if semantic_hash(starts, tokens) != info['semantic_sha256']:
        raise ValueError(f'原トークン・小節位置が変更された: {key}')
    bars = decode_bars(starts, tokens, WORKER['vocab'])
    scores, detail = unpacked(blob)
    if len(labels) != len(bars) or scores.shape != (len(bars), len(ATTRS)):
        raise ValueError(f'末尾を含む小節ラベル数不一致: {key}')
    if not np.isfinite(scores).all() or not np.isfinite(detail).all():
        raise ValueError(f'非有限値: {key}')
    for row, label in zip(scores, labels):
        if set(label) != set(ATTRS):
            raise ValueError('属性キーが不一致')
        for name, v in zip(ATTRS, row):
            expected = 0 if v == 0 else 1+sum(v > b for b in WORKER['boundaries'][name])
            if type(label[name]) is not int or label[name] != expected or not 0 <= expected < 8:
                raise ValueError(f'クラス値不一致: {key} {name}')
    # 固定IDで約1/1000を独立再計算する。全件はpkl内容・ラベル対応を検証する。
    recomputed = int(key) % 1000 == 0
    if recomputed:
        recalculated, components, _ = score_piece(bars)
        if not np.allclose(recalculated, scores, rtol=0, atol=1e-12) or not np.allclose(components, detail, rtol=0, atol=1e-12):
            raise ValueError(f'生スコアの再計算不一致: {key}')
    return key, len(bars), recomputed


def connect(meta):
    db = sqlite3.connect(meta/'scores.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS scores (id TEXT PRIMARY KEY, info TEXT, payload BLOB, output_hash TEXT)')
    return db


def pool(args, boundaries=None):
    return ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                               initargs=(args.input, args.source_meta/'drum_vocab.pkl', args.output, boundaries))


def score_all(args, records, db):
    done = {r[0] for r in db.execute('SELECT id FROM scores')}
    pending = [records[k] for k in sorted(records) if k not in done]
    start = time.monotonic()
    print(f'算出: {len(pending)}曲、再開済み{len(done)}曲', flush=True)
    with pool(args) as executor:
        for i, (info, blob) in enumerate(executor.map(score_worker, pending, chunksize=8), 1):
            db.execute('INSERT INTO scores VALUES (?,?,?,NULL)',
                       (info['id'], json.dumps(info, ensure_ascii=False), blob))
            if i % 128 == 0:
                db.commit()
            if i % 500 == 0 or i == len(pending):
                print(f'算出 {i+len(done)}/{len(records)} ({time.monotonic()-start:.0f}秒)', flush=True)
    db.commit()
    actual = {r[0] for r in db.execute('SELECT id FROM scores')}
    if actual != set(records):
        raise ValueError('算出IDの不足・余分')


def fit_bounds(args, records, db):
    if db.execute('SELECT COUNT(*) FROM scores').fetchone()[0] != len(records):
        raise ValueError('全曲の算出後に境界を推定してください')
    counts = Counter()
    status = Counter()
    for (raw,) in db.execute('SELECT info FROM scores'):
        info = json.loads(raw)
        status[(info['dataset'], info['split'], info['status'])] += 1
        if info['status'] == 'ok' and info['split'] == 'train':
            counts[info['dataset']] += 1
    if set(counts) != {r['dataset'] for r in records.values()}:
        raise ValueError('有効trainがないデータセット')
    weights = [defaultdict(float) for _ in ATTRS]
    for raw, blob in db.execute('SELECT info,payload FROM scores ORDER BY id'):
        info = json.loads(raw)
        if info['status'] != 'ok' or info['split'] != 'train':
            continue
        scores, _ = unpacked(blob)
        weight = 1/(len(counts)*counts[info['dataset']]*len(scores))
        for j in range(len(ATTRS)):
            values, multiplicities = np.unique(scores[:, j][scores[:, j] > 0], return_counts=True)
            for value, multiplicity in zip(values, multiplicities):
                weights[j][float(value)] += float(multiplicity)*weight
    boundaries = {name: weighted_boundaries(w) for name, w in zip(ATTRS, weights)}
    meta = dict(version=VERSION, boundaries=boundaries, positive_training_unique_values={
        name: len(w) for name, w in zip(ATTRS, weights)}, train_pieces=dict(counts),
        fitting_split='train', weighting='dataset_equal/piece_equal/bar_equal',
        zero_class=0, positive_classes=[1,7], quantile_method='inverse_weighted_empirical_cdf',
        status_counts=[dict(dataset=a, split=b, status=c, count=n) for (a,b,c),n in sorted(status.items())])
    dump_json(args.meta/'bounds.json', meta)
    print('trainのみで境界確定:', dict(counts), flush=True)
    return boundaries


def label_all(args, db, boundaries):
    tasks = []
    for raw, blob, output_hash in db.execute('SELECT info,payload,output_hash FROM scores ORDER BY id'):
        info = json.loads(raw)
        if info['status'] == 'ok' and output_hash is None:
            tasks.append((info, blob))
    print(f'pkl生成: {len(tasks)}曲', flush=True)
    start = time.monotonic()
    with pool(args, boundaries) as executor:
        for i, (key, checksum) in enumerate(executor.map(label_worker, tasks, chunksize=8), 1):
            db.execute('UPDATE scores SET output_hash=? WHERE id=?', (checksum, key))
            if i % 128 == 0:
                db.commit()
            if i % 1000 == 0 or i == len(tasks):
                print(f'pkl生成 {i}/{len(tasks)} ({time.monotonic()-start:.0f}秒)', flush=True)
    db.commit()


def publish_metadata(args, records, db, bounds):
    splits = defaultdict(list)
    histogram = defaultdict(lambda: np.zeros((len(ATTRS),8), dtype=np.int64))
    flags = Counter()
    bars_by = Counter()
    sensitivity = defaultdict(lambda: [0.,0.,0.,0])
    norm_stats = {name: [math_inf(), 0., 0.] for name in ATTRS}
    total_bars = 0
    with (args.meta/'manifest.jsonl').open('w') as f:
        for raw, blob, output_hash in db.execute('SELECT info,payload,output_hash FROM scores ORDER BY id'):
            info = json.loads(raw)
            info['source_metadata'] = records[info['id']]
            info['output_sha256'] = output_hash
            f.write(json.dumps(info, ensure_ascii=False)+'\n')
            flags.update(issue for _, issue in info.get('flags', []))
            if info['status'] != 'ok':
                continue
            if not output_hash:
                raise ValueError('生成されていないpkl')
            name = info['id']+'.pkl'
            splits[info['split']].append(name)
            splits[info['dataset']+'_'+info['split']].append(name)
            scores, detail = unpacked(blob)
            labels = discretize(scores, bounds)
            group = info['dataset']+'/'+info['split']
            bars_by[group] += len(scores)
            total_bars += len(scores)
            for j, attr in enumerate(ATTRS):
                histogram[group][j] += np.bincount(labels[:, j], minlength=8)
                norm_stats[attr][0] = min(norm_stats[attr][0], float(scores[:, j].min()))
                norm_stats[attr][1] = max(norm_stats[attr][1], float(scores[:, j].max()))
                norm_stats[attr][2] += float(scores[:, j].sum())
            for rate_name, col in [('hand',4),('kick',8),('pedal',12)]:
                d,q80,q90,peak = detail[:,col:col+4].T
                row = sensitivity[rate_name]
                row[0] += float(np.maximum(d,q80).sum())
                row[1] += float(np.maximum(d,q90).sum())
                row[2] += float(np.maximum(d,peak).sum())
                row[3] += len(scores)
    for split in SPLITS:
        splits.setdefault(split, [])
    for name, files in splits.items():
        (args.meta/f'{name}_split.pkl').write_bytes(pickle.dumps(files, protocol=4))
    shutil.copyfile(args.source_meta/'drum_vocab.pkl', args.meta/'drum_vocab.pkl')
    write_dataset_tokenization_metadata(str(args.output), 'standard',
        dict(difficulty_version=VERSION, attributes=list(ATTRS), model_conditioned_attributes=list(ATTRS[:5])))
    dump_json(args.meta/'summary.json', dict(version=VERSION, input_pieces=len(records),
        output_pieces=sum(len(splits[s]) for s in SPLITS), output_bars=total_bars,
        split_counts={k:len(v) for k,v in sorted(splits.items())}, bar_counts=dict(bars_by),
        audit_flags=dict(flags), continuous_stats={a:dict(min=v[0], max=v[1], mean=v[2]/total_bars) for a,v in norm_stats.items()},
        class_histograms={g:{a:row.tolist() for a,row in zip(ATTRS,v)} for g,v in histogram.items()},
        speed_sensitivity={name:dict(mean_q80=v[0]/v[3],mean_q90=v[1]/v[3],mean_max=v[2]/v[3]) for name,v in sensitivity.items()},
        rhythm_status='検証用の反復性ラベル。現行モデルの条件入力には使われない。'))


def math_inf():
    return float('inf')


def validate_all(args, records, db, boundaries):
    tasks = []
    for raw, blob, checksum in db.execute('SELECT info,payload,output_hash FROM scores ORDER BY id'):
        info = json.loads(raw)
        if info['status'] == 'ok':
            if not checksum:
                raise ValueError('pkl未生成')
            tasks.append((info, blob, checksum))
    expected = {r[0]['id']+'.pkl' for r in tasks}
    actual = {p.name for p in args.output.glob('*.pkl')}
    if actual != expected:
        raise ValueError('出力pkl一覧の不足・余分')
    seen = set()
    for split in SPLITS:
        ids = pickle.loads((args.meta/f'{split}_split.pkl').read_bytes())
        if len(ids) != len(set(ids)) or seen & set(ids):
            raise ValueError('分割間重複')
        for name in ids:
            if records[name[:-4]]['split'] != split:
                raise ValueError('元分割からの移動')
        seen.update(ids)
    if seen != expected:
        raise ValueError('出力と分割リストが不一致')
    bars = recomputed = 0
    start = time.monotonic()
    with pool(args, boundaries) as executor:
        for i, (_, n, recalculated) in enumerate(executor.map(validate_worker, tasks, chunksize=8), 1):
            bars += n
            recomputed += recalculated
            if i % 1000 == 0 or i == len(tasks):
                print(f'全件検証 {i}/{len(tasks)} ({time.monotonic()-start:.0f}秒)', flush=True)
    dump_json(args.meta/'validation.json', dict(all_output_payloads_checked=True, pieces=len(tasks), bars=bars,
        independently_recomputed_pieces=recomputed, original_tokens_preserved=True, split_disjoint=True,
        original_split_preserved=True, last_bar_labels_checked=True, finite_raw_scores=True,
        class_values_checked=True, output_checksums_checked=True, bounds_sha256=file_hash(args.meta/'bounds.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/tokens/drum_collection_v1'))
    parser.add_argument('--source-meta', type=Path, default=Path('data/metadata/drum_collection_v1'))
    parser.add_argument('--output', type=Path, default=Path('data/processed/drum_collection_difficulty_v2'))
    parser.add_argument('--meta', type=Path, default=Path('data/metadata/drum_collection_difficulty_v2'))
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--stage', choices=['score','bounds','label','validate','all'], default='all')
    parser.add_argument('--limit', type=int, help='動作確認用。別の出力先を指定し、各dataset/splitから最大N曲')
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error('workersとlimitは正数')
    if args.input.resolve() == args.output.resolve() or args.meta.resolve() == args.source_meta.resolve():
        parser.error('入力の上書きは禁止')
    for path in (args.output, args.meta, ROOT/'output/revise_attribute_v2/tmp'):
        path.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(ROOT/'output/revise_attribute_v2/tmp')
    records = input_records(args.source_meta)
    if args.limit:
        counts = Counter()
        limited = {}
        for key, r in sorted(records.items()):
            group = r['dataset'], r['split']
            if counts[group] < args.limit:
                limited[key] = r
                counts[group] += 1
        records = limited
    identity = dict(version=VERSION, input=str(args.input.resolve()), source_meta=str(args.source_meta.resolve()),
        output=str(args.output.resolve()), limit=args.limit, attributes=list(ATTRS), components=list(AUX),
        formula_sha256=file_hash(ROOT/'src/data/drum_difficulty_v2.py'),
        pipeline_sha256=file_hash(__file__), python=platform.python_version(), numpy=np.__version__,
        v1_constants_sha256=file_hash(ROOT/'src/data/drum_difficulty_calculator.py'),
        vocab_sha256=file_hash(args.source_meta/'drum_vocab.pkl'),
        manifest_sha256=file_hash(args.source_meta/'manifest.jsonl'),
        splits_sha256={s:file_hash(args.source_meta/f'{s}_split.pkl') for s in SPLITS})
    ipath = args.meta/'identity.json'
    if ipath.exists() and json.loads(ipath.read_text()) != identity:
        raise ValueError('コード・設定・入力が既存出力と異なるため、別の出力先を使用してください')
    dump_json(ipath, identity)
    with connect(args.meta) as db:
        if args.stage in ('score','all'):
            score_all(args, records, db)
        if args.stage in ('bounds','all'):
            bounds = fit_bounds(args, records, db)
        elif args.stage != 'score':
            bounds = json.loads((args.meta/'bounds.json').read_text())['boundaries']
        if args.stage in ('label','all'):
            label_all(args, db, bounds)
            publish_metadata(args, records, db, bounds)
        if args.stage in ('validate','all'):
            validate_all(args, records, db, bounds)


if __name__ == '__main__':
    main()
