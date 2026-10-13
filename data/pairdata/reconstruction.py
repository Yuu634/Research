"""検証済み編曲ペアを保持し、元小節の自己再構成ペアを追加する。"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import pickle
import shutil

import numpy as np

from data.pairdata.core import canonical_hash, pattern_hash
from data.pairdata.io import file_hash, iter_pairs, read_pickle, write_json, write_pickle
from src.data.drum_difficulty_v2 import ATTRS, decode_bars, discretize, score_bar

VERSION = 'drum_pairdata_plus_reconst_v1'
SPLITS = ('train', 'val', 'test', 'itest')
METADATA = ('bounds.json', 'drum_vocab.pkl', 'sources.jsonl')


def anchor_key(row):
    source = row['source']
    return (source['dataset'], source['id'], source['before_bar'])


def reconstruction_pair(row):
    """元の文脈でbeforeを再構成する。別小節だったafterの来歴も置き換える。"""
    result = deepcopy(row)
    result.update(task_type='reconstruction', parent_pair_id=row['pair_id'],
                  pair_id='reconst:' + canonical_hash(anchor_key(row)),
                  X_after=deepcopy(row['X_before']), after_pattern=row['before_pattern'],
                  d_after=deepcopy(row['d_before']), delta_d=[0.] * 5,
                  raw_after=deepcopy(row['raw_before']),
                  difficulty_after=deepcopy(row['difficulty_before']),
                  original_after_raw=deepcopy(row['raw_before']),
                  original_after_classes=deepcopy(row['difficulty_before']),
                  q=1., similarities=dict(rhythm=1., skeleton=1., context=1.), difficulty_distance=0.)
    result['source']['after_bar'] = row['source']['before_bar']
    result['context']['original_after_previous'] = deepcopy(row['context']['previous'])
    result['context']['original_after_following'] = deepcopy(row['context']['following'])
    return result


def check_reconstruction(row, vocabulary, boundaries):
    """コピーしたラベルを信用せず、前小節付きで6属性を再計算する。"""
    if row['X_before'] != row['X_after'] or row['d_before'] != row['d_after'] or row['delta_d'] != [0.] * 5:
        raise ValueError('自己再構成の音符列・目標・差分が不正')
    before = decode_bars([0], row['X_before'], vocabulary)[0]
    previous_tokens = row['context']['previous']
    previous = decode_bars([0], previous_tokens, vocabulary)[0] if previous_tokens is not None else None
    raw, _, flags = score_bar(before, previous)
    if flags or not np.isfinite(raw).all():
        raise ValueError('再構成元小節の難易度算出が不正')
    classes = discretize(np.asarray([raw]), boundaries)[0]
    np.testing.assert_allclose(raw, row['raw_before'], rtol=0, atol=1e-10)
    np.testing.assert_allclose(raw, row['raw_after'], rtol=0, atol=1e-10)
    np.testing.assert_allclose(classes[:5] / 7., row['d_after'], rtol=0, atol=1e-12)
    if dict(zip(ATTRS, map(int, classes))) != row['difficulty_after']:
        raise ValueError('再構成ラベルの不一致')
    if pattern_hash(before) != row['before_pattern'] or row['before_pattern'] != row['after_pattern']:
        raise ValueError('再構成パターン署名の不一致')


def check_parent(root):
    root = Path(root)
    index = json.loads((root / 'index.json').read_text())
    validation = json.loads((root / 'validation.json').read_text())
    if not validation.get('passed') or validation.get('index_sha256') != file_hash(root / 'index.json'):
        raise ValueError('元pairdataの全件検証記録がない、または索引が変更された')
    identity = json.loads((root / 'identity.json').read_text())
    if identity['formula_sha256'] != file_hash('src/data/drum_difficulty_v2.py'):
        raise ValueError('元pairdata作成後に難易度式が変更された')
    for name in ('bounds.json', 'drum_vocab.pkl'):
        if identity['source_hashes'][name] != file_hash(root / name):
            raise ValueError('元pairdataの語彙・境界が変更された')
    paths = set()
    for shard in index['shards']:
        path = (root / shard['path']).resolve()
        if not path.is_relative_to(root.resolve()) or path in paths or shard['split'] not in SPLITS:
            raise ValueError('元pairdataのシャード参照が不正')
        paths.add(path)
        if file_hash(path) != shard['sha256'] or len(read_pickle(path)) != shard['pairs']:
            raise ValueError('元pairdataのシャード破損')
    if sum(s['pairs'] for s in index['shards']) != validation['pairs']:
        raise ValueError('元pairdataの件数不一致')
    return {name: file_hash(root / name) for name in (*METADATA, 'index.json', 'identity.json', 'validation.json')}


def derived_rows(root, split, add_reconstruction):
    anchors = {}
    for row in iter_pairs(root, split):
        if row.get('task_type', 'arrangement') != 'arrangement' or row['X_before'] == row['X_after']:
            raise ValueError('追加元には編曲ペアのみを指定する')
        yield row
        if add_reconstruction:
            key = anchor_key(row)
            result = reconstruction_pair(row)
            if key in anchors:
                previous = dict(anchors[key]); current = dict(result)
                previous.pop('parent_pair_id'); current.pop('parent_pair_id')
                if previous != current:
                    raise ValueError('同じ曲・同じ元小節の文脈またはラベルが不一致')
            else:
                anchors[key] = result
    yield from anchors.values()


def validate_dataset(root, parent):
    root, parent = Path(root), Path(parent)
    identity = json.loads((root / 'identity.json').read_text())
    if identity['version'] != VERSION or identity['parent_hashes'] != check_parent(parent):
        raise ValueError('追加元の来歴が不一致')
    for name in METADATA:
        if file_hash(root / name) != identity['parent_hashes'][name]:
            raise ValueError('語彙・境界・出典のコピーが不一致')
    vocabulary = pickle.loads((root / 'drum_vocab.pkl').read_bytes())['idx2token']
    boundaries = json.loads((root / 'bounds.json').read_text())['boundaries']
    index = json.loads((root / 'index.json').read_text())
    paths = [s['path'] for s in index['shards']]
    if len(paths) != len(set(paths)) or set(paths) != {str(p.relative_to(root)) for p in (root / 'shards').glob('*.pkl.gz')}:
        raise ValueError('シャード一覧が不一致')
    for shard in index['shards']:
        if shard['split'] not in SPLITS or file_hash(root / shard['path']) != shard['sha256']:
            raise ValueError('出力シャードの分割またはハッシュが不一致')
        if len(read_pickle(root / shard['path'])) != shard['pairs']:
            raise ValueError('出力シャード件数が不一致')
    ids, sources, patterns, counts = set(), {}, {}, {}
    for split in SPLITS:
        expected = iter(derived_rows(parent, split, split in identity['reconstruction_splits']))
        count = Counter()
        for row in iter_pairs(root, split):
            if row != next(expected, None):
                raise ValueError('既存ペア保持または再構成ペア生成規約に不一致')
            if row['pair_id'] in ids or row['source']['split'] != split:
                raise ValueError('ペアID重複または分割名不一致')
            ids.add(row['pair_id'])
            source = anchor_key(row)[:2]
            if sources.setdefault(source, split) != split:
                raise ValueError('曲が分割間で重複')
            for pattern in (row['before_pattern'], row['after_pattern']):
                if patterns.setdefault(pattern, split) != split:
                    raise ValueError('現在小節の内容が分割間で重複')
            count[row.get('task_type', 'arrangement')] += 1
            if row.get('task_type') == 'reconstruction':
                check_reconstruction(row, vocabulary, boundaries)
        if next(expected, None) is not None:
            raise ValueError('必要なペアが欠落')
        counts[split] = {task: count[task] for task in ('arrangement', 'reconstruction')}
        print(f'{split}: {counts[split]} 検証済み', flush=True)
    summary = json.loads((root / 'summary.json').read_text())
    if summary['task_counts'] != counts or summary['pairs'] != len(ids):
        raise ValueError('タスク件数の集計不一致')
    return dict(passed=True, pairs=len(ids), task_counts=counts, index_sha256=file_hash(root / 'index.json'),
                original_pairs_preserved=True, reconstruction_difficulty_recomputed=True,
                source_piece_split_disjoint=True, current_bar_content_split_disjoint=True,
                limitation='元pairdataにある編曲前小節のみを追加。前後文脈の共有や近似重複は元データの制約を継承。')


def build_dataset(parent, output, reconstruction_splits=('train',), shard_size=4096):
    parent, output = Path(parent), Path(output)
    if shard_size < 1 or not reconstruction_splits or len(set(reconstruction_splits)) != len(reconstruction_splits) or set(reconstruction_splits) - set(SPLITS):
        raise ValueError('シャード件数または追加する分割が不正')
    if output.exists():
        raise ValueError('既存出力は上書きしない。新しい出力先を指定する')
    hashes = check_parent(parent)
    output.mkdir(parents=True)
    for name in METADATA:
        shutil.copyfile(parent / name, output / name)
    write_json(output / 'identity.json', dict(version=VERSION, parent=str(parent.resolve()), parent_hashes=hashes,
               reconstruction_splits=list(reconstruction_splits), shard_size=shard_size,
               deduplication='dataset/source_id/before_bar',
               code_hashes={p: file_hash(p) for p in ('data/pairdata/reconstruction.py', 'data/pairdata/core.py',
                           'data/pairdata/io.py', 'src/data/drum_difficulty_v2.py')}))
    shards, counts = [], {}
    parent_index = json.loads((parent / 'index.json').read_text())
    for split in SPLITS:
        if split not in reconstruction_splits:
            entries = [s for s in parent_index['shards'] if s['split'] == split]
            for entry in entries:
                destination = output / entry['path']
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(parent / entry['path'], destination)
                shards.append(dict(entry))
            counts[split] = dict(arrangement=sum(s['pairs'] for s in entries), reconstruction=0)
            continue
        buffer, count, number = [], Counter(), 0
        def flush():
            nonlocal number
            path = f'shards/{split}-{number:05d}.pkl.gz'
            write_pickle(output / path, buffer)
            shards.append(dict(path=path, split=split, pairs=len(buffer), sha256=file_hash(output / path)))
            number += 1
            buffer.clear()
        for row in derived_rows(parent, split, split in reconstruction_splits):
            buffer.append(row)
            count[row.get('task_type', 'arrangement')] += 1
            if len(buffer) == shard_size:
                flush()
        if buffer:
            flush()
        counts[split] = {task: count[task] for task in ('arrangement', 'reconstruction')}
    write_json(output / 'index.json', dict(version=VERSION, shards=shards))
    write_json(output / 'summary.json', dict(version=VERSION, pairs=sum(sum(c.values()) for c in counts.values()),
               split_counts={s: sum(c.values()) for s, c in counts.items()}, task_counts=counts))
    report = validate_dataset(output, parent)
    write_json(output / 'validation.json', report)
    return report
