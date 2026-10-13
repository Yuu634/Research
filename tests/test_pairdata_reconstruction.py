"""自己再構成の文脈、分割、元ペア保持と破損拒否を検証する。"""
from copy import deepcopy
import json
import pickle

import numpy as np
import pytest

from data.pairdata.core import pattern_hash
from data.pairdata.io import file_hash, iter_pairs, read_pickle, write_json, write_pickle
from data.pairdata.reconstruction import build_dataset, check_reconstruction, reconstruction_pair, validate_dataset
from src.data.drum_difficulty_v2 import ATTRS, decode_bars, discretize, score_bar


@pytest.fixture
def parent(tmp_path):
    root = tmp_path / 'parent'
    root.mkdir()
    strings = ['<PAD>', '<EOS>', '<TEMPO_120>', '<BAR>', '<BEAT_1>', '<BEAT_2>',
               '<POS_0>', '<POS_12>', 'KICK_HIT_Normal', 'SNARE_HIT_Normal', 'HH_CLOSED_HIT_Normal']
    vocab = dict(token2idx={t: i for i, t in enumerate(strings)}, idx2token=dict(enumerate(strings)))
    bounds = {a: [.1, .5, 1, 2, 4, 8] for a in ATTRS}
    (root / 'drum_vocab.pkl').write_bytes(pickle.dumps(vocab))
    write_json(root / 'bounds.json', dict(boundaries=bounds))
    (root / 'sources.jsonl').write_text('')
    shards = []
    for split in ('train', 'val', 'test', 'itest'):
        tokens = ['<TEMPO_120>', '<BAR>', '<BEAT_1>', '<POS_0>', 'KICK_HIT_Normal',
                  '<BEAT_2>', '<POS_0>', 'SNARE_HIT_Normal']
        # 各分割で現在小節の内容が異なるよう、HHの位置を変える。
        strings_extra = [f'<POS_{p}>' for p in (1, 2, 3, 4)]
        for token in strings_extra:
            if token not in vocab['token2idx']:
                index = len(vocab['token2idx'])
                vocab['token2idx'][token] = index
                vocab['idx2token'][index] = token
        tokens += [strings_extra[('train', 'val', 'test', 'itest').index(split)], 'HH_CLOSED_HIT_Normal', '<EOS>']
        before = [vocab['token2idx'][t] for t in tokens]
        after = before[:-3] + [vocab['token2idx']['<POS_12>'], vocab['token2idx']['KICK_HIT_Normal'], before[-1]]
        bar = decode_bars([0], before, vocab['idx2token'])[0]
        other = decode_bars([0], after, vocab['idx2token'])[0]
        # 前小節ありの値を使い、ゼロからの再採点による誤りを検出する。
        raw = list(score_bar(bar, bar)[0])
        classes = list(map(int, discretize(np.asarray([raw]), bounds)[0]))
        row = dict(pair_id=split + ':a', X_before=before, X_after=after, bar_positions=[0],
                   before_pattern=pattern_hash(bar), after_pattern=pattern_hash(other),
                   d_before=[c / 7 for c in classes[:5]], d_after=[0.] * 5,
                   delta_d=[-c / 7 for c in classes[:5]], raw_before=raw, raw_after=[0.] * 6,
                   difficulty_before=dict(zip(ATTRS, classes)), difficulty_after=dict(zip(ATTRS, [0] * 6)),
                   original_after_raw=[0.] * 6, original_after_classes=dict(zip(ATTRS, [0] * 6)),
                   q=.9, similarities=dict(rhythm=.9, skeleton=.9, context=.9), difficulty_distance=.4,
                   source=dict(id=split, dataset='fixture', split=split, before_bar=4, after_bar=8),
                   context=dict(previous=before, following=before, original_after_previous=after,
                                original_after_following=after, after_difficulty_context='before_previous'))
        # 元小節は同一だが編曲先が異なるという候補重複を模擬する。
        rows = [row]
        if split == 'train':
            duplicate = deepcopy(row)
            duplicate['pair_id'] = 'train:b'
            duplicate['source']['after_bar'] = 12
            rows.append(duplicate)
        path = f'shards/{split}-00000.pkl.gz'
        write_pickle(root / path, rows)
        shards.append(dict(path=path, split=split, pairs=len(rows), sha256=file_hash(root / path)))
    (root / 'drum_vocab.pkl').write_bytes(pickle.dumps(vocab))
    # afterの内容も分割ごとに異なるものへ調整する。
    for i, entry in enumerate(shards):
        rows = read_pickle(root / entry['path'])
        for row in rows:
            row['X_after'][-3] = vocab['token2idx'][f'<POS_{i + 1}>']
            row['after_pattern'] = pattern_hash(decode_bars([0], row['X_after'], vocab['idx2token'])[0])
        write_pickle(root / entry['path'], rows)
        entry['sha256'] = file_hash(root / entry['path'])
    write_json(root / 'index.json', dict(shards=shards))
    write_json(root / 'validation.json', dict(passed=True, pairs=5, index_sha256=file_hash(root / 'index.json')))
    write_json(root / 'identity.json', dict(formula_sha256=file_hash('src/data/drum_difficulty_v2.py'),
               source_hashes={name: file_hash(root / name) for name in ('bounds.json', 'drum_vocab.pkl')}))
    return root, vocab, bounds


def test_train_only_preserves_every_original_and_evaluation_bytes(parent, tmp_path):
    root, _, _ = parent
    output = tmp_path / 'augmented'
    report = build_dataset(root, output, shard_size=2)
    assert report['passed'] and report['pairs'] == 6
    rows = list(iter_pairs(output, 'train'))
    assert rows[:2] == list(iter_pairs(root, 'train'))
    assert len(rows) == 3 and rows[-1]['task_type'] == 'reconstruction'
    for split in ('val', 'test', 'itest'):
        name = f'shards/{split}-00000.pkl.gz'
        assert (root / name).read_bytes() == (output / name).read_bytes()
    with pytest.raises(ValueError, match='上書き'):
        build_dataset(root, output)


def test_reconstruction_resets_target_and_after_provenance_without_mutating_parent(parent):
    root, vocab, bounds = parent
    row = next(iter_pairs(root))
    original = deepcopy(row)
    result = reconstruction_pair(row)
    check_reconstruction(result, vocab['idx2token'], bounds)
    assert result['source']['after_bar'] == result['source']['before_bar']
    assert result['context']['original_after_previous'] == row['context']['previous']
    assert result['context']['original_after_following'] == row['context']['following']
    assert result['original_after_raw'] == row['raw_before']
    assert result['parent_pair_id'] == row['pair_id']
    result['X_after'].append(99)
    assert row == original


def test_reconstruction_checks_actual_difficulty(parent):
    root, vocab, bounds = parent
    row = reconstruction_pair(next(iter_pairs(root)))
    row['raw_after'][0] += 1
    with pytest.raises(AssertionError):
        check_reconstruction(row, vocab['idx2token'], bounds)


def test_corrupt_parent_rejected_before_creating_output(parent, tmp_path):
    root, _, _ = parent
    (root / 'shards/train-00000.pkl.gz').write_bytes(b'broken')
    output = tmp_path / 'augmented'
    with pytest.raises(ValueError, match='破損'):
        build_dataset(root, output)
    assert not output.exists()


def test_revalidation_detects_modified_output_even_with_updated_shard_hash(parent, tmp_path):
    root, _, _ = parent
    output = tmp_path / 'augmented'
    build_dataset(root, output)
    index = json.loads((output / 'index.json').read_text())
    entry = index['shards'][0]
    rows = read_pickle(output / entry['path'])
    rows[-1]['d_after'][0] = 123
    write_pickle(output / entry['path'], rows)
    entry['sha256'] = file_hash(output / entry['path'])
    write_json(output / 'index.json', index)
    with pytest.raises(ValueError, match='生成規約'):
        validate_dataset(output, root)


def test_conflicting_context_for_same_anchor_rejected(parent, tmp_path):
    root, _, _ = parent
    entry = json.loads((root / 'index.json').read_text())['shards'][0]
    rows = read_pickle(root / entry['path'])
    rows[1]['context']['following'] = None
    write_pickle(root / entry['path'], rows)
    index = json.loads((root / 'index.json').read_text())
    index['shards'][0]['sha256'] = file_hash(root / entry['path'])
    write_json(root / 'index.json', index)
    write_json(root / 'validation.json', dict(passed=True, pairs=5, index_sha256=file_hash(root / 'index.json')))
    with pytest.raises(ValueError, match='文脈またはラベル'):
        build_dataset(root, tmp_path / 'augmented')
