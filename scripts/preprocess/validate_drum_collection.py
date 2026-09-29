"""全件のメタデータ・分割・出力一覧と、保存ファイルの内容を検証する。"""
import argparse
from collections import Counter, defaultdict
import multiprocessing as mp
import hashlib
import json
from pathlib import Path
import pickle

from scripts.preprocess.build_drum_collection import CollectionTokenizer, SOURCES, write_json


def validate_row(row):
    if 'extracted' in row:
        path = Path(row['extracted'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['extracted_sha256'], row['id']
    if row.get('token_status') != 'ok':
        return
    with Path(row['tokens']).open('rb') as f:
        bars, tokens = pickle.load(f)
    tokenizer = CollectionTokenizer()
    assert len(tokens) == row['token_count'] and len(bars) == row['bars'], row['id']
    assert bars and bars[0] == 0 and all(a < b for a, b in zip(bars, bars[1:])), row['id']
    assert all(0 <= n < tokenizer.vocab_size for n in tokens), row['id']
    assert 0 not in tokens and tokens[-1] == tokenizer.token2idx['<EOS>'], row['id']
    assert all(tokenizer.idx2token[tokens[b]].startswith('<TEMPO_') for b in bars), row['id']
    hits = sum(not tokenizer.idx2token[t].startswith('<') for t in tokens)
    assert hits == row['supported_note_count'] == row['tokenized_hits'], row['id']
    import numpy as np
    assert hashlib.sha256(np.asarray(tokens, dtype='<i4').tobytes()).hexdigest() == row['token_sha256'], row['id']


def choose_rows(rows, sample_per_group=None, include_ids=()):
    required = set(include_ids)
    assert required <= {r['id'] for r in rows}, '指定IDがmanifestに存在しません'
    if sample_per_group is None:
        return rows
    groups = defaultdict(list)
    for row in rows:
        key = (row['dataset'], row.get('split') or row.get('official_split'),
               row.get('token_status', row['status']))
        groups[key].append(row)
    selected = {r['id']: r for r in rows if r['id'] in required}
    for group in groups.values():
        ordered = sorted(group, key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())
        selected.update((r['id'], r) for r in ordered[:sample_per_group])
    return sorted(selected.values(), key=lambda r: r['id'])


def validate(data_root, collection, workers, sample_per_group=None, include_ids=()):
    meta = data_root / 'metadata' / collection
    rows = [json.loads(line) for line in (meta / 'manifest.jsonl').open()]
    assert len({r['id'] for r in rows}) == len(rows)
    assert len({r['source'] for r in rows}) == len(rows)
    expected = json.loads((meta / 'policy.json').read_text())['expected_inputs']
    assert len(rows) == expected
    token_root = data_root / 'tokens' / collection
    actual_tokens = {str(p) for p in token_root.glob('*.pkl')}
    expected_tokens = {r['tokens'] for r in rows if r.get('token_status') == 'ok'}
    assert actual_tokens == expected_tokens, 'トークンファイル一覧とmanifestが不一致'
    actual_midi = {str(p) for dataset in SOURCES
                   for p in (data_root / 'extracted' / collection / dataset).glob('*.mid')}
    assert actual_midi == {r['extracted'] for r in rows if 'extracted' in r}, '抽出MIDI一覧とmanifestが不一致'
    for r in rows:
        if r.get('token_status') == 'ok':
            assert r['supported_note_count'] == r['tokenized_hits'] > 0, r['id']
            assert r['additional_tokenizer_dropped_notes'] == 0, r['id']
            assert r['bars'] > 0 and r['token_count'] > r['tokenized_hits'], r['id']
    all_hashes, all_paths = set(), set()
    split_counts = {}
    for split in ('train', 'val', 'test', 'itest'):
        selected = [r for r in rows if r.get('split') == split and not r.get('split_exclusion')]
        with (meta / f'{split}_split.pkl').open('rb') as f:
            paths = pickle.load(f)
        expected_paths = [Path(r['tokens']).name for r in selected]
        assert paths == expected_paths
        assert not all_paths.intersection(paths)
        all_paths.update(paths)
        hashes = {r['token_sha256'] for r in selected}
        assert len(hashes) == len(selected)
        assert not all_hashes.intersection(hashes)
        all_hashes.update(hashes)
        assert all(r['official_split'] in (None, split) for r in selected)
        split_counts[split] = len(selected)
    checked = choose_rows(rows, sample_per_group, include_ids)
    print(f'全件一覧・メタデータ・分割: {len(rows)}件確認。再読み出し: {len(checked)}件', flush=True)
    with mp.Pool(processes=workers) as pool:
        for index, _ in enumerate(pool.imap_unordered(validate_row, checked, chunksize=1), 1):
            if index % (100 if sample_per_group else 10000) == 0:
                print(f'内容検証 {index}/{len(checked)}', flush=True)
    mode = 'full' if sample_per_group is None else 'stratified_sample'
    result = {
        'verification_mode': mode, 'sample_per_group': sample_per_group,
        'manifest_rows': len(rows), 'source_counts': dict(Counter(r['dataset'] for r in rows)),
        'file_catalog_verified': {'midi': len(actual_midi), 'tokens': len(actual_tokens)},
        'extracted_files_verified': sum('extracted' in r for r in checked),
        'token_files_verified': sum(r.get('token_status') == 'ok' for r in checked),
        'checked_rows': len(checked), 'required_ids_verified': sorted(set(include_ids)),
        'split_counts': split_counts, 'exact_token_overlap_across_splits': 0,
        'scope': '全件メタデータ・出力一覧・分割。内容再読み出し範囲はverification_mode参照。近似重複は未判定',
    }
    if sample_per_group is not None:
        result['checked_ids'] = [r['id'] for r in checked]
    write_json(meta / 'validation.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('checked_ids', 'required_ids_verified')}, ensure_ascii=False, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('data'))
    p.add_argument('--collection', default='drum_collection_v1')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--sample-per-group', type=int, help='データセット・分割・処理結果ごとの再読み出し件数。省略時は全件')
    p.add_argument('--include-ids', type=Path, help='再読み出し必須IDのJSON配列、またはidsキーを持つJSON')
    args = p.parse_args()
    if args.workers < 1 or (args.sample_per_group is not None and args.sample_per_group < 1):
        p.error('workers/sample-per-group は1以上')
    required = []
    if args.include_ids:
        payload = json.loads(args.include_ids.read_text())
        required = payload['ids'] if isinstance(payload, dict) else payload
    validate(args.data_root, args.collection, args.workers, args.sample_per_group, required)


if __name__ == '__main__':
    main()
