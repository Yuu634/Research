"""共通入力と中間譜面の準備。教師譜面はDecoderの教師信号だけに用いる。"""
from dataclasses import dataclass
from itertools import islice
import json
from pathlib import Path
import pickle
import numpy as np
import torch

from data.pairdata.io import file_hash, iter_pairs, read_pickle, write_json, write_pickle
from src.arrangement.inputs import parse_bar, prepare_input, request_from_pair
from src.arrangement.representation import to_tokens
from src.beamsearch import beam_search, load_selected_model
from src.beam_muse.features import extract_skeleton


@dataclass
class Example:
    case_id: str
    source_id: str
    split: str
    prepared: object
    intermediate: object
    teacher: object
    history: list
    beam_seconds: float


def load_metadata(meta, selected_model):
    scorer, search, actions, evaluation, selected = load_selected_model(selected_model, meta)
    vocabulary = pickle.loads((Path(meta)/'drum_vocab.pkl').read_bytes())
    ids = vocabulary['token2idx']
    if set(ids.values()) != set(range(len(ids))) or not {'<PAD>', '<EOS>', '<BAR>'} <= ids.keys():
        raise ValueError('連続IDとPAD/EOS/BARを持つstandard語彙が必要')
    return scorer, search, actions, evaluation, selected, vocabulary


def prepare_example(pair, vocabulary, scorer, search, actions, max_tokens):
    request = request_from_pair(pair, vocabulary['idx2token'])
    prepared = prepare_input(request, scorer)
    teacher = parse_bar(dict(tokens=[vocabulary['idx2token'][int(i)] for i in pair['X_after']],
                             meter=[4, 4], length=96), scorer.vocabulary)
    if teacher.bpm != prepared.original.bpm:
        raise ValueError('教師譜面と原譜のBPMが異なる')
    # 信頼する教師ラベルが現在の算出式・前小節と一致することも検査する。
    actual = scorer.difficulty(teacher, prepared.previous).normalized
    if any(abs(actual[k]-prepared.target[k]) > 1e-6 for k in actual):
        raise ValueError('教師譜面の難易度が目標と不一致')
    result = beam_search(prepared, scorer, search, actions)
    bars = (prepared.previous, prepared.original, result.candidate, prepared.following, teacher)
    if any(len(to_tokens(bar)) > max_tokens for bar in bars if bar is not None):
        raise ValueError('トークン上限超過（自動切り詰めなし）')
    return Example(pair['pair_id'], pair['source']['dataset']+':'+pair['source']['id'], pair['source']['split'],
                   prepared, result.candidate, teacher, [a.as_dict() for a in result.history], result.elapsed_seconds)


def collate(examples, vocabulary, scorer, variant, device='cpu', *, training=True):
    if variant not in ('musemorphose', 'beam_muse') or not examples:
        raise ValueError('ニューラル方式と1件以上の入力が必要')
    ids = vocabulary['token2idx']
    encoder, targets, current, skeleton, present = [], [], [], [], []
    for example in examples:
        prepared = example.prepared
        center = example.intermediate if variant == 'beam_muse' else prepared.original
        def encode(bar):
            return [ids[t] for t in to_tokens(bar)] if bar is not None else [ids['<BAR>'], ids['<EOS>']]
        encoder.append([encode(prepared.previous), encode(center), encode(prepared.following)])
        targets.append(encode(example.teacher if training else center))
        current.append(list(scorer.difficulty(center, prepared.previous).normalized.values()))
        skeleton.append(extract_skeleton(prepared.original, section=prepared.context.get('section')).vector())
        present.append([prepared.previous is not None, prepared.following is not None])
    length = max(len(bar) for row in encoder for bar in row)
    token_count = max(len(row)-1 for row in targets)
    enc = torch.full((len(examples), 3, length), ids['<PAD>'], dtype=torch.long)
    inp = torch.full((token_count, len(examples)), ids['<PAD>'], dtype=torch.long)
    target = torch.full_like(inp, ids['<PAD>'])
    for i, row in enumerate(encoder):
        for j, bar in enumerate(row):
            enc[i, j, :len(bar)] = torch.tensor(bar)
        inp[:len(targets[i])-1, i] = torch.tensor(targets[i][:-1])
        target[:len(targets[i])-1, i] = torch.tensor(targets[i][1:])
    values = dict(encoder_tokens=enc, decoder_input=inp, decoder_target=target,
                  skeleton=torch.tensor(np.stack(skeleton)), target=torch.tensor([list(e.prepared.target.values()) for e in examples]),
                  current=torch.tensor(current), context_present=torch.tensor(present, dtype=torch.float32))
    return {key: value.to(device) for key, value in values.items()}


def source_hashes():
    paths = list(Path('src/arrangement').glob('*.py')) + list(Path('src/beam_muse').glob('*.py'))
    paths += list(Path('src/musemorphose').glob('*.py')) + list(Path('src/models').glob('*.py'))
    paths += [Path('src/data/drum_difficulty_v2.py'), Path('src/data/drum_difficulty_calculator.py')]
    paths += [Path('data/pairdata/io.py'), Path('scripts/train/tune_beam_muse.py'),
              Path('scripts/generate/arrange_beam_muse.py')]
    return {str(p): file_hash(p) for p in sorted(paths)}


def prepare_cache(pairdata, meta, selected_model, output, *, max_tokens=512, smoke_limit=None):
    output, pairdata = Path(output), Path(pairdata)
    if max_tokens < 4 or (smoke_limit is not None and smoke_limit < 1):
        raise ValueError('上限が不正')
    scorer, search, actions, _, _, vocabulary = load_metadata(meta, selected_model)
    index = json.loads((pairdata/'index.json').read_text())
    for shard in index['shards']:
        if file_hash(pairdata/shard['path']) != shard['sha256']:
            raise ValueError('ペアシャードのハッシュ不一致')
    identity = dict(schema_version=1, pair_index=file_hash(pairdata/'index.json'),
                    vocabulary=file_hash(Path(meta)/'drum_vocab.pkl'), bounds=file_hash(Path(meta)/'bounds.json'),
                    beam_model=file_hash(selected_model), max_tokens=max_tokens, smoke_limit=smoke_limit, sources=source_hashes())
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved['identity'] != identity:
            raise ValueError('入力・設定・実装が変わったため別のキャッシュ出力先が必要')
        for split, info in saved['splits'].items():
            if file_hash(output/f'{split}.pkl.gz') != info['sha256']:
                raise ValueError('キャッシュ破損')
        return saved
    summaries, source_splits, seen_ids = {}, {}, set()
    splits = ('train', 'val') if smoke_limit else ('train', 'val', 'test', 'itest')
    for split in splits:
        rows, excluded = [], []
        pairs = iter_pairs(pairdata, split)
        if smoke_limit:
            pairs = islice(pairs, smoke_limit)
        for pair in pairs:
            source = pair['source']['dataset']+':'+pair['source']['id']
            if pair['source']['split'] != split or source_splits.setdefault(source, split) != split or pair['pair_id'] in seen_ids:
                raise ValueError('分割をまたぐ曲・重複ペア・分割名の不一致')
            seen_ids.add(pair['pair_id'])
            try:
                rows.append(prepare_example(pair, vocabulary, scorer, search, actions, max_tokens))
            except ValueError as error:
                excluded.append(dict(case_id=pair['pair_id'], reason=str(error)))
            if (len(rows)+len(excluded)) % 100 == 0:
                print(f'{split}: {len(rows)}件を準備、除外{len(excluded)}件', flush=True)
        if not rows:
            raise ValueError(split+'に適格な入力がない')
        path = output/f'{split}.pkl.gz'
        write_pickle(path, rows)
        summaries[split] = dict(n=len(rows), excluded=excluded, sha256=file_hash(path))
        print(f'{split}: {len(rows)}件を保存、除外{len(excluded)}件', flush=True)
    manifest = dict(identity=identity, splits=summaries)
    write_json(manifest_path, manifest)
    return manifest


def load_cache(root, split, meta, selected_model):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    identity = manifest['identity']
    for key, path in (('vocabulary', Path(meta)/'drum_vocab.pkl'), ('bounds', Path(meta)/'bounds.json'), ('beam_model', selected_model)):
        if file_hash(path) != identity[key]:
            raise ValueError('キャッシュのメタデータ不一致: '+key)
    if identity['sources'] != source_hashes():
        raise ValueError('キャッシュ作成後に実装が変更された')
    if split not in manifest['splits'] or file_hash(root/f'{split}.pkl.gz') != manifest['splits'][split]['sha256']:
        raise ValueError('分割がない、またはキャッシュが破損')
    return read_pickle(root/f'{split}.pkl.gz'), manifest
