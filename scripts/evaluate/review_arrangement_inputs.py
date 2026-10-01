"""ステップ3：BeamSearchの入力を検証し、目標変換・保護音・無編集スコアを確認する。"""
import argparse
import hashlib
import html
from itertools import islice
import json
from pathlib import Path
import pickle

from scripts.evaluate.review_arrangement_rules import svg
from src.arrangement.examples import make_bar
from src.arrangement.inputs import bar_to_payload, prepare_input, request_from_pair
from src.arrangement.rules import apply_rule
from src.arrangement.scoring import ScoringModel


def _unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('JSONのキーが重複: ' + key)
        result[key] = value
    return result


def read_request(path):
    def invalid_constant(value):
        raise ValueError('JSONに非有限数は指定不可: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'),
                      object_pairs_hook=_unique_object, parse_constant=invalid_constant)


def example_requests(model):
    original = make_bar([(p, 'hh_closed') for p in range(0, 96, 6)] +
                        [(p, 'kick') for p in range(0, 96, 24)] +
                        [(24, 'snare'), (72, 'snare', 'Accent'),
                         (18, 'snare', 'Ghost'), (66, 'snare', 'Ghost')])
    first = apply_rule(original, 'hh_16_to_8')
    second = apply_rule(first.after, 'remove_ghost_notes', original=original)
    target = model.difficulty(second.after).normalized
    before = model.difficulty(original).normalized
    shared = dict(schema_version=1, original=bar_to_payload(original), previous=None,
                  following=None, context=dict(section=None, phrase=None), extra_locked_ids=[])
    return {
        'absolute': dict(shared, target=dict(mode='absolute', values=target)),
        'delta': dict(shared, target=dict(mode='delta', values={k: target[k]-before[k] for k in target})),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', type=Path, help='確認するJSON入力')
    group.add_argument('--examples', action='store_true', help='等価な絶対値・差分の入力例を生成して検証')
    group.add_argument('--pairdata', type=Path, help='ローカル生成済みv1ペアデータを入力へ変換して確認')
    parser.add_argument('--split', choices=('train', 'val', 'test', 'itest'), default='train')
    parser.add_argument('--limit', type=int, default=16, help='ペアデータから確認する件数（正の整数）')
    parser.add_argument('--meta', type=Path, default=Path('data/metadata/drum_collection_difficulty_v2'))
    parser.add_argument('--output', type=Path, default=Path('output/beamsearch_step3'))
    args = parser.parse_args()
    try:
        bounds_path = args.meta / 'bounds.json'
        vocab_path = args.meta / 'drum_vocab.pkl'
        bounds = json.loads(bounds_path.read_text())['boundaries']
        vocabulary = pickle.loads(vocab_path.read_bytes())
        if vocabulary['tokenization_method'] != 'standard':
            raise ValueError('standard語彙が必要')
        model = ScoringModel(bounds, vocabulary['token2idx'])
        sources = {}
        if args.limit <= 0:
            raise ValueError('--limitは正の整数')
        if args.pairdata:
            from data.pairdata.io import iter_pairs
            pairs = list(islice(iter_pairs(args.pairdata, args.split), args.limit))
            if not pairs:
                raise ValueError('指定した分割にペアデータがない')
            requests = {f'{args.split}_{i:04d}': request_from_pair(pair, vocabulary['idx2token'])
                        for i, pair in enumerate(pairs)}
            sources = {f'{args.split}_{i:04d}': dict(pair_id=pair['pair_id'], source=pair['source'])
                       for i, pair in enumerate(pairs)}
        else:
            requests = example_requests(model) if args.examples else {'input': read_request(args.input)}
        prepared = {name: prepare_input(payload, model) for name, payload in requests.items()}
        provenance = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in (
            ('bounds_sha256', bounds_path), ('vocabulary_sha256', vocab_path),
            ('input_parser_sha256', Path('src/arrangement/inputs.py')),
            ('scoring_sha256', Path('src/arrangement/scoring.py')),
            ('formula_sha256', Path('src/data/drum_difficulty_v2.py')),
        )}
        args.output.mkdir(parents=True, exist_ok=True)
        articles = []
        for name, value in prepared.items():
            payload = requests[name]
            input_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                                  allow_nan=False).encode('utf-8')).hexdigest()
            report = dict(value.report(model), provenance=provenance, input_sha256=input_hash,
                          source=sources.get(name))
            for suffix, content in (('request', payload), ('report', report)):
                (args.output / f'{name}_{suffix}.json').write_text(
                    json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
            pretty = html.escape(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
            articles.append(f'<article><h2>{name}</h2>{svg(value.original)}<pre>{pretty}</pre></article>')
        (args.output / 'index.html').write_text(
            '<!doctype html><html lang="ja"><meta charset="utf-8"><title>入力確認：ステップ3</title>'
            '<style>body{font:16px sans-serif;max-width:1000px;margin:30px auto;padding:10px}'
            'pre{white-space:pre-wrap}svg{width:100%}article{border-top:1px solid #ccc}</style>'
            '<h1>BeamSearch入力確認：ステップ3</h1>'
            '<p>入力検証と無編集スコアのみ。探索・モデル選定は未実施です。</p>'
            + ''.join(articles) + '</html>', encoding='utf-8')
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(f'{len(prepared)}件の入力を確認: {args.output}/index.html')


if __name__ == '__main__':
    main()
