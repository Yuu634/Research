"""ステップ4：固定例・無編集・固定ルール列で評価指標を確認する（探索・選定は行わない）。"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import html
from itertools import islice
import json
from pathlib import Path
import pickle
from time import perf_counter

from scripts.evaluate.review_arrangement_inputs import example_requests, read_request
from scripts.evaluate.review_arrangement_rules import svg
from src.arrangement.evaluation import EvaluationConfig, evaluate_result, summarize_results
from src.arrangement.inputs import prepare_input, request_from_pair
from src.arrangement.rules import apply_rule
from src.arrangement.scoring import ScoringModel


def fixed_rules(prepared, names):
    current = prepared.original
    trace = []
    for name in names:
        result = apply_rule(current, name, original=prepared.original, guards=prepared.guards)
        current = result.after
        trace.append(dict(rule=name, status=result.status, reason=result.reason))
    return current, sum(step['status'] == 'applied' for step in trace), trace


def artificial_examples(model, config):
    prepared = prepare_input(example_requests(model)['absolute'], model)
    hh, _, _ = fixed_rules(prepared, ('hh_16_to_8',))
    ghost, _, _ = fixed_rules(prepared, ('remove_ghost_notes',))
    both, count, _ = fixed_rules(prepared, ('hh_16_to_8', 'remove_ghost_notes'))
    cases = [
        ('無編集', prepared.original, 0), ('HH間引き', hh, 1), ('Ghost削除', ghost, 1),
        ('HH間引き＋Ghost削除', both, count),
        ('同じ譜面へ3ルールで到達した場合', both, 3),
        ('保護Kickを削除した不適格出力', prepared.original.with_notes(
            n for n in prepared.original.notes if n.instrument != 'kick'), 1),
        ('出力失敗', None, 0),
    ]
    rows = [evaluate_result(prepared, bar, model, case_id=name, source_id='artificial',
                            rule_count=count, config=config) for name, bar, count in cases]
    figures = [(name, prepared.original, bar) for name, bar, _ in cases]
    return {'動作確認用の7出力': rows}, figures


def write_report(report, figures, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / 'review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                                       encoding='utf-8')
    columns = ('method', 'case_id', 'source_id', 'status', 'baseline_mae', 'mae', 'mae_improvement',
               'target_success', 'joint_success', 'skeleton_similarity', 'edit_distance',
               'rule_count', 'elapsed_seconds')
    records = []
    for method, rows in report['results'].items():
        for row in rows:
            metric = row['metrics'] or {}
            records.append({key: method if key == 'method' else row.get(key, metric.get(key)) for key in columns})
    with (output / 'cases.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    def cell(value):
        return html.escape('—' if value is None else f'{value:.6f}' if type(value) is float else str(value))
    table = '<table><tr>'+''.join(f'<th>{key}</th>' for key in columns)+'</tr>'
    table += ''.join('<tr>'+''.join(f'<td>{cell(row[key])}</td>' for key in columns)+'</tr>' for row in records)+'</table>'
    images = ''.join('<article><h2>'+html.escape(name)+'</h2><p>元譜</p>'+svg(before)+
                     ('<p>出力</p>'+svg(after) if after is not None else '<p>出力なし</p>')+'</article>'
                     for name, before, after in figures)
    summary = html.escape(json.dumps(report['summaries'], ensure_ascii=False, indent=2, allow_nan=False))
    (output / 'index.html').write_text(
        '<!doctype html><html lang="ja"><meta charset="utf-8"><title>評価指標：ステップ4</title>'
        '<style>body{font:15px sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ccc;'
        'padding:6px}pre{white-space:pre-wrap}article{max-width:1000px}svg{width:100%}</style>'
        '<h1>評価指標：ステップ4</h1><p>'+html.escape(report['purpose'])+'</p>'
        '<p>成功条件：全5属性の誤差が '+cell(report['evaluation_config']['target_tolerance'])+
        ' 以下、骨格類似度が '+cell(report['evaluation_config']['skeleton_threshold'])+
        ' 以上、制約違反なし。無出力・違反出力も成功率の分母に含めます。</p>'
        '<p>骨格は研究上の定義による指標です。演奏者による自然さ・類似度の評価は未実施です。</p>'
        + table + '<details><summary>集計値と分母</summary><pre>'+summary+'</pre></details>'+images+'</html>',
        encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--examples', action='store_true', help='既知の7出力で指標の動作を確認')
    group.add_argument('--input', type=Path, help='ステップ3のJSON入力を無編集・固定ルール列で評価')
    group.add_argument('--pairdata', type=Path, help='生成済みv1ペアデータで無編集・固定ルール列を診断')
    parser.add_argument('--split', choices=('train', 'val', 'test', 'itest'), default='train')
    parser.add_argument('--limit', type=int, default=16)
    parser.add_argument('--meta', type=Path, default=Path('data/metadata/drum_collection_difficulty_v2'))
    parser.add_argument('--output', type=Path, default=Path('output/beamsearch_step4'))
    parser.add_argument('--target-tolerance', type=float, default=.5/7, help='全モデル共通の属性ごとの許容誤差')
    parser.add_argument('--skeleton-threshold', type=float, default=.90, help='全モデル共通の骨格類似度閾値')
    args = parser.parse_args()
    try:
        config = EvaluationConfig(args.target_tolerance, args.skeleton_threshold).validate()
        if args.limit <= 0:
            raise ValueError('--limitは正の整数')
        bounds_path, vocab_path = args.meta/'bounds.json', args.meta/'drum_vocab.pkl'
        bounds = json.loads(bounds_path.read_text())['boundaries']
        vocabulary = pickle.loads(vocab_path.read_bytes())
        if vocabulary['tokenization_method'] != 'standard':
            raise ValueError('standard語彙が必要')
        model = ScoringModel(bounds, vocabulary['token2idx'])
        manifests = []
        if args.examples:
            results, figures = artificial_examples(model, config)
            purpose = '同じ人工入力に対する7出力の動作例。集計値はモデル性能や汎化性能を示しません。'
        else:
            if args.pairdata:
                from data.pairdata.io import iter_pairs
                pairs = list(islice(iter_pairs(args.pairdata, args.split), args.limit))
                inputs = [(p['pair_id'], p['source']['dataset']+':'+p['source']['id'],
                           request_from_pair(p, vocabulary['idx2token'])) for p in pairs]
            else:
                inputs = [('input', 'input', read_request(args.input))]
            if not inputs:
                raise ValueError('評価入力がない')
            results = {'no_edit': [], 'hh_then_ghost': []}
            figures = []
            for case_id, source_id, payload in inputs:
                prepared = prepare_input(payload, model)
                manifests.append(dict(case_id=case_id, source_id=source_id,
                    input_sha256=hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                                         allow_nan=False).encode('utf-8')).hexdigest()))
                for method, names in (('no_edit', ()), ('hh_then_ghost', ('hh_16_to_8', 'remove_ghost_notes'))):
                    start = perf_counter()
                    candidate, count, trace = fixed_rules(prepared, names)
                    elapsed = perf_counter()-start
                    row = evaluate_result(prepared, candidate, model, case_id=case_id, source_id=source_id,
                                          rule_count=count, elapsed_seconds=elapsed, config=config)
                    row['trace'] = trace
                    results[method].append(row)
                    if len(figures) < 4:
                        figures.append((method+': '+case_id, prepared.original, candidate))
            purpose = '無編集と固定のHH間引き→Ghost削除による指標診断。探索・学習・モデル選定は未実施です。'
        provenance = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in (
            ('bounds_sha256', bounds_path), ('vocabulary_sha256', vocab_path),
            ('evaluation_sha256', Path('src/arrangement/evaluation.py')),
            ('scoring_sha256', Path('src/arrangement/scoring.py')),
            ('inputs_sha256', Path('src/arrangement/inputs.py')),
            ('review_cli_sha256', Path(__file__)),
            ('formula_sha256', Path('src/data/drum_difficulty_v2.py')),
            ('rules_sha256', Path('src/arrangement/rules.py')),
        )}
        report = dict(purpose=purpose, evaluation_config=asdict(config), provenance=provenance,
                      input_manifest=manifests, split=args.split if args.pairdata else None,
                      summaries={method: summarize_results(rows) for method, rows in results.items()},
                      runtime_scope='固定ルール適用のみ。読込・入力検証・評価計算は含まない。人工例は未計測。',
                      results=results)
        write_report(report, figures, args.output)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(f'評価指標の確認結果: {args.output}/index.html')


if __name__ == '__main__':
    main()
