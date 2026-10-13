"""比較実験の結果を検証し、選定設定・表・譜面/MIDIの確認資料を出力する。"""
import argparse
from collections import Counter
import csv
from dataclasses import asdict
import html
import json
from pathlib import Path

from data.pairdata.io import file_hash, read_pickle, write_json
from scripts.evaluate.compare_arrangement_models import read_evaluation_batch, selection_key
from scripts.evaluate.review_arrangement_rules import midi, svg
from src.common.evaluation import evaluate_result
from src.beamsearch.beam_search import SearchConfig
from src.beamsearch.rules import apply_rule
from src.beamsearch.selected_model import load_selected_model
from src.common.provenance import implementation_hashes, validate_sources


def export_selected_settings(selected, model, output):
    """推論設定は既定値も展開し、将来の既定値変更で係数が変わらないように保存する。"""
    deployment = dict(selected)
    deployment['candidate'] = dict(selected['candidate'], score=asdict(model.config),
                                   search=asdict(SearchConfig(**selected['candidate']['search']).validate()))
    write_json(output, deployment)


def report(args):
    selected_path = args.run/'selected_model.json'
    model, _, _, evaluation, selected = load_selected_model(selected_path, args.meta)
    run_manifest = json.loads((args.run/'run_manifest.json').read_text())
    if file_hash(args.run/'run_manifest.json') != selected['run_manifest_sha256']:
        raise ValueError('選定後に実行記録が変更された')
    validate_sources(run_manifest['sources'], implementation_hashes('beamsearch'))
    summaries = {split: json.loads((args.run/split/'summaries.json').read_text()) for split in ('val', 'test', 'itest')}
    candidates = run_manifest['config']['candidates']
    ranking = sorted((c['name'] for c in candidates), key=lambda name: selection_key(name, summaries['val'][name]))
    if ranking != selected['ranking'] or ranking[0] != selected['candidate']['name']:
        raise ValueError('valの評価値と選定結果が不一致')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'midi').mkdir(exist_ok=True)
    checked, figures, manifests, diagnostics = {}, [], {}, {}
    category_counts = {}
    name = selected['candidate']['name']
    for split in ('val', 'test', 'itest'):
        directory = args.run/split
        manifest = json.loads((directory/'manifest.json').read_text())
        if file_hash(directory/'inputs.pkl.gz') != manifest['inputs_sha256']:
            raise ValueError('保存入力が変更された')
        inputs = {item['case_id']: item for item in read_pickle(directory/'inputs.pkl.gz')}
        prepared_inputs = {key: item['prepared'] for key, item in inputs.items()}
        manifests[split] = {key: manifest[key] for key in ('considered', 'accepted', 'excluded', 'exclusion_reasons')}
        count = 0
        usage, stops = Counter(), Counter()
        for path in sorted((directory/name).glob('batch_*.pkl.gz')):
            receipt = json.loads(path.with_suffix('.json').read_text())
            if file_hash(path) != receipt['sha256']:
                raise ValueError('評価バッチが変更された')
            batch_rows, _ = read_evaluation_batch(path, prepared_inputs)
            for row in batch_rows:
                prepared = inputs[row['case_id']]['prepared']
                bar = prepared.original.with_notes(prepared.original.notes)
                for action in row['history']:
                    step = apply_rule(bar, action['name'], original=prepared.original, guards=prepared.guards,
                                      **action['parameters'])
                    if step.status != 'applied':
                        raise ValueError('履歴を再生してルールが適用されない')
                    bar = step.after.with_notes(step.after.notes)
                if asdict(bar) != row['candidate'] or len(row['history']) != row['rule_count']:
                    raise ValueError('保存した最終譜面とルール履歴が不一致')
                actual = evaluate_result(prepared, bar, model, case_id=row['case_id'], source_id=row['source_id'],
                                         rule_count=row['rule_count'], config=evaluation)
                for key in ('status', 'target_success', 'joint_success', 'failure_adjusted_mae', 'metrics'):
                    if actual[key] != row[key]:
                        raise ValueError('評価値の再計算が不一致: ' + key)
                count += 1
                usage.update(action['name'] for action in row['history'])
                stops.update([row['search_statistics']['stopped']])
                category = '達成' if row['joint_success'] else '変更あり・未達' if row['metrics']['changed'] else '無編集・未達'
                bucket = split, category
                if split in ('test', 'itest') and category_counts.get(bucket, 0) < 2:
                    category_counts[bucket] = category_counts.get(bucket, 0)+1
                    stem = f'{split}_{count:05d}'
                    midi(prepared.original, args.output/'midi'/f'{stem}_before.mid')
                    midi(bar, args.output/'midi'/f'{stem}_after.mid')
                    details = dict(split=split, category=category, case_id=row['case_id'], target=prepared.target,
                                   history=row['history'], evaluation=actual)
                    figures.append('<article><h2>'+html.escape(split+' / '+category)+'</h2><p>入力</p>'+
                                   svg(prepared.original)+'<p>出力</p>'+svg(bar)+
                                   f'<p><a href="midi/{stem}_before.mid">入力MIDI</a> / '
                                   f'<a href="midi/{stem}_after.mid">出力MIDI</a></p><details><summary>詳細</summary><pre>'+
                                   html.escape(json.dumps(details, ensure_ascii=False, indent=2))+'</pre></details></article>')
        if count != manifest['accepted']:
            raise ValueError('再検証件数が入力件数と一致しない')
        checked[split] = count
        diagnostics[split] = dict(selected_rule_usage=dict(usage.most_common()), search_stops=dict(stops))
    fields = ('split', 'model', 'n', 'sources', 'valid_rate', 'joint_success_rate', 'source_macro_joint',
              'mae', 'source_macro_mae', 'skeleton_similarity', 'unchanged_rate', 'mean_rule_count', 'mean_seconds')
    table = []
    for split, methods in summaries.items():
        for method, s in methods.items():
            table.append(dict(split=split, model=method, n=s['n'], sources=s['source_macro']['joint_success_rate']['n'],
                              valid_rate=s['valid_rate']['value'], joint_success_rate=s['joint_success_rate']['value'],
                              source_macro_joint=s['source_macro']['joint_success_rate']['mean'],
                              mae=s['failure_adjusted_mae']['mean'], source_macro_mae=s['source_macro']['failure_adjusted_mae']['mean'],
                              skeleton_similarity=s['valid_output_means']['skeleton_similarity']['mean'],
                              unchanged_rate=s['unchanged_rate_valid']['value'], mean_rule_count=s['rule_count_valid']['mean'],
                              mean_seconds=s['runtime_seconds']['mean']))
    with (args.output/'comparison.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    write_json(args.output/'verification.json', dict(selected_model=name, replayed_and_metrics_matched=checked,
                                                   manifests=manifests, diagnostics=diagnostics, gallery_examples=len(figures)))
    def value(v):
        return html.escape(f'{v:.6f}' if type(v) is float else str(v))
    contents = '<table><tr>'+''.join(f'<th>{key}</th>' for key in fields)+'</tr>'
    contents += ''.join('<tr>'+''.join(f'<td>{value(row[key])}</td>' for key in fields)+'</tr>' for row in table)+'</table>'
    (args.output/'index.html').write_text(
        '<!doctype html><html lang="ja"><meta charset="utf-8"><title>BeamSearch比較実験</title>'
        '<style>body{font:15px sans-serif;margin:24px}table{border-collapse:collapse}th,td{border:1px solid #ccc;'
        'padding:6px}article{max-width:1000px}svg{width:100%}pre{white-space:pre-wrap}</style>'
        '<h1>BeamSearch比較実験</h1><p>valで選定した候補：'+html.escape(name)+'</p>'
        '<p>第一指標は曲単位の同時達成率、第二指標は曲単位の失敗補正MAE。test/itestでは選定後の設定を使用。</p>'
        '<p>自動指標は研究上の定義です。人手評価は未実施。以下の譜面は達成・変更あり未達・無編集未達を'
        '各分割から最大2例ずつ掲載し、結果全体の代表性を主張しません。MIDIは奏法専用のアーティキュレーションを再現しません。</p>'
        + contents + ''.join(figures)+'</html>', encoding='utf-8')
    if args.export_model:
        export_selected_settings(selected, model, args.export_model)
    print(json.dumps(dict(selected=name, verified=checked, report=str(args.output/'index.html')), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('output/beamsearch_step5'))
    parser.add_argument('--output', type=Path, default=Path('output/beamsearch_step5/review'))
    parser.add_argument('--meta', type=Path, default=Path('data/metadata/drum_collection_difficulty_v2'))
    parser.add_argument('--export-model', type=Path, help='推論用の選定設定JSONの保存先')
    args = parser.parse_args()
    try:
        report(args)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
