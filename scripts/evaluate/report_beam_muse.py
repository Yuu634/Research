"""学習・選定記録を照合し、人手確認用の比較表をHTMLに出力する。"""
import argparse
import html
import json
from pathlib import Path

from data.pairdata.io import file_hash
from src.beam_muse.data import source_hashes
from src.beam_muse.experiment import expand_trials, selection_key


def report(run):
    selected = json.loads((run/'selection.json').read_text())
    record = json.loads((run/'run.json').read_text())
    if selected['identity'] != record['identity'] or selected['selected_on'] != 'val':
        raise ValueError('実行記録とval選定記録が不一致')
    if record['identity']['sources'] != source_hashes():
        raise ValueError('実験後に実装が変更された')
    trials = expand_trials(record['identity']['config'])
    completed = []
    for trial in trials:
        result = json.loads((run/trial['name']/'completed.json').read_text())
        if (result['identity'] != record['identity'] or result['trial'] != trial or
            file_hash(run/trial['name']/'best.pt') != result['checkpoint_sha256']):
            raise ValueError('trialの設定・重みが実行記録と不一致')
        completed.append(result)
    for variant, winner in selected['winners'].items():
        best = min((r for r in completed if r['trial']['variant'] == variant),
                   key=lambda r: selection_key(r['validation'], r['trial']['name']))
        if best['trial']['name'] != winner['name'] or best['checkpoint_sha256'] != winner['checkpoint_sha256']:
            raise ValueError('選定順位または選定重みが不一致')
    def cell(value):
        return html.escape(f'{value:.6f}' if type(value) is float else str(value))
    def row(label, summary):
        return '<tr>'+''.join('<td>'+cell(v)+'</td>' for v in (label, summary['n'],
                summary['valid_rate']['value'], summary['joint_success_rate']['value'],
                summary['source_macro']['joint_success_rate']['mean'],
                summary['source_macro']['failure_adjusted_mae']['mean'],
                summary['valid_output_means']['skeleton_similarity']['mean']))+'</tr>'
    header = '<tr>'+''.join('<th>'+v+'</th>' for v in ('方式', '入力数', '適格率', '小節同時達成率', '曲平均同時達成率', '曲平均失敗補正MAE', '適格出力の骨格類似度'))+'</tr>'
    parts = ['<h2>全候補のval評価</h2><table>'+header+''.join(row(r['trial']['name'], r['validation']) for r in completed)+'</table>']
    for split in ('val', 'test', 'itest'):
        path = run/f'{split}_comparison.json'
        if path.exists():
            summaries = json.loads(path.read_text())
            parts.append('<h2>'+split+'</h2><table>'+header+''.join(row(k,v) for k,v in summaries.items())+'</table>')
    smoke = record['identity']['smoke']
    title = '接続確認用smoke実験（性能比較ではありません）' if smoke else '3方式の比較実験'
    output = run/'review.html'
    output.write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><title>'+title+'</title>'
                     '<style>body{font:16px sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}'
                     'pre{white-space:pre-wrap}</style><h1>'+title+'</h1><p>失敗・制約違反も分母に含みます。'
                     '骨格類似度は適格出力だけの平均で、nullは対象なしです。人手の自然さ評価は未実施です。</p>'
                     +''.join(parts)+'<h2>選定設定</h2><pre>'+html.escape(json.dumps(selected, ensure_ascii=False, indent=2))+'</pre></html>', encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    try:
        print(report(args.run))
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
