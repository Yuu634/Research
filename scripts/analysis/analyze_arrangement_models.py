"""新MuseMorphose単体／複合モデルの選定記録・評価・重み統計を保存する。"""
import argparse
import html
import json
from pathlib import Path


def weight_statistics(state_dict):
    """学習済み重みの個数・有限性・RMSを構成要素ごとに集約する。"""
    import torch
    groups = {}
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            continue
        value = tensor.detach().double()
        if not torch.isfinite(value).all():
            raise ValueError('重みに非有限値: ' + name)
        group = groups.setdefault(name.split('.')[0], dict(elements=0, squared_norm=0.))
        group['elements'] += value.numel()
        group['squared_norm'] += float(value.square().sum())
    return {name: dict(elements=g['elements'], l2_norm=g['squared_norm']**.5,
                       rms=(g['squared_norm']/g['elements'])**.5 if g['elements'] else None)
            for name, g in sorted(groups.items())}


def analyze(run, variant, output, scope='selected'):
    import torch
    torch.set_num_threads(1)
    from data.pairdata.io import file_hash, write_json
    from src.common.provenance import implementation_hashes, validate_sources
    record = json.loads((run/'run.json').read_text())
    selected = json.loads((run/'selection.json').read_text())
    if selected['identity'] != record['identity'] or selected['selected_on'] != 'val':
        raise ValueError('実行記録とval選定記録が不一致')
    validate_sources(record['identity']['sources'], implementation_hashes('beam_muse'))
    winner = selected['winners'][variant]
    paths = [run/winner['name']/'best.pt'] if scope == 'selected' else sorted(run.glob('*/best.pt'))
    results = []
    for path in paths:
        completed = json.loads((path.parent/'completed.json').read_text())
        if completed['trial']['variant'] != variant:
            continue
        checksum = file_hash(path)
        if checksum != completed['checkpoint_sha256'] or completed['identity'] != record['identity']:
            raise ValueError('学習完了記録と重みが不一致: ' + str(path))
        if path.parent.name == winner['name'] and checksum != winner['checkpoint_sha256']:
            raise ValueError('val選定記録と重みが不一致')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if (checkpoint['trial'] != completed['trial'] or checkpoint['identity'] != completed['identity']
                or checkpoint['epoch'] != completed['epoch']):
            raise ValueError('チェックポイント内の設定が学習完了記録と不一致')
        summaries = dict(val=completed['validation'])
        for split in ('test', 'itest'):
            own = path.parent/f'{split}_summary.json'
            comparison = run/f'{split}_comparison.json'
            if own.exists():
                summaries[split] = json.loads(own.read_text())
            elif path.parent.name == winner['name'] and comparison.exists():
                summaries[split] = json.loads(comparison.read_text()).get(variant)
        results.append(dict(name=path.parent.name, selected=path.parent.name == winner['name'],
                            checkpoint_sha256=checksum, best_epoch=completed['epoch'], trial=completed['trial'],
                            evaluations=summaries, weights=weight_statistics(checkpoint['state_dict'])))
    if not results:
        raise ValueError('指定方式の学習済みモデルがありません')
    output.mkdir(parents=True, exist_ok=True)
    report = dict(variant=variant, scope=scope, selected_on='val', winner=winner['name'], models=results,
                  analysis='構成要素別の保存済み重み統計と評価値。旧モデルのBias・Attention解析とは別処理。')
    write_json(output/'model_analysis.json', report)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    for result in results:
        figure = Figure(figsize=(12, 4), constrained_layout=True)
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        axis.bar(list(result['weights']), [g['rms'] for g in result['weights'].values()])
        axis.set_title(result['name'] + ' / saved-weight RMS by component')
        axis.set_ylabel('RMS')
        axis.tick_params(axis='x', labelrotation=30)
        figure.savefig(output/(result['name']+'_weight_rms.png'), dpi=150)
        figure.clear()
    (output/'index.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8">'
        '<title>モデル解析</title><h1>選定結果・評価・重み統計</h1>'
        + ''.join('<h2>'+html.escape(r['name'])+'</h2><img style="max-width:100%" src="'
                  +r['name']+'_weight_rms.png">' for r in results)
        + '<pre>'+html.escape(json.dumps(report, ensure_ascii=False, indent=2))+'</pre>', encoding='utf-8')
    return output/'index.html'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('musemorphose', 'beam_muse'), required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scope', choices=('selected', 'all'), default='selected')
    args = parser.parse_args()
    try:
        print(analyze(args.run, args.model, args.output, args.scope))
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
