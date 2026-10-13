"""計算結果だけをJSON LinesとPNGへ保存する。学習・探索の判定には使用しない。"""
from collections import defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path

from data.pairdata.io import write_json


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def append_loss(path, record):
    """1回の計算を1行で即時保存。非有限値はJSONのnullで記録する。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(_finite(record), ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')


class TrialLossLog:
    def __init__(self, directory, configuration, *, reset=False, log_interval=1):
        if type(log_interval) is not int or log_interval < 1:
            raise ValueError('log_intervalは正整数')
        self.log_interval = log_interval
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json(self.directory / 'config.json', configuration)
        for name in ('train_loss.log', 'eval_loss.log'):
            path = self.directory / name
            if reset or not path.exists():
                path.write_text('', encoding='utf-8')
        self.beam_seen = set()

    def should_log(self, step):
        return step % self.log_interval == 0

    def write(self, phase, **record):
        if phase == 'train' and record.get('step') is not None and not self.should_log(record['step']):
            return
        name = 'train_loss.log' if phase == 'train' else 'eval_loss.log'
        append_loss(self.directory / name, dict(phase=phase, **record))

    def intermediate(self, examples, scorer, variant, phase, epoch, step):
        if variant != 'beam_muse' or (phase == 'train' and not self.should_log(step)):
            return
        for example in examples:
            key = phase, example.case_id
            if key in self.beam_seen:
                continue
            prepared = example.prepared
            score = scorer.score(prepared.original, example.intermediate, prepared.target,
                                 previous=prepared.previous, guards=prepared.guards, rule_count=len(example.history))
            self.write(phase, event='beamsearch_score', epoch=epoch, step=step, case_id=example.case_id,
                       source='cached_intermediate', musemorphose=None, beamsearch=asdict(score))
            self.beam_seen.add(key)

    def plot(self, *, best_epoch=None):
        plot_loss_curves(self.directory, best_epoch=best_epoch)


class MetricMeans:
    """有限な数値項目を逐次集約する。非数値の一覧は保持しない。"""

    def __init__(self):
        self.sums = defaultdict(float)
        self.counts = defaultdict(float)
        self.children = {}

    def add(self, values, weight=1):
        for key, value in values.items():
            if isinstance(value, dict):
                self.children.setdefault(key, MetricMeans()).add(value, weight)
            elif value is None or isinstance(value, (int, float)):
                self.sums[key] += 0
                if value is not None and math.isfinite(value):
                    self.sums[key] += value * weight
                    self.counts[key] += weight

    def means(self):
        return {key: value / self.counts[key] if self.counts[key] else None
                for key, value in self.sums.items()} | {
                    key: child.means() for key, child in self.children.items()}


def _read_json(path, default):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def _plot_data(directory, best_epoch=None):
    """ログを逐次集約する。旧検証ログより分割全体の集約値を優先する。"""
    history = _read_json(directory / 'history.json', [])
    epochs = {row['epoch']: row['steps'] for row in history}
    if best_epoch is None:
        best_epoch = _read_json(directory / 'completed.json', {}).get('epoch')
    losses, beam, auxiliary = defaultdict(list), defaultdict(list), defaultdict(float)
    legacy, summaries = {}, {}
    beam_means = {}
    betas = defaultdict(set)
    for name in ('train_loss.log', 'eval_loss.log'):
        path = directory / name
        if not path.exists():
            continue
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                phase, event = row['phase'], row['event']
                epoch = row.get('epoch')
                step = row.get('step')
                if step is not None and epoch is not None and phase != 'train':
                    epochs.setdefault(epoch, step)
                step = step if step is not None else epochs.get(epoch)
                key = phase, step
                values = row.get('musemorphose')
                if values and event in ('base_loss', 'evaluation_summary'):
                    beta = row.get('beta')
                    if beta is not None and math.isfinite(beta):
                        betas[phase].add(beta)
                    if phase == 'train':
                        for metric, value in values.items():
                            if step is not None:
                                losses[metric, phase].append((step, value))
                    elif event == 'evaluation_summary':
                        summaries[key] = values
                    else:
                        legacy.setdefault(key, MetricMeans()).add(values)
                if event == 'auxiliary_loss' and step is not None:
                    value = row['auxiliary']['scaled_total']
                    if value is not None and math.isfinite(value):
                        auxiliary[phase, step] += value
                values = row.get('beamsearch')
                if values and event in ('beamsearch_score', 'evaluation_summary'):
                    beam_means.setdefault((phase, step, event), MetricMeans()).add(values)
    # 旧形式の1譜面1行ログはstepごとに平均。履歴に残るトークン加重RCがあれば使用する。
    for key, means in legacy.items():
        if key in summaries:
            continue
        values = means.means()
        phase, step = key
        record = next((r for r in history if phase == 'val' and r['steps'] == step), None)
        if record is not None:
            values['reconstruction'] = record['validation']['validation_reconstruction']
            if len(betas[phase]) == 1 and values.get('kl') is not None:
                values['total'] = values['reconstruction'] + next(iter(betas[phase])) * values['kl']
        summaries[key] = values
    for (phase, step), values in summaries.items():
        for metric, value in values.items():
            losses[metric, phase].append((step, value))
    for (phase, step, event), means in beam_means.items():
        if event != 'evaluation_summary' and (phase, step, 'evaluation_summary') in beam_means:
            continue
        for metric, value in means.means().items():
            if not isinstance(value, dict):
                beam[metric, phase].append((step, value))
    return losses, beam, auxiliary, betas, epochs.get(best_epoch), best_epoch


def _new_figure(columns, rows=1):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=((4 if columns == 5 else 6) * columns, 4.5 * rows), constrained_layout=True)
    FigureCanvasAgg(figure)
    return figure, figure.subplots(rows, columns, squeeze=False)


_PHASE_STYLES = dict(train=('#1f77b4', '-'), val=('#ff7f0e', '--'),
                     test=('#2ca02c', 'D'), itest=('#9467bd', 's'))


def _draw_series(axis, points, phase, *, label=None, raw=False):
    # 欠損はNaNとして線を切る。間引いても横軸の実stepを保持する。
    points = sorted((x, float('nan') if y is None else y) for x, y in points if x is not None)
    if not points:
        return
    stride = max(1, math.ceil((len(points) - 1) / 1999))
    selected = points[::stride]
    if selected[-1] != points[-1]:
        selected.append(points[-1])
    color, style = _PHASE_STYLES[phase]
    label = label or phase
    if phase in ('test', 'itest'):
        axis.plot(*zip(*selected), color=color, marker=style, linestyle='None', label=label)
    else:
        axis.plot(*zip(*selected), color=color, linestyle=':' if raw else style,
                  marker='o' if phase == 'val' else None, markersize=3,
                  linewidth=1.2, alpha=.6 if raw else 1., label=label)


def _finish_axis(axis, title, xlabel, *, best_step=None, best_epoch=None):
    has_data = bool(axis.lines or axis.patches)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel('Loss / score')
    axis.grid(alpha=.25)
    if has_data and best_step is not None:
        axis.axvline(best_step, color='#444444', linestyle=':', linewidth=1.2,
                     label=f'Best checkpoint (epoch {best_epoch}, step {best_step})')
    if has_data:
        axis.legend(fontsize=8)
    else:
        axis.text(.5, .5, 'No recorded training loss' if xlabel == 'Training step' else 'No recorded scores',
                  ha='center', va='center', transform=axis.transAxes)


def _save_figure(figure, directory, name):
    temporary = directory / (name + '.tmp')
    figure.savefig(temporary, format='png', dpi=150)
    temporary.replace(directory / name)
    figure.clear()


def plot_loss_curves(directory, *, best_epoch=None):
    """実stepを共有した損失曲線と、独立した探索スコア・補助損失を保存する。"""
    directory = Path(directory)
    losses, beam, auxiliary, betas, best_step, best_epoch = _plot_data(directory, best_epoch)
    figure, axes = _new_figure(3)
    beta_text = '; '.join(f'{phase} beta=' + '/'.join(f'{b:g}' for b in sorted(values))
                          for phase, values in sorted(betas.items()))
    figure.suptitle(directory.name + ('\n' + beta_text if beta_text else ''), fontsize=13)
    for axis, metric, title in zip(axes.flat, ('reconstruction', 'kl', 'total'),
                                   ('Reconstruction loss', 'KL loss (before beta; after free bits)',
                                    'Base total loss (reconstruction + beta * KL)\nExcludes policy loss')):
        for phase in _PHASE_STYLES:
            points = losses.get((metric, phase), [])
            # holdoutは採用checkpoint上の独立した点として表示し、曲線には接続しない。
            if phase in ('test', 'itest'):
                points = [(x if x is not None else best_step, y) for x, y in points]
            _draw_series(axis, points, phase)
            if metric == 'kl':
                raw = losses.get(('kl_raw', phase), [])
                if raw != losses.get(('kl', phase), []):
                    raw = [(x if x is not None else best_step, y) for x, y in raw]
                    _draw_series(axis, raw, phase, label=phase + ' KL raw', raw=True)
        _finish_axis(axis, title, 'Training step', best_step=best_step, best_epoch=best_epoch)
        axis.set_ylabel('Loss')
    _save_figure(figure, directory, 'loss_curve.png')

    if auxiliary:
        figure, axes = _new_figure(1)
        figure.suptitle(directory.name)
        for phase in _PHASE_STYLES:
            points = [(step, value) for (p, step), value in auxiliary.items() if p == phase]
            _draw_series(axes[0, 0], points, phase)
        _finish_axis(axes[0, 0], 'Auxiliary policy loss (sum per step; signed, not a quality score)',
                     'Training step', best_step=best_step, best_epoch=best_epoch)
        _save_figure(figure, directory, 'auxiliary_loss_curve.png')

    if beam:
        figure, axes = _new_figure(5, 2)
        neural = bool(losses)
        figure.suptitle(directory.name + ('\nBeamSearch cached intermediate scores (mean per step)' if neural
                                        else '\nBeamSearch selected-result scores (no gradient training)'))
        metrics = ('difficulty_loss', 'skeleton_similarity', 'rule_count', 'edit_distance', 'total')
        titles = ('Difficulty loss (lower is better)', 'Skeleton similarity (higher is better)',
                  'Rule count', 'Edit distance', 'Total score (higher is better)')
        for col, (metric, title) in enumerate(zip(metrics, titles)):
            for phase in ('train', 'val'):
                _draw_series(axes[0, col], beam.get((metric, phase), []), phase)
            _finish_axis(axes[0, col], title, 'Training step' if neural else 'Input step')
            for phase in ('val', 'test', 'itest'):
                points = beam.get((metric, phase), [])
                # val全epochを棒にせず、採用epochの値（なければ最新）だけを表示する。
                selected = next((y for x, y in points if best_step is not None and x == best_step),
                                points[-1][1] if points else None)
                if selected is not None and math.isfinite(selected):
                    axes[1, col].bar(phase, selected, color=_PHASE_STYLES[phase][0], label=phase)
            _finish_axis(axes[1, col], title, 'Evaluation split (best checkpoint for neural models)')
        _save_figure(figure, directory, 'beamsearch_scores.png')
