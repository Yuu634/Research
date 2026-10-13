"""損失ログの粒度・再集約・学習計算からの独立性を確認する。"""
from dataclasses import asdict
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import pytest

from data.pairdata.io import file_hash, read_pickle, write_json, write_pickle
from scripts.evaluate import compare_arrangement_models as comparison
from src.beamsearch.beam_search import SearchConfig, standard_actions
from src.beamsearch.scoring import ScoringModel
from src.common.evaluation import EvaluationConfig
from src.common.examples import make_bar
from src.common.inputs import bar_to_payload, prepare_input
from src.common.loss_logging import MetricMeans, TrialLossLog, append_loss, plot_loss_curves
from src.common.representation import to_tokens
from src.data.drum_difficulty_v2 import ATTRS


def test_nonfinite_logs_are_standard_json(tmp_path):
    path = tmp_path/'loss.log'
    append_loss(path, dict(total=float('inf'), parts=[float('nan'), 1.]))
    assert path.read_text() == '{"total":null,"parts":[null,1.0]}\n'


def test_intermediate_logged_once_per_phase_and_input(tmp_path):
    original = make_bar([(0, 'kick'), (24, 'snare')])
    scorer = ScoringModel({a: [.5, 1, 2, 4, 8, 16] for a in ATTRS}, to_tokens(original))
    prepared = prepare_input(dict(schema_version=1, original=bar_to_payload(original), previous=None,
        target=dict(mode='absolute', values=scorer.difficulty(original).normalized)), scorer)
    example = SimpleNamespace(case_id='a', prepared=prepared, intermediate=original, history=[])
    log = TrialLossLog(tmp_path, {})
    for epoch in (1, 2):
        log.intermediate([example], scorer, 'musemorphose', 'train', epoch, epoch)
        log.intermediate([example, example], scorer, 'beam_muse', 'train', epoch, epoch)
        log.intermediate([example], scorer, 'beam_muse', 'val', epoch, epoch)
    assert len((tmp_path/'train_loss.log').read_text().splitlines()) == 1
    assert len((tmp_path/'eval_loss.log').read_text().splitlines()) == 1


def test_beam_batch_logs_selected_results_and_rebuilds_without_duplicates(tmp_path, monkeypatch):
    original = make_bar([(0, 'kick'), (24, 'snare')])
    vocabulary = {t: i for i, t in enumerate(to_tokens(original))}
    bounds = {a: [.5, 1, 2, 4, 8, 16] for a in ATTRS}
    scorer = ScoringModel(bounds, vocabulary)
    prepared = prepare_input(dict(schema_version=1, original=bar_to_payload(original), previous=None,
        target=dict(mode='absolute', values=scorer.difficulty(original).normalized)), scorer)
    inputs = [dict(case_id=str(i), source_id='song', prepared=prepared) for i in range(2)]
    method = dict(name='example', score={}, search=asdict(SearchConfig(max_depth=0)))
    config = dict(evaluation=asdict(EvaluationConfig()), action_set='standard_v1')
    meta = tmp_path/'metadata'
    meta.mkdir()
    (meta/'drum_vocab.pkl').write_bytes(pickle.dumps(dict(token2idx=vocabulary)))
    (meta/'bounds.json').write_text(json.dumps(dict(boundaries=bounds)))
    monkeypatch.setattr(comparison, 'WORKER', dict(bounds=bounds, vocabulary=vocabulary,
        inputs=inputs, evaluation=EvaluationConfig(), actions=standard_actions()))
    folder = tmp_path/'val'/method['name']
    folder.mkdir(parents=True)
    path = folder/'batch_000000.pkl.gz'
    write_pickle(tmp_path/'val'/'inputs.pkl.gz', inputs)
    comparison.run_batch(method, 0, 2, path)
    records = [json.loads(line) for line in path.with_suffix('.loss.log').read_text().splitlines()]
    assert len(records) == 2
    assert all(r['beamsearch']['rule_count'] == 0 and r['source'] == 'selected_result' for r in records)
    rows = read_pickle(path)
    assert [r['case_id'] for r in records] == [r['case_id'] for r in rows]
    # 指標追加前の保存済みバッチでも、探索を再実行せず類似度を補完する。
    for row in rows:
        row['metrics'].pop('rhythm_similarity')
        row['metrics'].pop('hihat_similarity')
    write_pickle(path, rows)
    receipt = json.loads(path.with_suffix('.json').read_text())
    write_json(path.with_suffix('.json'), receipt | dict(sha256=file_hash(path)))
    write_json(tmp_path/'val'/'manifest.json', dict(accepted=2, considered=2, excluded=0,
        inputs_sha256=file_hash(tmp_path/'val'/'inputs.pkl.gz'), inputs=[dict(case_id=str(i)) for i in range(2)]))
    before = path.read_bytes()
    args = SimpleNamespace(output=tmp_path, meta=meta, pairdata=Path('fixture'), batch_size=2)
    comparison.save_model_logs(args, method, config)
    model_folder = tmp_path/method['name']
    assert (model_folder/'train_loss.log').read_text() == ''
    assert len((model_folder/'eval_loss.log').read_text().splitlines()) == 1
    comparison.save_model_logs(args, method, config)
    assert len((model_folder/'eval_loss.log').read_text().splitlines()) == 1
    summary_log = (model_folder/'eval_loss.log').read_text()
    path.with_suffix('.loss.log').unlink()
    comparison.save_model_logs(args, method, config)
    restored = [json.loads(line) for line in (model_folder/'eval_loss.log').read_text().splitlines()]
    assert len(restored) == 1
    assert restored[0]['event'] == 'evaluation_summary' and restored[0]['n'] == 2
    assert restored[0]['evaluation']['n'] == 2
    result = restored[0]['evaluation']
    assert result['rhythm_similarity'] == dict(mean=1., n=2)
    assert result['hihat_similarity'] == dict(mean=None, n=0)
    assert result['label_distance'] == dict(mean=0., n=2)
    assert all(value['reason'] == 'constant_target' for value in result['attribute_spearman'].values())
    comparison.evaluate_split(args, 'val', [method], config, scorer, vocabulary)
    saved_summary = json.loads((tmp_path/'val'/'summaries.json').read_text())
    assert saved_summary[method['name']] == result
    assert restored[0]['beamsearch']['total'] == pytest.approx(
        sum(r['beamsearch']['total'] for r in records)/2)
    assert (model_folder/'eval_loss.log').read_text() == summary_log
    assert path.read_bytes() == before
    assert (model_folder/'loss_curve.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert json.loads((model_folder/'config.json').read_text())['method'] == method
    comparison.WORKER['inputs'] = [dict(case_id=str(i), source_id='song', prepared=prepared) for i in range(5)]
    train_folder = tmp_path/'train'/method['name']
    train_folder.mkdir(parents=True)
    for start, end in ((0, 3), (3, 5)):
        comparison.run_batch(method, start, end, train_folder/f'batch_{start:06d}.pkl.gz')
    args.log_interval = 2
    comparison.save_model_logs(args, method, config)
    train_records = [json.loads(line) for line in (model_folder/'train_loss.log').read_text().splitlines()]
    assert [r['step'] for r in train_records] == [2, 4]
    assert (model_folder/'eval_loss.log').read_text() == summary_log


@pytest.mark.parametrize('interval', [0, -1, 1.5, True])
def test_invalid_log_interval(tmp_path, interval):
    with pytest.raises(ValueError, match='正整数'):
        TrialLossLog(tmp_path, {}, log_interval=interval)


def test_metric_means_excludes_missing_and_nonfinite_values():
    means = MetricMeans()
    means.add(dict(total=2., missing=None, parts=dict(value=4.), valid=True))
    means.add(dict(total=4., missing=float('inf'), parts=dict(value=8.), valid=False))
    assert means.means() == dict(total=3., missing=None, parts=dict(value=6.), valid=.5)


def test_training_log_interval_does_not_filter_evaluation(tmp_path):
    log = TrialLossLog(tmp_path, {}, log_interval=3)
    for step in range(1, 8):
        log.write('train', event='base_loss', step=step)
    log.write('val', event='evaluation_summary', step=7)
    assert [json.loads(line)['step'] for line in (tmp_path/'train_loss.log').read_text().splitlines()] == [3, 6]
    assert len((tmp_path/'eval_loss.log').read_text().splitlines()) == 1


@pytest.fixture
def captured_plots(monkeypatch):
    """PNG保存直前のデータを確認し、画像の見た目だけに依存しない。"""
    from matplotlib.figure import Figure
    captured = {}
    def capture(figure, path, **kwargs):
        captured[Path(path).name] = dict(
            title=figure._suptitle.get_text(),
            axes=[dict(title=ax.get_title(), xlabel=ax.get_xlabel(),
                       lines={line.get_label(): dict(x=list(line.get_xdata()), y=list(line.get_ydata()),
                                                    style=line.get_linestyle()) for line in ax.lines},
                       bars=[patch.get_height() for patch in ax.patches]) for ax in figure.axes])
        Path(path).write_bytes(b'PNG fixture')
    monkeypatch.setattr(Figure, 'savefig', capture)
    return captured


def _loss(rec, kl=2.):
    return dict(reconstruction=rec, kl=kl, kl_raw=kl, total=rec + .01 * kl)


def test_loss_plot_shares_steps_and_marks_selected_checkpoint(tmp_path, captured_plots):
    log = TrialLossLog(tmp_path, {})
    for step in (500, 1000, 1500):
        log.write('train', event='base_loss', step=step, musemorphose=_loss(.5), beta=.01)
    for epoch, step, rec in ((1, 750, .4), (2, 1500, .3)):
        log.write('val', event='evaluation_summary', epoch=epoch, step=step,
                  musemorphose=_loss(rec), beta=.01)
    # 最小lossのepoch 2ではなく、実際に選定したepoch 1を使う。
    write_json(tmp_path/'completed.json', dict(epoch=1))
    write_json(tmp_path/'history.json', [dict(epoch=1, steps=750), dict(epoch=2, steps=1500)])
    log.write('test', event='evaluation_summary', epoch=1, step=None, musemorphose=_loss(.45), beta=.01)
    log.write('itest', event='evaluation_summary', epoch=1, step=None, musemorphose=_loss(.46), beta=.01)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    log.plot()
    plot = captured_plots['loss_curve.png.tmp']
    assert 'train beta=0.01' in plot['title']
    for ax in plot['axes']:
        assert ax['xlabel'] == 'Training step'
        assert ax['lines']['train']['x'] == [500, 1000, 1500]
        assert ax['lines']['val']['x'] == [750, 1500]
        assert ax['lines']['test']['x'] == [750]
        assert ax['lines']['itest']['style'] == 'None'
        assert ax['lines']['Best checkpoint (epoch 1, step 750)']['x'] == [750, 750]
    assert plot['axes'][2]['lines']['val']['y'] == pytest.approx([.42, .32])
    assert not (tmp_path/'beamsearch_scores.png').exists()
    assert all((tmp_path/name).read_bytes() == content for name, content in before.items())


def test_legacy_evaluation_aggregation_and_summary_precedence(tmp_path, captured_plots):
    log = TrialLossLog(tmp_path, {})
    for rec in (.2, .8):
        log.write('val', event='base_loss', step=100, musemorphose=_loss(rec), beta=.01)
        log.write('val', event='base_loss', step=200, musemorphose=_loss(rec), beta=.01)
    log.write('val', event='evaluation_summary', step=200, musemorphose=_loss(.1), beta=.01)
    write_json(tmp_path/'history.json', [dict(epoch=1, steps=100, validation=dict(validation_reconstruction=.6))])
    log.plot(best_epoch=1)
    axes = captured_plots['loss_curve.png.tmp']['axes']
    assert axes[0]['lines']['val']['x'] == [100, 200]
    assert axes[0]['lines']['val']['y'] == pytest.approx([.6, .1])
    assert axes[2]['lines']['val']['y'] == pytest.approx([.62, .12])


def test_separate_scores_and_sum_of_policy_contributions(tmp_path, captured_plots):
    log = TrialLossLog(tmp_path, {})
    log.write('train', event='base_loss', step=500, musemorphose=_loss(.3), beta=.01)
    for value in (-2., 3.):
        log.write('train', event='auxiliary_loss', step=500, auxiliary=dict(scaled_total=value))
        log.write('train', event='beamsearch_score', step=500,
                  beamsearch=dict(total=value, difficulty_loss=.1, rule_count=2))
    log.write('val', event='evaluation_summary', step=1000, beamsearch=dict(total=.5))
    log.write('test', event='evaluation_summary', beamsearch=dict(total=.7))
    log.plot()
    assert len(captured_plots['loss_curve.png.tmp']['axes']) == 3
    policy = captured_plots['auxiliary_loss_curve.png.tmp']['axes'][0]['lines']['train']
    assert policy['x'] == [500] and policy['y'] == [1.]
    beam = captured_plots['beamsearch_scores.png.tmp']['axes']
    assert beam[4]['lines']['train']['x'] == [500]
    assert beam[4]['lines']['train']['y'] == [.5]
    assert beam[9]['bars'] == [.5, .7]


def test_plot_downsampling_preserves_real_steps_and_missing_values(tmp_path, captured_plots):
    log = TrialLossLog(tmp_path, {})
    for i in range(4001):
        log.write('train', event='base_loss', step=500*(i+1),
                  musemorphose=dict(reconstruction=None if i == 0 else .2))
    log.plot()
    line = captured_plots['loss_curve.png.tmp']['axes'][0]['lines']['train']
    assert line['x'][0] == 500 and line['x'][-1] == 2000500
    assert len(line['x']) <= 2000
    import math
    assert math.isnan(line['y'][0])


def test_plot_cli_finds_nested_trials_without_changing_logs(tmp_path, monkeypatch):
    from scripts.analysis.plot_loss_curves import main
    log = TrialLossLog(tmp_path/'experiment copy'/'trial', {})
    log.write('val', event='evaluation_summary', beamsearch=dict(total=.5))
    before = (log.directory/'eval_loss.log').read_bytes()
    legacy = tmp_path/'checkpoints_drum_standard-lakh1-kl14'
    legacy.mkdir()
    (legacy/'valloss.txt').write_text('[step 1000] RC: 0.2 | KL: 0.1 | [val] | RC: 0.3 | KL: 0.1\n')
    monkeypatch.setattr('sys.argv', ['plot_loss_curves', '-d', str(tmp_path)])
    main()
    for name in ('loss_curve.png', 'beamsearch_scores.png'):
        assert (log.directory/name).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert (log.directory/'eval_loss.log').read_bytes() == before
    assert (legacy/'loss_curves.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
