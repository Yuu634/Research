"""CPU再評価のバッチ集約と、未学習モデルの扱いを確認する。"""
from copy import deepcopy
from types import SimpleNamespace
import json

import pytest

from scripts.evaluate.refresh_test_metrics import aggregate_neural, refresh_neural
from src.common.comparison import evaluate_output
from src.common.examples import make_bar
from src.common.inputs import bar_to_payload, prepare_input
from src.common.representation import to_tokens
from src.beamsearch.scoring import ScoringModel
from src.data.drum_difficulty_v2 import ATTRS


def test_aggregate_neural_recomputes_correlation_and_weights_losses_by_tokens():
    original = make_bar([(0, 'kick'), (24, 'snare')])
    scorer = ScoringModel({a: [.5, 1, 2, 4, 8, 16] for a in ATTRS}, to_tokens(original))
    prepared = prepare_input(dict(schema_version=1, original=bar_to_payload(original), previous=None,
        target=dict(mode='absolute', values=scorer.difficulty(original).normalized)), scorer)
    base = evaluate_output(prepared, original, scorer, case_id='base', source_id='song')
    rows = []
    for i, target in enumerate((1, 3, 5)):
        row = deepcopy(base)
        row['case_id'] = str(i)
        for key in ATTRS[:5]:
            row['target'][key] = target / 7
            row['metrics']['difficulty']['classes'][key] = target + 1
        rows.append(row)
    batches = [dict(rows=rows[:1], log=dict(tokens=3, n=1, beamsearch=None,
                    musemorphose=dict(reconstruction=2., kl=.1, kl_raw=.1))),
               dict(rows=rows[1:], log=dict(tokens=9, n=2, beamsearch=None,
                    musemorphose=dict(reconstruction=4., kl=.4, kl_raw=.4)))]
    summary, actual, losses = aggregate_neural(iter(batches))
    assert actual == rows
    assert summary['validation_reconstruction'] == 3.5
    assert losses['musemorphose']['kl'] == pytest.approx(.3)
    assert losses['musemorphose']['total'] == pytest.approx(3.503)
    assert summary['label_distance'] == dict(mean=1., n=3)
    assert all(value['coefficient'] == 1. for value in summary['attribute_spearman'].values())
    assert losses['tokens'] == 12


def test_missing_checkpoint_is_recorded_without_fabricated_metrics(tmp_path):
    status = {}
    refresh_neural(tmp_path, SimpleNamespace(scope='all'), status)
    assert status[str(tmp_path)]['status'] == 'unavailable'
    assert not list(tmp_path.glob('*summary.json'))


@pytest.mark.parametrize('scope', ['all', 'selected'])
def test_evaluation_filters_variant_in_mixed_experiment(tmp_path, monkeypatch, scope):
    from scripts.evaluate import refresh_test_metrics as refresh
    winners = {}
    for variant in ('musemorphose', 'beam_muse'):
        directory = tmp_path/variant
        directory.mkdir()
        (directory/'best.pt').write_bytes(b'fixture')
        (directory/'completed.json').write_text(json.dumps(dict(trial=dict(variant=variant))))
        winners[variant] = dict(name=variant)
    (tmp_path/'selection.json').write_text(json.dumps(dict(winners=winners)))
    called = []
    def evaluate(root, checkpoint, split, workers, batch_size):
        called.append(checkpoint.parent.name)
        return dict(n=1), []
    monkeypatch.setattr(refresh, 'evaluate_checkpoint', evaluate)
    options = SimpleNamespace(scope=scope, variant='beam_muse', splits=['itest'], workers=1, batch_size=1, root=tmp_path)
    refresh.refresh_neural(tmp_path, options, {})
    assert called == ['beam_muse']
    aggregate = json.loads((tmp_path/'itest_all_models.json').read_text())
    comparison = json.loads((tmp_path/'itest_comparison.json').read_text())
    assert set(aggregate) == {'beam_muse'} and set(comparison) == {'beam_muse'}
