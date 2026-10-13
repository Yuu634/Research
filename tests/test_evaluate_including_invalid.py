"""制約違反出力の測定と、算出不能・再現不一致の扱いを検証する。"""
from copy import deepcopy
from dataclasses import replace

import pytest

from scripts.evaluate.evaluate_including_invalid import assert_reproduced, measure_output, summarize
from src.beamsearch.scoring import ScoringModel
from src.common.comparison import evaluate_output
from src.common.examples import make_bar
from src.common.inputs import bar_to_payload, prepare_input
from src.common.representation import to_tokens
from src.data.drum_difficulty_v2 import ATTRS


def test_protected_note_deletion_is_measured_without_changing_invalid_status():
    original = make_bar([(0, 'kick'), (24, 'snare')])
    model = ScoringModel({a: [.5, 1, 2, 4, 8, 16] for a in ATTRS}, to_tokens(original))
    prepared = prepare_input(dict(schema_version=1, original=bar_to_payload(original), previous=None,
        target=dict(mode='absolute', values=model.difficulty(original).normalized)), model)
    candidate = replace(prepared.original, notes=tuple(n for n in prepared.original.notes if n.instrument != 'kick'))
    prepared = replace(prepared, target=model.difficulty(candidate).normalized)
    saved = evaluate_output(prepared, candidate, model, case_id='invalid', source_id='song')
    assert saved['status'] == 'invalid' and saved['metrics'] is None
    measured = measure_output(prepared, candidate, model, saved)
    result = summarize([measured], .5/7)
    assert measured['status'] == 'invalid' and saved['metrics'] is None
    assert result['all_outputs']['label_distance'] == dict(mean=0., n=1)
    assert result['all_outputs']['valid_rate'] == 0
    assert result['all_outputs']['target_success_ignoring_validity']['value'] == 1
    assert result['invalid_only']['measured_n'] == 1
    assert result['valid_only']['label_distance'] == dict(mean=None, n=0)


def test_missing_outputs_remain_in_success_denominator_and_do_not_become_zero_error():
    keys = ATTRS[:5]
    valid = dict(case_id='v', source_id='song', status='valid', reasons=[],
        target={k: 1/7 for k in keys}, measurement_error=None,
        metrics=dict(difficulty=dict(classes={k: 1 for k in keys}, normalized={k: 1/7 for k in keys}),
                     rhythm_similarity=1., hihat_similarity=None))
    invalid = deepcopy(valid)
    invalid.update(case_id='i', status='invalid', reasons=['protected_note:n0'])
    for k in keys:
        invalid['metrics']['difficulty']['classes'][k] = 3
        invalid['metrics']['difficulty']['normalized'][k] = 3/7
    failed = dict(valid, case_id='f', status='failed', reasons=['no_output'])
    failed = measure_output(None, None, None, failed)
    result = summarize([valid, invalid, failed], .5/7)
    all_outputs = result['all_outputs']
    assert all_outputs['n'] == 3 and all_outputs['measured_n'] == 2 and all_outputs['unmeasurable_n'] == 1
    assert all_outputs['label_distance'] == dict(mean=1., n=2)
    assert all_outputs['target_success_ignoring_validity'] == dict(numerator=1, denominator=3, value=1/3)
    assert all_outputs['hihat_similarity'] == dict(mean=None, n=0)
    assert all_outputs['attribute_spearman']['s_tech']['reason'] == 'constant_target'
    assert result['invalid_only']['label_distance'] == dict(mean=2., n=1)


def test_unmeasurable_output_records_reason():
    class Scorer:
        def difficulty(self, candidate, previous):
            raise ValueError('難易度算出不可')
    row = dict(case_id='x', source_id='song', status='invalid', reasons=['bad'], target={})
    prepared = type('Prepared', (), {'previous': None})()
    result = measure_output(prepared, object(), Scorer(), row)
    assert result['metrics'] is None and result['measurement_error'] == '難易度算出不可'


def test_regeneration_must_match_prior_metrics_and_status():
    saved = dict(case_id='x', status='valid', metrics=dict(score=1), elapsed_seconds=1.)
    assert_reproduced(saved, dict(saved, elapsed_seconds=2.))
    for changed in (dict(saved, status='invalid'), dict(saved, metrics=dict(score=2))):
        with pytest.raises(ValueError, match='再生成が従来評価と不一致'):
            assert_reproduced(saved, changed)
