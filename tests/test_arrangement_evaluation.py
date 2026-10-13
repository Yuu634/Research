"""評価基準の固定、失敗の分母、属性別判定、曲単位集計を検証する。"""
from dataclasses import replace
import json

import pytest

from src.common.evaluation import EvaluationConfig, evaluate_result, summarize_control_metrics, summarize_results
from src.common.metrics import onset_similarities
from src.common.examples import make_bar
from src.common.inputs import bar_to_payload, prepare_input
from src.common.representation import to_tokens
from src.beamsearch.rules import apply_rule
from src.beamsearch.scoring import DIFFICULTY_KEYS, ScoreConfig, ScoringModel
from src.data.drum_difficulty_v2 import ATTRS


@pytest.fixture
def setup():
    original = make_bar([(p, 'hh_closed') for p in range(0, 96, 6)] +
                        [(0, 'kick'), (48, 'kick'), (24, 'snare'), (72, 'snare')])
    candidate = apply_rule(original, 'hh_16_to_8').after
    vocabulary = set(to_tokens(original)) | set(to_tokens(candidate))
    vocabulary.update(f'<POS_{i}>' for i in range(24))
    model = ScoringModel({a: [.5, 1, 2, 4, 8, 16] for a in ATTRS}, vocabulary)
    payload = dict(schema_version=1, original=bar_to_payload(original), previous=None,
                   target=dict(mode='absolute', values=model.difficulty(candidate).normalized))
    prepared = prepare_input(payload, model)
    candidate = apply_rule(prepared.original, 'hh_16_to_8').after
    return prepared, candidate, model


def evaluate(setup, candidate=None, **kwargs):
    prepared, after, model = setup
    return evaluate_result(prepared, after if candidate is None else candidate, model,
                           case_id=kwargs.pop('case_id', 'case'), source_id=kwargs.pop('source_id', 'song'), **kwargs)


def test_exact_target_retains_skeleton_and_matches_manual_improvement(setup):
    prepared, candidate, _ = setup
    row = evaluate(setup, rule_count=1)
    metric = row['metrics']
    assert row['status'] == 'valid' and row['target_success'] and row['joint_success']
    assert metric['mae'] == 0 and metric['exact_target']
    expected = sum(abs(prepared.target[k]-prepared.original_difficulty.normalized[k]) for k in DIFFICULTY_KEYS)/5
    assert row['baseline_mae'] == pytest.approx(expected)
    assert metric['mae_improvement'] == pytest.approx(expected)
    assert metric['skeleton_similarity'] == 1
    assert metric['edit_distance'] == pytest.approx(8/20)
    assert metric['note_count_delta'] == -8
    assert metric['direction_rate'] == 1
    assert metric['held_attribute_mae'] == 0
    assert metric['rhythm_similarity'] == 1
    assert metric['hihat_similarity'] == pytest.approx(2/3)


def test_unchanged_baseline_is_not_reported_as_progress(setup):
    row = evaluate(setup, candidate=setup[0].original)
    assert row['metrics']['mae_improvement'] == 0
    assert row['metrics']['direction_rate'] == 0
    assert not row['metrics']['changed'] and not row['metrics']['improved']
    assert not row['target_success']


def test_all_attribute_condition_cannot_be_hidden_by_small_mean(setup):
    prepared, candidate, model = setup
    target = dict(prepared.target, s_move=.1)
    row = evaluate_result(replace(prepared, target=target), candidate, model, case_id='case', source_id='song')
    assert row['metrics']['mae'] == pytest.approx(.02)
    assert not row['target_success'] and not row['joint_success']
    assert row['metrics']['within_one_class']


def test_half_class_threshold_boundary_and_exact_match_are_distinct(setup):
    prepared, candidate, model = setup
    for value, success in ((.5/7, True), (.5/7+1e-6, False)):
        target = dict(prepared.target, s_move=value)
        row = evaluate_result(replace(prepared, target=target), candidate, model, case_id='case', source_id='song')
        assert row['target_success'] is success
        assert not row['metrics']['exact_target']


def test_metrics_independent_of_search_weights(setup):
    prepared, candidate, model = setup
    changed_model = ScoringModel(model.boundaries, model.vocabulary, ScoreConfig(
        lambda_difficulty=100, lambda_skeleton=0, lambda_rules=10,
        difficulty_weights=(0, 1, 0, 0, 0), skeleton_weights=(0, 0, 1, 0)))
    a = evaluate(setup, rule_count=1)
    b = evaluate_result(prepared, candidate, changed_model, case_id='case', source_id='song', rule_count=1)
    assert a == b


def test_redundant_rule_count_does_not_change_quality_metrics(setup):
    a = evaluate(setup, rule_count=1)
    b = evaluate(setup, rule_count=3)
    assert a['metrics'] == b['metrics'] and a['joint_success'] == b['joint_success']
    assert a['rule_count'] == 1 and b['rule_count'] == 3


def test_invalid_and_missing_outputs_remain_in_denominator(setup):
    prepared, _, model = setup
    valid = evaluate(setup, case_id='good')
    broken = prepared.original.with_notes(n for n in prepared.original.notes if n.instrument != 'kick')
    invalid = evaluate(setup, candidate=broken, case_id='invalid')
    failed = evaluate_result(prepared, None, model, case_id='failed', source_id='song', failure_reason='timeout')
    summary = summarize_results([valid, invalid, failed])
    assert invalid['status'] == 'invalid' and failed['status'] == 'failed'
    assert invalid['metrics'] is None and failed['metrics'] is None
    assert summary['joint_success_rate'] == dict(value=1/3, numerator=1, denominator=3)
    assert summary['failure_adjusted_mae'] == dict(mean=2/3, n=3)
    assert summary['valid_output_means']['mae'] == dict(mean=0, n=1)
    assert summary['reason_counts']['timeout'] == 1
    assert summary['label_distance'] == dict(mean=0., n=1)
    assert summary['rhythm_similarity'] == dict(mean=1., n=1)
    assert summary['hihat_similarity']['n'] == 1
    assert all(value['n'] == 1 for value in summary['attribute_spearman'].values())
    json.dumps(summary, allow_nan=False)


def test_no_outputs_and_empty_dataset_have_no_fake_perfect_means(setup):
    prepared, _, model = setup
    row = evaluate_result(prepared, None, model, case_id='failed', source_id='song')
    summary = summarize_results([row])
    assert summary['valid_output_means']['mae'] == dict(mean=None, n=0)
    assert summary['valid_rate']['value'] == 0
    assert summary['failure_adjusted_mae']['mean'] == 1
    empty = summarize_results([])
    assert empty['joint_success_rate']['value'] is None
    assert empty['source_macro']['joint_success_rate'] == dict(mean=None, n=0)
    for result in (summary, empty):
        for key in ('label_distance', 'rhythm_similarity', 'hihat_similarity'):
            assert result[key] == dict(mean=None, n=0)
        assert all(value == dict(coefficient=None, n=0, reason='insufficient_samples')
                   for value in result['attribute_spearman'].values())
    json.dumps(empty, allow_nan=False)


def test_missing_skeleton_evidence_is_not_success_even_at_zero_threshold(setup):
    _, _, model = setup
    original = make_bar([(0, 'hh_closed'), (12, 'hh_closed')])
    prepared = prepare_input(dict(schema_version=1, original=bar_to_payload(original), previous=None,
                                 target=dict(mode='absolute', values=model.difficulty(original).normalized)), model)
    row = evaluate_result(prepared, prepared.original, model, case_id='hh', source_id='song',
                          config=EvaluationConfig(skeleton_threshold=0))
    assert row['target_success'] and not row['joint_success']
    assert row['metrics']['skeleton_similarity'] is None
    assert not row['baseline_joint_success']
    assert summarize_results([row])['skeleton_component_means']['fill'] == dict(mean=None, n=0)


def test_target_can_be_met_while_unprotected_skeleton_changes_too_much(setup):
    _, _, model = setup
    original = make_bar([(p, 'hh_closed') for p in range(0, 96, 6)] +
                        [(p, 'kick') for p in (0, 12, 36, 48, 60)] + [(24, 'snare'), (72, 'snare')])
    after = original.with_notes(n for n in original.notes if not (n.instrument == 'kick' and n.position % 24))
    prepared = prepare_input(dict(schema_version=1, original=bar_to_payload(original), previous=None,
                                 target=dict(mode='absolute', values=model.difficulty(after).normalized)), model)
    after = prepared.original.with_notes(n for n in prepared.original.notes if not (n.instrument == 'kick' and n.position % 24))
    row = evaluate_result(prepared, after, model, case_id='kick', source_id='song', rule_count=1)
    assert row['status'] == 'valid' and row['target_success'] and not row['joint_success']
    assert row['metrics']['skeleton_similarity'] == pytest.approx((8/11+1)/2)


def test_no_requested_change_has_no_direction_denominator(setup):
    prepared, _, model = setup
    prepared = replace(prepared, target=prepared.original_difficulty.normalized)
    row = evaluate_result(prepared, prepared.original, model, case_id='same', source_id='song')
    assert row['joint_success'] and row['baseline_joint_success']
    assert row['metrics']['direction_rate'] is None
    assert row['metrics']['requested_progress_rate'] is None
    assert row['metrics']['held_attribute_mae'] == 0


def test_unrequested_attribute_changes_are_reported_as_drift(setup):
    prepared, candidate, model = setup
    prepared = replace(prepared, target=prepared.original_difficulty.normalized)
    row = evaluate_result(prepared, candidate, model, case_id='drift', source_id='song')
    assert row['metrics']['direction_rate'] is None
    assert row['metrics']['held_attribute_mae'] > 0
    assert row['metrics']['mae_improvement'] < 0
    assert not row['joint_success']


def test_overshoot_in_requested_direction_is_not_automatically_progress(setup):
    prepared, candidate, model = setup
    before = prepared.original_difficulty.normalized
    after = model.difficulty(candidate).normalized
    target = {k: before[k]+.1*(after[k]-before[k]) for k in DIFFICULTY_KEYS}
    row = evaluate_result(replace(prepared, target=target), candidate, model, case_id='over', source_id='song')
    assert row['metrics']['direction_rate'] == 1
    assert row['metrics']['requested_progress_rate'] == 0
    assert row['metrics']['mae_improvement'] < 0


def test_source_macro_does_not_overweight_longer_songs(setup):
    prepared, _, model = setup
    rows = [evaluate(setup, case_id=f'a{i}', source_id='a') for i in range(3)]
    rows.append(evaluate_result(prepared, None, model, case_id='b', source_id='b'))
    summary = summarize_results(rows)
    assert summary['joint_success_rate']['value'] == .75
    assert summary['source_macro']['joint_success_rate'] == dict(mean=.5, n=2)
    assert summary['source_macro']['failure_adjusted_mae'] == dict(mean=.5, n=2)


def test_different_thresholds_and_duplicate_cases_cannot_be_mixed(setup):
    a = evaluate(setup)
    with pytest.raises(ValueError, match='重複'):
        summarize_results([a, a])
    b = evaluate(setup, case_id='b', config=EvaluationConfig(target_tolerance=0))
    with pytest.raises(ValueError, match='閾値'):
        summarize_results([a, b])


def test_runtime_counts_only_measured_cases(setup):
    rows = [evaluate(setup, case_id=str(i), elapsed_seconds=t) for i, t in enumerate((None, .2, .4))]
    times = summarize_results(rows)['runtime_seconds']
    assert times['n'] == 2 and times['mean'] == pytest.approx(.3)
    assert times['median'] == pytest.approx(.3) and times['maximum'] == .4


@pytest.mark.parametrize('kwargs', [dict(elapsed_seconds=-1), dict(elapsed_seconds=float('nan')),
                                    dict(elapsed_seconds=True), dict(rule_count=-1), dict(rule_count=True),
                                    dict(failure_reason='timeout'), dict(source_id=''), dict(case_id='')])
def test_invalid_evaluation_arguments(setup, kwargs):
    with pytest.raises(ValueError):
        evaluate(setup, **kwargs)


@pytest.mark.parametrize('config', [EvaluationConfig(-1), EvaluationConfig(float('nan')),
                                    EvaluationConfig(True), EvaluationConfig(skeleton_threshold=1.1)])
def test_invalid_thresholds_rejected(setup, config):
    with pytest.raises(ValueError):
        evaluate(setup, config=config)


def test_onset_similarity_separates_hihat_and_keeps_instrument_identity():
    original = make_bar([(0, 'kick'), (24, 'snare'), (0, 'hh_closed'), (12, 'hh_open'), (24, 'hh_pedal')])
    candidate = make_bar([(0, 'snare'), (24, 'snare', 'Ghost'), (0, 'hh_open'), (12, 'hh_closed'), (48, 'hh_pedal')])
    scores = onset_similarities(original, candidate)
    assert scores['rhythm_similarity'] == .5
    assert scores['hihat_similarity'] == pytest.approx(2/3)
    assert onset_similarities(original, original) == dict(rhythm_similarity=1., hihat_similarity=1.)


def test_onset_similarity_handles_empty_groups_added_notes_and_duplicate_hihat_positions():
    empty = make_bar([])
    hihat = make_bar([(0, 'hh_closed'), (0, 'hh_pedal')])
    same_position = make_bar([(0, 'hh_open')])
    assert onset_similarities(hihat, same_position) == dict(rhythm_similarity=None, hihat_similarity=1.)
    assert onset_similarities(empty, empty) == dict(rhythm_similarity=None, hihat_similarity=None)
    assert onset_similarities(empty, hihat) == dict(rhythm_similarity=None, hihat_similarity=0.)
    assert onset_similarities(hihat, empty) == dict(rhythm_similarity=None, hihat_similarity=0.)
    assert onset_similarities(make_bar([(0, 'kick')]), make_bar([(0, 'kick'), (24, 'snare')])) == {
        'rhythm_similarity': pytest.approx(2/3), 'hihat_similarity': None}


def control_rows(targets, generated):
    return [dict(status='valid', target={key: target/7 for key in DIFFICULTY_KEYS},
                 metrics=dict(difficulty=dict(classes={key: output for key in DIFFICULTY_KEYS})))
            for target, output in zip(targets, generated)]


@pytest.mark.parametrize('targets,generated,expected', [
    ([1, 3, 5], [2, 4, 6], 1.),
    ([1, 3, 5], [6, 4, 2], -1.),
    ([1, 1, 3, 5], [2, 4, 4, 6], 5/6),
])
def test_attribute_spearman_uses_paired_samples_and_average_ranks_for_ties(targets, generated, expected):
    result = summarize_control_metrics(control_rows(targets, generated))
    for value in result['attribute_spearman'].values():
        assert value['coefficient'] == pytest.approx(expected)
        assert value['n'] == len(targets) and value['reason'] is None
    assert result['label_distance']['mean'] == pytest.approx(
        sum(abs(a-b) for a, b in zip(targets, generated))/len(targets))
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('targets,generated,reason', [
    ([1], [2], 'insufficient_samples'),
    ([1, 1], [2, 3], 'constant_target'),
    ([1, 2], [3, 3], 'constant_generated'),
])
def test_undefined_spearman_is_null_with_reason(targets, generated, reason):
    result = summarize_control_metrics(control_rows(targets, generated))
    assert all(value == dict(coefficient=None, n=len(targets), reason=reason)
               for value in result['attribute_spearman'].values())
    json.dumps(result, allow_nan=False)


def test_label_distance_averages_all_attributes_without_rounding_continuous_targets():
    rows = control_rows([1.5, 3.5], [2, 4])
    first = DIFFICULTY_KEYS[0]
    for row in rows:
        row['target'][first] = row['metrics']['difficulty']['classes'][first]/7
    result = summarize_control_metrics(rows)
    assert result['attribute_label_distance'][first] == dict(mean=0., n=2)
    for key in DIFFICULTY_KEYS[1:]:
        assert result['attribute_label_distance'][key] == dict(mean=.5, n=2)
    assert result['label_distance'] == dict(mean=.4, n=2)
