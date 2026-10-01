"""Beamの非貪欲な経路、制約、重複・予算、val選定の優先順位を検証する。"""
from dataclasses import replace
from itertools import product

import pytest

from scripts.evaluate.compare_arrangement_models import load_candidates, selection_key
from src.beamsearch.beam_search import Action, SearchConfig, beam_search, standard_actions
from src.common.examples import make_bar
from src.common.inputs import bar_to_payload, prepare_input
from src.beamsearch.rules import REGISTRY, apply_rule
from src.beamsearch.scoring import ScoreConfig, ScoringModel
from src.common.representation import to_tokens
from src.data.drum_difficulty_v2 import ATTRS


@pytest.fixture
def setup():
    original = make_bar([(p, 'hh_closed') for p in range(0, 96, 6)] +
                        [(p, 'kick') for p in (0, 24, 48, 72)] +
                        [(24, 'snare'), (72, 'snare', 'Accent'), (18, 'snare', 'Ghost'), (66, 'snare', 'Ghost')])
    first = apply_rule(original, 'hh_16_to_8').after
    after = apply_rule(first, 'remove_ghost_notes', original=original).after
    vocab = set(to_tokens(original)) | set(to_tokens(first)) | set(to_tokens(after))
    vocab.update(f'<POS_{i}>' for i in range(24))
    model = ScoringModel({a: [.5, 1, 2, 4, 8, 16] for a in ATTRS}, vocab)
    prepared = prepare_input(dict(schema_version=1, original=bar_to_payload(original), previous=None,
                                 target=dict(mode='absolute', values=model.difficulty(after).normalized)), model)
    return prepared, model


def test_action_catalog_covers_all_rules_and_is_unique():
    actions = standard_actions()
    assert {action.name for action in actions} == set(REGISTRY)
    assert len(actions) == len({action.key for action in actions}) == 79
    assert dict(next(a.parameters for a in actions if a.name == 'add_roll')) == dict(start=72, end=96)


def test_small_search_matches_exhaustive_search(setup):
    prepared, model = setup
    actions = (Action('hh_16_to_8'), Action('remove_ghost_notes'))
    scores = [model.score(prepared.original, prepared.original, prepared.target).total]
    for length in (1, 2, 3):
        for path in product(actions, repeat=length):
            bar, count = prepared.original, 0
            for action in path:
                result = apply_rule(bar, action.name, original=prepared.original, guards=prepared.guards)
                if result.status == 'applied':
                    count += 1
                bar = result.after
            score = model.score(prepared.original, bar, prepared.target, rule_count=count)
            if score.valid:
                scores.append(score.total)
    result = beam_search(prepared, model, SearchConfig(8, 3), actions)
    assert result.score.total == pytest.approx(max(scores))
    assert result.statistics['counters'].get('layer_duplicate', 0) > 0


def test_intermediate_score_drop_is_not_pruned(setup, monkeypatch):
    prepared, base_model = setup
    def conditional_ghost(bar, ctx):
        return REGISTRY['remove_ghost_notes'](bar, ctx=ctx) if sum(n.instrument == 'hh_closed' for n in bar.notes) == 8 else bar
    monkeypatch.setitem(REGISTRY, 'conditional_ghost', conditional_ghost)
    first = apply_rule(prepared.original, 'hh_16_to_8').after
    root_score = base_model.score(prepared.original, prepared.original, prepared.target)
    first_score = base_model.score(prepared.original, first, prepared.target, rule_count=1)
    # 初手の改善を編集ペナルティがわずかに上回る設定を作る。
    coefficient = (root_score.difficulty_loss-first_score.difficulty_loss+.001)/first_score.edit_distance
    model = ScoringModel(base_model.boundaries, base_model.vocabulary,
                         ScoreConfig(lambda_rules=0, lambda_edit=coefficient))
    root = model.score(prepared.original, prepared.original, prepared.target)
    assert model.score(prepared.original, first, prepared.target, rule_count=1).total < root.total
    result = beam_search(prepared, model, SearchConfig(1, 2), (Action('hh_16_to_8'), Action('conditional_ghost')))
    assert len(result.history) == 2 and result.score.total > root.total


def test_no_edit_and_depth_zero_are_valid_fallbacks(setup):
    prepared, model = setup
    for config, actions in ((SearchConfig(1, 0), standard_actions()), (SearchConfig(1, 3), ())):
        result = beam_search(prepared, model, config, actions)
        assert not result.history and result.score.valid and result.score.edit_distance == 0
        assert result.statistics['counters'].get('rule_applications', 0) == 0


def test_increase_target_can_select_an_addition_rule(setup):
    _, base = setup
    original = make_bar([(p, 'hh_closed') for p in (0, 12, 24, 36)] +
                        [(0, 'kick'), (48, 'kick'), (24, 'snare'), (72, 'snare')])
    after = apply_rule(original, 'hh_8_to_16').after
    model = ScoringModel(base.boundaries, base.vocabulary, ScoreConfig(lambda_difficulty=10))
    target = model.difficulty(after).normalized
    assert target['s_hand'] > model.difficulty(original).normalized['s_hand']
    prepared = prepare_input(dict(schema_version=1, original=bar_to_payload(original), previous=None,
                                 target=dict(mode='absolute', values=target)), model)
    result = beam_search(prepared, model, SearchConfig(2, 1), (Action('hh_8_to_16'),))
    assert result.history and result.history[0].name == 'hh_8_to_16'
    assert len(result.candidate.notes) > len(prepared.original.notes)
    assert result.score.difficulty_loss == 0


def test_rule_application_budget_is_deterministic_and_returns_best_so_far(setup):
    prepared, model = setup
    result = beam_search(prepared, model, SearchConfig(8, 3, 5))
    assert result.statistics['counters']['rule_applications'] == 5
    assert result.statistics['stopped'] == 'rule_application_budget'
    assert result.score.valid and result.score.total >= result.statistics['baseline_score']


def test_reordered_actions_produce_same_result_and_statistics(setup):
    prepared, model = setup
    actions = (Action('hh_16_to_8'), Action('remove_ghost_notes'), Action('reduce_ghost_note_ratio'))
    a = beam_search(prepared, model, SearchConfig(4, 3), actions)
    b = beam_search(prepared, model, SearchConfig(4, 3), reversed(actions))
    assert (a.candidate, a.history, a.score, a.statistics) == (b.candidate, b.history, b.score, b.statistics)


def test_selected_history_replays_and_preserves_original_guards(setup):
    prepared, model = setup
    result = beam_search(prepared, model, SearchConfig(4, 2))
    bar = prepared.original.with_notes(prepared.original.notes)
    for action in result.history:
        step = apply_rule(bar, action.name, original=prepared.original, guards=prepared.guards, **dict(action.parameters))
        assert step.status == 'applied'
        bar = step.after.with_notes(step.after.notes)
    assert bar == result.candidate
    assert model.score(prepared.original, bar, prepared.target, rule_count=len(result.history), guards=prepared.guards).valid
    assert len(result.history) <= 2


def test_vocabulary_invalid_additions_never_returned(setup):
    prepared, model = setup
    # FLAM語彙はこの試験語彙にない。
    result = beam_search(prepared, model, SearchConfig(2, 1), (Action('add_flams'),))
    assert not result.history
    assert result.statistics['counters'].get('score_invalid', 0) == 1


def test_annotation_only_state_changes_are_not_collapsed(setup, monkeypatch):
    prepared, model = setup
    def annotate(bar, ctx):
        return bar.with_notes(replace(n, ornament_group='g') if n.id not in ctx.protected else n for n in bar.notes)
    monkeypatch.setitem(REGISTRY, 'annotate', annotate)
    result = beam_search(prepared, model, SearchConfig(2, 2), (Action('annotate'),))
    assert result.statistics['layers'][0]['unique_valid_candidates'] == 1
    assert not result.history  # 同じ譜面なら無編集の方が高得点


@pytest.mark.parametrize('config', [SearchConfig(0, 1), SearchConfig(True, 1), SearchConfig(1, -1), SearchConfig(1, 1, 0)])
def test_invalid_search_configuration(setup, config):
    with pytest.raises(ValueError):
        beam_search(*setup, config=config)


@pytest.mark.parametrize('actions', [(Action('unknown'),), (Action('hh_16_to_8', (('unknown', 1),)),),
                                    (Action('hh_16_to_8', (('ctx', None),)),), (Action('hh_16_to_8'), Action('hh_16_to_8'))])
def test_invalid_actions(setup, actions):
    with pytest.raises(ValueError):
        beam_search(*setup, actions=actions)


def test_candidate_configuration_and_selection_priority():
    config = load_candidates('config/beamsearch_candidates.json')
    assert len(config['candidates']) == 6
    def summary(success, mae, edit):
        return dict(source_macro=dict(joint_success_rate=dict(mean=success), failure_adjusted_mae=dict(mean=mae)),
                    valid_output_means=dict(edit_distance=dict(mean=edit)), rule_count_valid=dict(mean=1),
                    runtime_seconds=dict(mean=1))
    assert selection_key('a', summary(.2, .5, .9)) < selection_key('b', summary(.1, 0, 0))
    assert selection_key('a', summary(.2, .1, .9)) < selection_key('b', summary(.2, .2, 0))
    assert selection_key('a', summary(.2, .1, .1)) < selection_key('b', summary(.2, .1, .2))
