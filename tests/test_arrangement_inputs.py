"""入力の曖昧さ・文脈依存・教師情報の混入を防ぐ契約の検証。"""
from copy import deepcopy
import json

import pytest

from scripts.evaluate.review_arrangement_inputs import read_request
from src.common.examples import make_bar
from src.common.inputs import bar_to_payload, parse_bar, prepare_input, request_from_pair
from src.common.representation import to_tokens
from src.beamsearch.scoring import DIFFICULTY_KEYS, ScoringModel
from src.data.drum_difficulty_v2 import ATTRS


@pytest.fixture
def setup():
    bar = make_bar([(p, 'hh_closed') for p in range(0, 96, 6)] +
                   [(0, 'kick'), (24, 'snare'), (48, 'kick'), (72, 'snare')])
    vocabulary = set(to_tokens(bar)) | {'SNARE_HIT_Ghost', 'SNARE_HIT_Accent', '<TEMPO_240>'}
    vocabulary.update(f'<POS_{i}>' for i in range(24))
    model = ScoringModel({a: [.5, 1, 2, 4, 8, 16] for a in ATTRS}, vocabulary)
    payload = dict(schema_version=1, original=bar_to_payload(bar), previous=None,
                   target=dict(mode='absolute', values=model.difficulty(bar).normalized))
    return payload, model


def test_absolute_and_delta_resolve_to_same_target_and_score(setup):
    payload, model = setup
    payload['target']['values']['s_hand'] -= 1/7
    absolute = prepare_input(payload, model)
    other = deepcopy(payload)
    other['target'] = dict(mode='delta', values={
        k: absolute.target[k]-absolute.original_difficulty.normalized[k] for k in DIFFICULTY_KEYS})
    delta = prepare_input(other, model)
    assert delta.target == pytest.approx(absolute.target)
    assert delta.report(model)['baseline']['total'] == pytest.approx(absolute.report(model)['baseline']['total'])


def test_previous_required_even_when_absent(setup):
    payload, model = setup
    del payload['previous']
    with pytest.raises(ValueError, match='必須キー不足'):
        prepare_input(payload, model)


def test_delta_is_based_on_previous_context_and_following_does_not_change_score(setup):
    payload, model = setup
    payload['original'] = bar_to_payload(make_bar([(0, 'snare')]))
    payload['target'] = dict(mode='delta', values={k: 0 for k in DIFFICULTY_KEYS})
    isolated = prepare_input(payload, model)
    payload['previous'] = bar_to_payload(make_bar([(95, 'snare')]))
    contextual = prepare_input(payload, model)
    assert contextual.original_difficulty.raw['s_hand'] == pytest.approx(48)
    assert contextual.target['s_hand'] > isolated.target['s_hand']
    payload['following'] = bar_to_payload(make_bar([(0, 'snare')]))
    payload['context'] = dict(section='Chorus', phrase='Middle')
    following = prepare_input(payload, model)
    assert following.target == contextual.target
    assert following.report(model)['baseline'] == contextual.report(model)['baseline']
    assert following.context == payload['context']


@pytest.mark.parametrize('mode,value', [('absolute', -.1), ('absolute', 1.1), ('delta', -2), ('delta', 2)])
def test_target_out_of_bounds_is_not_clipped(setup, mode, value):
    payload, model = setup
    payload['target']['mode'] = mode
    payload['target']['values']['s_hand'] = value
    with pytest.raises(ValueError, match='クリップ'):
        prepare_input(payload, model)


@pytest.mark.parametrize('value', [True, '0.3', None, float('nan'), float('inf')])
def test_target_numeric_type_strict(setup, value):
    payload, model = setup
    payload['target']['values']['s_hand'] = value
    with pytest.raises(ValueError, match='有限の数値'):
        prepare_input(payload, model)


def test_target_requires_mode_and_named_five_attributes(setup):
    payload, model = setup
    for target in ({'values': payload['target']['values']},
                   dict(mode='automatic', values=payload['target']['values']),
                   dict(mode='absolute', values=[.5]*5),
                   dict(mode='absolute', values=dict(payload['target']['values'], s_rhythm=0))):
        with pytest.raises(ValueError):
            prepare_input(dict(payload, target=target), model)


def test_annotations_and_extra_protection_preserved(setup):
    payload, model = setup
    payload['original']['fill_regions'] = [[72, 96]]
    payload['original']['note_annotations'] = {'n1': {'ornament_group': 'drag1', 'ornament_role': 'grace'}}
    baseline = prepare_input(payload, model)
    unlocked = next(n.id for n in baseline.original.notes if n.id not in baseline.guards.locked)
    payload['extra_locked_ids'] = [unlocked]
    value = prepare_input(payload, model)
    assert baseline.guards.locked < value.guards.locked
    assert value.original.notes[1].ornament_group == 'drag1'
    assert value.original.fill_regions == ((72, 96),)
    restored = parse_bar(bar_to_payload(value.original), model.vocabulary)
    assert [(n.position, n.ornament_group, n.ornament_role) for n in restored.notes] == [
        (n.position, n.ornament_group, n.ornament_role) for n in value.original.notes]


@pytest.mark.parametrize('extra', [['not_present'], ['n0', 'n0'], [True], 'n0'])
def test_invalid_extra_lock_ids_rejected(setup, extra):
    payload, model = setup
    with pytest.raises(ValueError):
        prepare_input(dict(payload, extra_locked_ids=extra), model)


def test_unknown_keys_not_silently_ignored(setup):
    payload, model = setup
    with pytest.raises(ValueError, match='未対応キー'):
        prepare_input(dict(payload, X_after=[]), model)
    payload['original']['bpm'] = 240
    with pytest.raises(ValueError, match='未対応キー'):
        prepare_input(payload, model)


@pytest.mark.parametrize('change', [dict(meter=[3, 4]), dict(length=48), dict(tokens=[1, 2, 3]),
                                  dict(fill_regions=[[48, 96], [72, 96]]),
                                  dict(note_annotations={'n1000': {}})])
def test_unrepresentable_bar_contract_rejected(setup, change):
    payload, model = setup
    payload['original'].update(change)
    with pytest.raises(ValueError):
        prepare_input(payload, model)


def test_duplicate_and_three_hand_hits_rejected(setup):
    payload, model = setup
    model.vocabulary |= {'HH_OPEN_HIT_Normal'}
    for tokens in (["<TEMPO_120>", "<BAR>", "<BEAT_1>", "<POS_0>",
                    "SNARE_HIT_Normal", "SNARE_HIT_Normal", "<EOS>"],
                   ["<TEMPO_120>", "<BAR>", "<BEAT_1>", "<POS_0>",
                    "SNARE_HIT_Normal", "HH_CLOSED_HIT_Normal", "HH_OPEN_HIT_Normal", "<EOS>"]):
        payload['original']['tokens'] = tokens
        with pytest.raises(ValueError, match='制約違反'):
            prepare_input(payload, model)


def test_non_monotonic_and_empty_position_blocks_rejected(setup):
    payload, model = setup
    for middle in (['<BEAT_2>', '<POS_0>', 'SNARE_HIT_Normal', '<BEAT_1>', '<POS_0>', 'SNARE_HIT_Normal'],
                   ['<BEAT_1>', '<POS_12>', 'SNARE_HIT_Normal', '<POS_0>', 'SNARE_HIT_Normal'],
                   ['<BEAT_1>', '<POS_0>']):
        payload['original']['tokens'] = ['<TEMPO_120>', '<BAR>']+middle+['<EOS>']
        with pytest.raises(ValueError):
            prepare_input(payload, model)


def test_empty_bar_can_be_explicitly_supplied(setup):
    payload, model = setup
    payload['original'] = bar_to_payload(make_bar([]))
    payload['target'] = dict(mode='delta', values={k: 0 for k in DIFFICULTY_KEYS})
    result = prepare_input(payload, model)
    assert result.original.notes == ()
    assert result.report(model)['baseline']['skeleton_similarity'] == 0


def test_input_and_context_are_not_mutated_and_tempo_is_local(setup):
    payload, model = setup
    payload['previous'] = bar_to_payload(make_bar([(95, 'snare')], bpm=240))
    before = deepcopy(payload)
    value = prepare_input(payload, model)
    assert payload == before
    assert value.original.bpm == 120 and value.previous.bpm == 240
    assert value.report(model)['baseline']['valid']


def test_pair_adapter_uses_original_context_without_teacher_score(setup):
    payload, model = setup
    tokens = payload['original']['tokens']
    pair = dict(X_before=list(range(len(tokens))), d_after=[.5]*5,
                context=dict(previous=None, following=None, section=None,
                             after_difficulty_context='before_previous'))
    # X_afterも候補の元の前小節もなくても変換可能。
    request = request_from_pair(pair, dict(enumerate(tokens)))
    value = prepare_input(request, model)
    assert value.previous is None
    assert value.target == {k: .5 for k in DIFFICULTY_KEYS}
    assert 'X_after' not in request
    pair['context']['after_difficulty_context'] = 'candidate_previous'
    with pytest.raises(ValueError):
        request_from_pair(pair, dict(enumerate(tokens)))


def test_json_rejects_duplicate_keys_and_nonfinite_numbers(tmp_path):
    path = tmp_path/'input.json'
    for text in ('{"target":{},"target":{}}', '{"x":NaN}', '{"x":Infinity}'):
        path.write_text(text)
        with pytest.raises(ValueError):
            read_request(path)
    path.write_text(json.dumps({'schema_version': 1}))
    assert read_request(path) == {'schema_version': 1}
