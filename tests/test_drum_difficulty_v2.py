"""v2の承認例、不変性、厳密DP、末尾小節とクラス境界を検証する。"""
import itertools
import math

import numpy as np
import pytest

from src.data.drum_difficulty_v2 import (
    ATTRS, Bar, Hit, decode_bars, discretize, lz78_phrases, movement,
    score_bar, score_piece, timed_hits, weighted_boundaries,
)
from src.data.drum_difficulty_calculator import DISTANCE_MATRIX, TOKEN_TO_DISTANCE_KEY


def bar(hits, bpm=120):
    return Bar(bpm, tuple(sorted(hits)))


def hits(positions, instrument='snare', dynamic=.5, special=False):
    return [Hit(p, instrument, dynamic, special) for p in positions]


def test_approved_independence_examples():
    hand = hits([0,24,48,72])
    assert score_bar(bar(hand))[0][1] == 0
    assert score_bar(bar(hand+hits([0,24,48,72], 'kick')))[0][1] == 0
    assert score_bar(bar(hand+hits([12,36,60,84], 'kick')))[0][1] == 4


def test_speed_detects_burst_and_separates_pedals():
    assert score_bar(bar(hits([0,24,48,72])))[0][2] == 2
    assert score_bar(bar(hits([0,6,12,18])))[0][2] == 8
    pedals = hits([0,24,48,72], 'kick')+hits([12,36,60,84], 'hh_pedal')
    assert score_bar(bar(pedals))[0][3] == 2


def test_tech_not_diluted_by_normal_kick_or_simultaneous_order():
    h = hits([0,24], dynamic=0, special=True)+hits([48,72], dynamic=1)
    a = score_bar(bar(h))[0]
    b = score_bar(bar(h+hits([0,12,24,36,48,60,72,84], 'kick')))[0]
    assert a[0] == b[0] == 1.5
    simultaneous = bar(h+hits([0,24,48,72], 'hh_closed'))
    np.testing.assert_equal(score_bar(simultaneous)[0], score_bar(Bar(120, tuple(reversed(simultaneous.hits))))[0])


def test_previous_bar_transition_and_empty_bar_reset():
    previous = bar(hits([95], dynamic=0, special=True))
    current = bar(hits([0], dynamic=1))
    assert score_bar(current, previous)[0][0] == .5
    assert score_bar(current, previous)[0][2] == pytest.approx(48)
    assert score_bar(current, bar([]))[0][0] == 0
    assert score_bar(current, bar([]))[0][2] == .5


def test_tempo_scaling_and_rhythm_invariance():
    h = hits([0,18,48], dynamic=0, special=True)+hits([6,42,78], 'floor')+hits([12,60], 'kick')
    low, high = score_bar(bar(h,120))[0], score_bar(bar(h,240))[0]
    np.testing.assert_allclose(np.array(high[:5]), 2*np.array(low[:5]))
    assert high[5] == low[5]


def brute_movement(current, previous):
    events = [(t, n, False) for t,n in previous]+[(t,n,True) for t,n in current]
    best = math.inf
    for assignment in itertools.product([0,1], repeat=len(events)):
        last = [None,None]
        value = 0
        for limb,(t,n,is_current) in zip(assignment, events):
            if last[limb] is not None:
                pt, pn = last[limb]
                if t == pt:
                    value = math.inf
                    break
                if is_current:
                    value = max(value, DISTANCE_MATRIX[TOKEN_TO_DISTANCE_KEY[pn.instrument]][
                        TOKEN_TO_DISTANCE_KEY[n.instrument]]/(t-pt))
            last[limb] = (t,n)
        best = min(best, value)
    return best


def test_movement_dp_matches_exhaustive_assignments():
    rng = np.random.default_rng(47)
    for _ in range(12):
        previous = bar([Hit(p, str(rng.choice(['snare','hh_closed','floor']))) for p in [0,48]])
        current = bar([Hit(p, str(rng.choice(['snare','hh_closed','floor']))) for p in [0,0,24,48,72]])
        prior = timed_hits(previous, -previous.duration)
        actual = movement(timed_hits(current), prior)
        assert actual == pytest.approx(brute_movement(timed_hits(current), prior))


def test_movement_fixed_hands_and_infeasible_are_distinct():
    fixed = bar(hits([0,24,48,72])+hits([0,24,48,72], 'hh_closed'))
    assert score_bar(fixed)[0][4] == 0
    invalid = bar(hits([0])+hits([0], 'hh_closed')+hits([0], 'floor'))
    scores, _, flags = score_bar(invalid)
    assert math.isnan(scores[4]) and 'movement_infeasible' in flags
    assert score_bar(bar([]))[0] == (0.,)*6


def test_lz_fixed_length_and_constant_baseline():
    assert lz78_phrases(bytes(96)) == 14
    assert lz78_phrases(bytes([1]*96)) == 14
    assert lz78_phrases(bytes([0,1,0,1])) == 3
    assert score_bar(bar(hits(range(96), 'kick')))[0][5] == 0
    a = bar(hits([0,6,24,42,48,78], 'kick'))
    b = bar(hits([0,6,24,42,48,78], 'kick', dynamic=1), 180)
    assert score_bar(a)[0][5] == score_bar(b)[0][5]


@pytest.mark.parametrize('sentinel', [False,True])
def test_decoder_retains_last_bar_and_tempo(sentinel):
    tokens = ['<TEMPO_120>','<BAR>','<BEAT_1>','<POS_0>','KICK_HIT_Normal',
              '<TEMPO_180>','<BAR>','<BEAT_4>','<POS_23>','SNARE_HIT_Ghost','<EOS>']
    starts = [0,5]+([len(tokens)] if sentinel else [])
    bars = decode_bars(starts, list(range(len(tokens))), dict(enumerate(tokens)))
    assert len(bars) == 2 and bars[-1].bpm == 180 and bars[-1].hits[0].position == 95
    assert score_piece(bars)[0].shape == (2,6)


def test_weighted_bounds_zero_and_ties():
    bounds = weighted_boundaries({1.:.9, 100.:.1})
    assert bounds == [1.]*6
    x = np.array([[0.]*6,[1.]*6,[100.]*6])
    y = discretize(x, {name:bounds for name in ATTRS})
    np.testing.assert_array_equal(y[:,0], [0,1,7])
    with pytest.raises(ValueError):
        discretize(np.full((1,6), np.nan), {name:bounds for name in ATTRS})


def test_dynamic_same_time_group_uses_max_once():
    prev = bar(hits([72], dynamic=0))
    current = bar(hits([0], dynamic=.5)+hits([0], 'snare_rim', dynamic=1, special=True))
    assert score_bar(current, prev)[0][0] == 1


@pytest.mark.parametrize('bad_token', ['UNKNOWN_HIT_Normal', '<POS_24>', '<EOS>'])
def test_decoder_rejects_invalid_input(bad_token):
    tokens = ['<TEMPO_120>', '<BAR>', '<BEAT_1>', '<POS_0>', bad_token, '<EOS>']
    with pytest.raises(ValueError):
        decode_bars([0], list(range(len(tokens))), dict(enumerate(tokens)))


def test_duplicate_hits_are_audited_without_silent_deduplication():
    scores, _, flags = score_bar(bar(hits([0, 0], 'kick')))
    assert 'duplicate_position_instrument' in flags
    assert scores[3] == 1  # 2打撃 / 2秒。0.5へ暗黙に変更しない。


def test_cross_bar_interval_uses_both_tempos():
    previous = bar(hits([72]), bpm=60)
    current = bar(hits([24]), bpm=240)
    # 前小節末尾まで1秒、現在小節の打撃まで0.25秒。
    _, components, _ = score_bar(current, previous)
    assert components[6] == pytest.approx(1 / 1.25)
