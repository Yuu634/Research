"""承認済みv2式。既存のv1ラベル・モデルの意味は変更しない。"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from src.data.drum_difficulty_calculator import (
    DISTANCE_MATRIX, FOOT_INSTRUMENTS, HAND_INSTRUMENTS,
    TOKEN_TO_DISTANCE_KEY, _extract_instrument,
)

VERSION = 'drum_difficulty_v2_20260930'
ATTRS = ('s_tech', 's_indep', 's_hand', 's_foot', 's_move', 's_rhythm')
AUX = ('duration', 'special_count', 'dynamic_change', 'async_count',
       'hand_density', 'hand_q80', 'hand_q90', 'hand_max',
       'kick_density', 'kick_q80', 'kick_q90', 'kick_max',
       'pedal_density', 'pedal_q80', 'pedal_q90', 'pedal_max')


@dataclass(frozen=True, order=True)
class Hit:
    position: int
    instrument: str
    dynamic: float = .5
    special: bool = False


@dataclass(frozen=True)
class Bar:
    bpm: float
    hits: tuple[Hit, ...]

    @property
    def duration(self):
        return 240 / self.bpm


@lru_cache(maxsize=1024)
def note_spec(token):
    instrument = _extract_instrument(token)
    if instrument not in HAND_INSTRUMENTS | FOOT_INSTRUMENTS:
        raise ValueError(f'未対応の打撃トークン: {token}')
    dynamic = {'Ghost': 0., 'Normal': .5, 'Accent': 1.}.get(token.split('_')[-1], .5)
    special = instrument in ('snare_rim', 'ride_bell') or any(
        word in token.lower() for word in ('ghost', 'flam', 'roll', 'choke', 'xstick'))
    return instrument, dynamic, special


def decode_bars(bar_positions, token_indices, idx2token):
    """終端番兵の有無を受け付け、最後の実小節を落とさない。"""
    tokens = [idx2token[int(i)] for i in token_indices]
    if not tokens or tokens[-1] != '<EOS>' or '<EOS>' in tokens[:-1]:
        raise ValueError('末尾に一つのEOSが必要')
    starts = list(map(int, bar_positions))
    if starts and starts[-1] == len(tokens):
        starts.pop()
    if not starts or starts[0] != 0 or starts != sorted(set(starts)) or starts[-1] >= len(tokens)-1:
        raise ValueError('小節開始位置が不正')
    bars = []
    for lo, hi in zip(starts, starts[1:] + [len(tokens)-1]):
        bpm = None
        beat = position = None
        hits = []
        saw_bar = False
        for token in tokens[lo:hi]:
            if token.startswith('<TEMPO_'):
                if saw_bar or bpm is not None:
                    raise ValueError('収集形式以外の小節内テンポ指定')
                bpm = float(token[7:-1])
                if not math.isfinite(bpm) or bpm <= 0:
                    raise ValueError('テンポが不正')
            elif token == '<BAR>':
                if saw_bar:
                    raise ValueError('小節開始情報とBAR数が不一致')
                saw_bar = True
            elif token.startswith('<BEAT_'):
                beat = int(token[6:-1])-1
                position = None
                if not 0 <= beat < 4:
                    raise ValueError('4/4以外の拍番号')
            elif token.startswith('<POS_'):
                position = int(token[5:-1])
                if beat is None or not 0 <= position < 24:
                    raise ValueError('位置指定が不正')
            else:
                if not saw_bar or beat is None or position is None:
                    raise ValueError('打撃の前にBAR/BEAT/POSが必要')
                hits.append(Hit(beat*24+position, *note_spec(token)))
        if bpm is None or not saw_bar:
            raise ValueError('各小節にTEMPOとBARが必要')
        bars.append(Bar(bpm, tuple(sorted(hits))))
    return bars


def dynamic_group(instrument):
    return {'snare_rim': 'snare', 'ride_bell': 'ride_cymbal',
            'hh_open': 'hh_closed'}.get(instrument, instrument)


def timed_hits(bar, offset=0):
    return [(offset + h.position * bar.duration / 96, h) for h in bar.hits]


def rate(current, previous, duration):
    groups = Counter(t for t, _ in current)
    last = max((t for t, _ in previous), default=None)
    speeds = []
    for t, count in sorted(groups.items()):
        if last is not None:
            speeds.append(count/(t-last))
        last = t
    density = len(current)/duration
    q80, q90, peak = np.quantile(speeds, [.8, .9, 1.], method='linear') if speeds else (0., 0., 0.)
    return max(density, q90), (density, float(q80), float(q90), float(peak))


def movement(current, previous):
    """2手の交換対称性を使う厳密な最小最大DP。対象小節の移動だけにコスト。"""
    groups = defaultdict(list)
    for is_current, hits in ((False, previous), (True, current)):
        for t, h in hits:
            if h.instrument in HAND_INSTRUMENTS:
                groups[(t, is_current)].append(TOKEN_TO_DISTANCE_KEY[h.instrument])
    if any(len(g) > 2 for g in groups.values()):
        return math.nan
    # 状態は各手が最後に叩いたイベントID。-1はまだ位置未定。
    states = {(-1, -1): 0.}
    events = []
    for (t, is_current), instruments in sorted(groups.items()):
        ids = []
        for inst in sorted(instruments):
            ids.append(len(events))
            events.append((t, inst))

        def cost(old, new):
            if old == -1 or not is_current:
                return 0.
            pt, pi = events[old]
            nt, ni = events[new]
            return DISTANCE_MATRIX[pi][ni]/(nt-pt)

        next_states = {}
        for (a, b), value in states.items():
            if len(ids) == 1:
                c = ids[0]
                options = [((b, c), max(value, cost(a, c))),
                           ((a, c), max(value, cost(b, c)))]
            else:
                c, d = ids
                options = [((c, d), max(value, cost(a, c), cost(b, d))),
                           ((c, d), max(value, cost(a, d), cost(b, c)))]
            for state, candidate in options:
                state = tuple(sorted(state))
                next_states[state] = min(next_states.get(state, math.inf), candidate)
        states = next_states
    return min(states.values())


@lru_cache(maxsize=32768)
def lz78_phrases(sequence):
    dictionary = set()
    start = count = 0
    while start < len(sequence):
        end = start+1
        while end <= len(sequence) and sequence[start:end] in dictionary:
            end += 1
        count += 1
        if end > len(sequence):
            break
        dictionary.add(sequence[start:end])
        start = end
    return count


def rhythm(hits):
    sequences = [bytearray(96) for _ in range(3)]
    for h in hits:
        group = 0 if h.instrument == 'kick' else 1 if h.instrument in ('snare', 'snare_rim') else 2
        sequences[group][h.position] = 1
    base = lz78_phrases(bytes(96))
    return sum(max(0, lz78_phrases(bytes(s))-base) for s in sequences)*math.log2(96)/96/3


@lru_cache(maxsize=16384)
def score_bar(bar, previous=None):
    """(6生スコア, 監査用構成値, フラグ)を返す。キャッシュに可変値を置かない。"""
    current = timed_hits(bar)
    prior = timed_hits(previous, -previous.duration) if previous is not None else []
    duration = bar.duration
    h = [(t, n) for t, n in current if n.instrument in HAND_INSTRUMENTS]
    f = [(t, n) for t, n in current if n.instrument in FOOT_INSTRUMENTS]
    ht, ft = {t for t, _ in h}, {t for t, _ in f}
    async_count = len(ht ^ ft)
    indep = (async_count/duration*2*min(len(ht), len(ft))/(len(ht)+len(ft))) if ht and ft else 0.
    last_dynamic = {}
    for t, n in prior:
        key = dynamic_group(n.instrument)
        if key not in last_dynamic or t > last_dynamic[key][0]:
            last_dynamic[key] = (t, n.dynamic)
        elif t == last_dynamic[key][0]:
            last_dynamic[key] = (t, max(n.dynamic, last_dynamic[key][1]))
    dynamic_events = defaultdict(list)
    for t, n in current:
        dynamic_events[(t, dynamic_group(n.instrument))].append(n.dynamic)
    change = 0.
    for (t, key), values in sorted(dynamic_events.items()):
        value = max(values)
        if key in last_dynamic:
            change += abs(value-last_dynamic[key][1])
        last_dynamic[key] = (t, value)
    special = sum(n.special for _, n in current)
    hand_rate, hand_aux = rate(h, [(t,n) for t,n in prior if n.instrument in HAND_INSTRUMENTS], duration)
    rates, aux = [], []
    for instrument in ('kick', 'hh_pedal'):
        r, detail = rate([(t,n) for t,n in current if n.instrument == instrument],
                         [(t,n) for t,n in prior if n.instrument == instrument], duration)
        rates.append(r)
        aux.extend(detail)
    move = movement(current, prior) if current else 0.
    flags = []
    if not math.isfinite(move):
        flags.append('movement_infeasible')
    if len({(n.position, n.instrument) for n in bar.hits}) != len(bar.hits):
        flags.append('duplicate_position_instrument')
    scores = ((special+change)/duration, indep, hand_rate, max(rates), move, rhythm(bar.hits))
    detail = (duration, special, change, async_count, *hand_aux, *aux)
    return scores, detail, tuple(flags)


def score_piece(bars):
    values, components, flags = [], [], []
    previous = None
    for i, bar in enumerate(bars):
        scores, detail, issues = score_bar(bar, previous)
        values.append(scores)
        components.append(detail)
        flags.extend((i, issue) for issue in issues)
        previous = bar
    return np.array(values, dtype=np.float64), np.array(components, dtype=np.float64), flags


def weighted_boundaries(value_weights):
    """正値の重み付き経験CDFの逆関数。ゼロは専用クラスで扱う。"""
    if not value_weights:
        return []
    values = np.array(sorted(value_weights), dtype=float)
    weights = np.array([value_weights[v] for v in values])
    if np.any(values <= 0) or np.any(weights <= 0):
        raise ValueError('分位点には正値と正の重みが必要')
    cumulative = np.cumsum(weights)/weights.sum()
    return values[np.searchsorted(cumulative, np.arange(1,7)/7, side='left')].tolist()


def discretize(scores, boundaries):
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[1] != len(ATTRS) or not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError('ラベル化には有限・非負の6属性が必要')
    out = np.zeros(scores.shape, dtype=np.int8)
    for j, name in enumerate(ATTRS):
        positive = scores[:, j] > 0
        # trainで正値がない場合、評価側の正値は1に割り当て、分布外件数を監査する。
        out[positive, j] = 1 + np.searchsorted(boundaries[name], scores[positive, j], side='left')
    return out
