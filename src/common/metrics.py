"""探索と評価で共用する音符編集距離・骨格類似度。"""
from collections import Counter
import numpy as np
from src.common.representation import Bar

SKELETON_KEYS = ('kick', 'snare', 'accent', 'fill')
HIHAT_INSTRUMENTS = frozenset(('hh_closed', 'hh_open', 'hh_pedal'))


def onset_similarities(original: Bar, candidate: Bar):
    """原譜との発音F1。非HHは楽器と位置、HH3種は位置だけを比較する。

    96位置の完全一致で比較し、強弱・奏法・音符IDは使用しない。
    両方空ならNone、片方だけ空なら0。同一イベントの重複は数えない。
    """
    def events(bar):
        rhythm = {(n.instrument, n.position) for n in bar.notes if n.instrument not in HIHAT_INSTRUMENTS}
        hihat = {n.position for n in bar.notes if n.instrument in HIHAT_INSTRUMENTS}
        return rhythm, hihat

    def f1(a, b):
        return 2 * len(a & b) / (len(a) + len(b)) if a or b else None

    original_rhythm, original_hihat = events(original)
    candidate_rhythm, candidate_hihat = events(candidate)
    return dict(rhythm_similarity=f1(original_rhythm, candidate_rhythm),
                hihat_similarity=f1(original_hihat, candidate_hihat))


def _musical_key(note):
    return note.position, note.instrument, note.level, note.technique


def musical_edit_distance(original: Bar, candidate: Bar):
    """音符単位の置換・追加・削除を各1とした最少編集数／原譜音符数。

    内容が完全一致する音符を相殺し、残りを可能な限り置換で対応させる。
    ID・格納順・注釈だけの違いは数えない。大きな移動も1置換として扱う。
    """
    a = Counter(_musical_key(n) for n in original.notes)
    b = Counter(_musical_key(n) for n in candidate.notes)
    return max(sum((a-b).values()), sum((b-a).values())) / max(1, len(original.notes))


def _skeleton(bar, fill_regions):
    arrays = {name: np.zeros(96, dtype=float) for name in SKELETON_KEYS[:3]}
    for note in bar.notes:
        p = note.position
        if note.instrument == 'kick':
            arrays['kick'][p] = 2 if p % 24 == 0 else 1
        if note.instrument in ('snare', 'snare_rim'):
            arrays['snare'][p] = 3 if p in (24, 72) else 1
        if note.level == 'Accent':
            arrays['accent'][p] = 1
    # Fillの開始／終了を別々に比較。候補自身の注釈で評価区間を変えない。
    fill = np.zeros((len(fill_regions), 2, 96), dtype=float)
    for i, (start, end) in enumerate(fill_regions):
        positions = [n.position for n in bar.notes if start <= n.position < end]
        if positions:
            fill[i, 0, min(positions)] = 1
            fill[i, 1, max(positions)] = 1
    arrays['fill'] = fill.reshape(-1)
    return arrays


def skeleton_similarity(original: Bar, candidate: Bar, weights=(.45, .45, .1, .1)):
    """両方空の成分・不明なFillを除外した重み付きDice。証拠が全くなければ加点0。"""
    a = _skeleton(original, original.fill_regions)
    b = _skeleton(candidate, original.fill_regions)
    components = {}; active_weights = {}
    for name, weight in zip(SKELETON_KEYS, weights):
        denominator = float(a[name].sum() + b[name].sum())
        components[name] = float(2 * np.minimum(a[name], b[name]).sum() / denominator) if denominator else None
        active_weights[name] = float(weight) if denominator else 0.
    normalizer = sum(active_weights.values())
    effective = {name: w / normalizer if normalizer else 0. for name, w in active_weights.items()}
    value = sum((components[name] or 0.) * effective[name] for name in SKELETON_KEYS)
    return float(value), components, effective
