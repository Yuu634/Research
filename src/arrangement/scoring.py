"""ステップ2：難易度・骨格・ルール数・編集量を分離した探索スコア。"""
from collections import Counter
from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from src.arrangement.representation import Bar, to_tokens
from src.arrangement.safety import Guards, default_guards, violations
from src.data.drum_difficulty_v2 import (
    ATTRS, Bar as DifficultyBar, Hit, discretize, note_spec, score_bar,
)

DIFFICULTY_KEYS = ATTRS[:5]
SKELETON_KEYS = ('kick', 'snare', 'accent', 'fill')


@dataclass(frozen=True)
class ScoreConfig:
    lambda_difficulty: float = 1.
    lambda_skeleton: float = .5
    lambda_rules: float = .02
    lambda_edit: float = .1
    difficulty_weights: tuple[float, ...] = (1., 1., 1., 1., 1.)
    skeleton_weights: tuple[float, ...] = (.45, .45, .1, .1)

    def validate(self):
        for value in (self.lambda_difficulty, self.lambda_skeleton, self.lambda_rules, self.lambda_edit):
            if not math.isfinite(value) or value < 0:
                raise ValueError('目的関数の係数は有限・非負')
        for weights, size in ((self.difficulty_weights, 5), (self.skeleton_weights, 4)):
            if len(weights) != size or not all(math.isfinite(v) and v >= 0 for v in weights) or sum(weights) <= 0:
                raise ValueError('成分重みの長さ・非負性・正の合計が必要')
        return self


@dataclass(frozen=True)
class Difficulty:
    raw: dict[str, float]
    classes: dict[str, int]
    normalized: dict[str, float]
    rhythm_reference: dict[str, float | int]


@dataclass(frozen=True)
class ScoreBreakdown:
    valid: bool
    total: float | None
    reasons: tuple[str, ...]
    difficulty: Difficulty | None = None
    target: dict[str, float] | None = None
    attribute_errors: dict[str, float] | None = None
    difficulty_loss: float | None = None
    skeleton_similarity: float | None = None
    skeleton_components: dict[str, float | None] | None = None
    skeleton_effective_weights: dict[str, float] | None = None
    rule_count: int | None = None
    edit_distance: float | None = None
    contributions: dict[str, float] | None = None

    @property
    def ranking_score(self):
        """JSONにはNoneを保存し、探索時だけ不適格候補を負の無限大へ写す。"""
        return self.total if self.valid else -math.inf


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


class ScoringModel:
    """探索本体に依存しない採点器。境界・語彙はv2成果物から明示的に渡す。"""

    def __init__(self, boundaries, vocabulary, config=ScoreConfig()):
        self.config = config.validate()
        if set(boundaries) != set(ATTRS):
            raise ValueError('v2の6属性の境界が必要。6番目は参考値としてのみ保存')
        self.boundaries = {}
        for name, values in boundaries.items():
            values = [float(v) for v in values]
            if len(values) != 6 or any(not math.isfinite(v) or v <= 0 for v in values) or values != sorted(values):
                raise ValueError('境界は正・有限・非減少の6値')
            self.boundaries[name] = values
        self.vocabulary = frozenset(vocabulary)
        if '<EOS>' not in self.vocabulary:
            raise ValueError('standardの語彙集合を指定してください')

    def _convert(self, bar):
        bar.validate()
        to_tokens(bar, self.vocabulary)  # 語彙で表現できない編集を黙って採点しない。
        return DifficultyBar(bar.bpm, tuple(sorted(Hit(n.position, *note_spec(n.token())) for n in bar.notes)))

    def difficulty(self, bar, previous=None):
        raw, _, flags = score_bar(self._convert(bar), self._convert(previous) if previous is not None else None)
        if flags or not np.isfinite(raw).all():
            raise ValueError('難易度算出不可: ' + ','.join(flags))
        classes = discretize(np.asarray([raw]), self.boundaries)[0]
        return Difficulty(
            raw={name: float(raw[i]) for i, name in enumerate(DIFFICULTY_KEYS)},
            classes={name: int(classes[i]) for i, name in enumerate(DIFFICULTY_KEYS)},
            normalized={name: float(classes[i])/7 for i, name in enumerate(DIFFICULTY_KEYS)},
            rhythm_reference={'raw': float(raw[5]), 'class': int(classes[5])},
        )

    @staticmethod
    def _target(target):
        if not isinstance(target, Mapping) or set(target) != set(DIFFICULTY_KEYS):
            raise ValueError('目標はv2の5属性名をキーとする辞書で指定してください')
        result = {name: float(target[name]) for name in DIFFICULTY_KEYS}
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in result.values()):
            raise ValueError('採点器への目標は0〜1の絶対値。負値や差分を自動解釈しない')
        return result

    def score(self, original, candidate, target, *, rule_count=0, previous=None, guards=None):
        target = self._target(target)
        if type(rule_count) is not int or rule_count < 0:
            raise ValueError('rule_countは実際に適用したルール数の非負整数')
        original.validate()
        self.difficulty(original, previous)  # 不正な入力は候補不採用と区別し、例外にする。
        guards = default_guards(original) if guards is None else guards
        try:
            errors = violations(original, candidate, guards)
            if candidate.fill_regions != original.fill_regions:
                errors += ('fill_annotations_changed',)
            if errors:
                return ScoreBreakdown(False, None, errors, target=target, rule_count=rule_count)
            difficulty = self.difficulty(candidate, previous)
        except ValueError as error:
            return ScoreBreakdown(False, None, (str(error),), target=target, rule_count=rule_count)
        errors = {name: abs(difficulty.normalized[name]-target[name]) for name in DIFFICULTY_KEYS}
        weights = self.config.difficulty_weights
        loss = sum(w*errors[name] for name, w in zip(DIFFICULTY_KEYS, weights)) / sum(weights)
        skeleton, components, effective = skeleton_similarity(original, candidate, self.config.skeleton_weights)
        edit = musical_edit_distance(original, candidate)
        contributions = {
            'difficulty': -self.config.lambda_difficulty * loss,
            'skeleton': self.config.lambda_skeleton * skeleton,
            'rules': -self.config.lambda_rules * rule_count,
            'edit': -self.config.lambda_edit * edit,
        }
        return ScoreBreakdown(True, float(sum(contributions.values())), (), difficulty, target, errors,
                              float(loss), skeleton, components, effective, rule_count, edit, contributions)
