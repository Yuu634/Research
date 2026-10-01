"""3方式共通のv2難易度計算と絶対目標の検証。"""
from dataclasses import dataclass
import math
from typing import Mapping
import numpy as np
from src.common.representation import to_tokens
from src.data.drum_difficulty_v2 import ATTRS, Bar as DifficultyBar, Hit, discretize, note_spec, score_bar

DIFFICULTY_KEYS = ATTRS[:5]


@dataclass(frozen=True)
class Difficulty:
    raw: dict[str, float]
    classes: dict[str, int]
    normalized: dict[str, float]
    rhythm_reference: dict[str, float | int]


class DifficultyModel:
    """境界と語彙を明示して難易度を計算する。探索処理には依存しない。"""

    def __init__(self, boundaries, vocabulary):
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
