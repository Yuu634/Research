"""難易度・骨格・ルール数・編集量からBeamSearchの目的関数を計算する。"""
from dataclasses import dataclass
import math
from src.common.difficulty import DIFFICULTY_KEYS, Difficulty, DifficultyModel
from src.common.metrics import musical_edit_distance, skeleton_similarity
from src.common.safety import default_guards, violations


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


class ScoringModel(DifficultyModel):
    """共通の難易度計算に探索係数と候補採点を追加する。"""

    def __init__(self, boundaries, vocabulary, config=ScoreConfig()):
        self.config = config.validate()
        super().__init__(boundaries, vocabulary)

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
