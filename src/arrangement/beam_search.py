"""原譜の制約を保持し、複数段のルール適用を探索する決定的BeamSearch。"""
from collections import Counter
from dataclasses import asdict, dataclass
import inspect
import json
from functools import cached_property
from time import perf_counter

from src.arrangement.representation import Bar
from src.arrangement.rules import Context, REGISTRY, apply_rule
from src.arrangement.scoring import ScoreBreakdown, ScoringModel


@dataclass(frozen=True)
class Action:
    name: str
    parameters: tuple = ()

    def validate(self):
        if self.name not in REGISTRY:
            raise ValueError('未知のルール: ' + self.name)
        parameters = dict(self.parameters)
        if len(parameters) != len(self.parameters) or 'ctx' in parameters:
            raise ValueError('ルール引数の重複またはctx指定')
        try:
            inspect.signature(REGISTRY[self.name]).bind(Bar(()), ctx=Context(), **parameters)
            json.dumps(parameters, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError('ルール引数が不正: ' + self.name) from error
        return self

    def as_dict(self):
        return dict(name=self.name, parameters=dict(self.parameters))

    @cached_property
    def key(self):
        return json.dumps(self.as_dict(), sort_keys=True, ensure_ascii=False)


def standard_actions():
    """65関数と引数違い。区間追加は明示的に最終拍[72,96)を指定し、Fillとは呼ばない。"""
    interval_rules = {'add_roll', 'add_hand_roll', 'add_kick_triplet', 'add_double_bass_pattern',
                      'add_hand_foot_interleave', 'add_polyrhythmic_relation'}
    actions = []
    for name in sorted(REGISTRY):
        parameters = dict(start=72, end=96) if name in interval_rules else {}
        actions.append(Action(name, tuple(sorted(parameters.items()))))
    for name, key, values in (
        ('reduce_ghost_note_ratio', 'r', (.25, .75)),
        ('reduce_kick_density', 'r', (.25, .75)),
        ('limit_hand_min_interval', 't', (1/12, 1/6, .25)),
        ('limit_kick_min_interval', 't', (1/12, 1/6, .25)),
        ('simplify_hand_foot_pattern', 'step', (6, 24)),
        ('merge_hand_foot_onsets', 'max_shift', (1, 6)),
    ):
        actions.extend(Action(name, ((key, value),)) for value in values)
    return tuple(sorted((action.validate() for action in actions), key=lambda action: action.key))


@dataclass(frozen=True)
class SearchConfig:
    beam_width: int = 4
    max_depth: int = 2
    max_rule_applications: int = 5000

    def validate(self):
        for value in (self.beam_width, self.max_rule_applications):
            if type(value) is not int or value < 1:
                raise ValueError('探索幅・ルール試行上限は正整数')
        if type(self.max_depth) is not int or self.max_depth < 0:
            raise ValueError('探索深さは非負整数')
        return self


@dataclass(frozen=True)
class SearchResult:
    candidate: Bar
    score: ScoreBreakdown
    history: tuple[Action, ...]
    statistics: dict
    elapsed_seconds: float


class _CachedScorer(ScoringModel):
    """1入力内だけのキャッシュ。元譜の難易度を候補ごとに再計算しない。"""
    def __init__(self, model):
        self.config, self.boundaries, self.vocabulary = model.config, model.boundaries, model.vocabulary
        self.model = model
        self.cache = {}

    def difficulty(self, bar, previous=None):
        key = bar, previous
        if key not in self.cache:
            self.cache[key] = self.model.difficulty(bar, previous)
        return self.cache[key]


def _order(node):
    bar, score, history = node
    return (-score.total, len(history), score.edit_distance, tuple(action.key for action in history))


def beam_search(prepared, model, config=SearchConfig(), actions=None):
    config.validate()
    actions = standard_actions() if actions is None else tuple(action.validate() for action in actions)
    if len({action.key for action in actions}) != len(actions):
        raise ValueError('同じアクションの重複')
    actions = tuple(sorted(actions, key=lambda action: action.key))
    start = perf_counter()
    scorer = _CachedScorer(model)
    original = prepared.original
    root = original.with_notes(original.notes)
    initial = scorer.score(original, root, prepared.target, previous=prepared.previous, guards=prepared.guards)
    if not initial.valid:
        raise ValueError('探索入力が不適格: ' + ', '.join(initial.reasons))
    best = (root, initial, ())
    frontier = [best]
    expanded = set()
    counts = Counter()
    invalid_reasons = Counter()
    layers = []
    stopped = 'max_depth'
    for depth in range(1, config.max_depth+1):
        proposals = {}
        exhausted_budget = False
        expanded.update(node[0] for node in frontier)
        for current, _, history in frontier:
            for action in actions:
                if counts['rule_applications'] >= config.max_rule_applications:
                    exhausted_budget = True
                    break
                counts['rule_applications'] += 1
                result = apply_rule(current, action.name, original=original, guards=prepared.guards,
                                    **dict(action.parameters))
                counts[result.status] += 1
                if result.status != 'applied':
                    invalid_reasons.update(result.reason if result.status == 'rejected' else ())
                    continue
                candidate = result.after.with_notes(result.after.notes)
                new_history = history+(action,)
                if candidate in expanded:
                    counts['expanded_duplicate'] += 1
                    continue
                if candidate in proposals:
                    counts['layer_duplicate'] += 1
                    # 同じ深さ・同じ状態なら最初の決定的経路を保持する。
                    continue
                score = scorer.score(original, candidate, prepared.target, rule_count=depth,
                                     previous=prepared.previous, guards=prepared.guards)
                counts['scored'] += 1
                if not score.valid:
                    counts['score_invalid'] += 1
                    invalid_reasons.update(score.reasons)
                    continue
                node = (candidate, score, new_history)
                proposals[candidate] = node
                if _order(node) < _order(best):
                    best = node
            if exhausted_budget:
                break
        frontier = sorted(proposals.values(), key=_order)[:config.beam_width]
        layers.append(dict(depth=depth, unique_valid_candidates=len(proposals), kept=len(frontier),
                           best_score=best[1].total))
        if exhausted_budget:
            stopped = 'rule_application_budget'
            break
        if not frontier:
            stopped = 'no_new_candidates'
            break
    statistics = dict(config=asdict(config), actions=len(actions), counters=dict(counts),
                      invalid_reasons=dict(invalid_reasons), layers=layers, stopped=stopped,
                      difficulty_cache_entries=len(scorer.cache), baseline_score=initial.total)
    return SearchResult(best[0], best[1], best[2], statistics, perf_counter()-start)
