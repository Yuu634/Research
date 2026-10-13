"""ステップ4：探索係数から独立した固定基準で編曲結果を評価・集計する。"""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math
from statistics import mean, median
from scipy.stats import spearmanr

from src.common.difficulty import DIFFICULTY_KEYS
from src.common.metrics import musical_edit_distance, onset_similarities, skeleton_similarity

EPSILON = 1e-12
# 探索モデルの重みを変更しても、この評価用の成分重みは変えない。
EVALUATION_SKELETON_WEIGHTS = (.45, .45, .1, .1)


@dataclass(frozen=True)
class EvaluationConfig:
    target_tolerance: float = .5 / 7
    skeleton_threshold: float = .90

    def validate(self):
        for value in (self.target_tolerance, self.skeleton_threshold):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('評価閾値は0〜1の有限数')
        return self


def _errors(difficulty, target):
    return {key: abs(difficulty.normalized[key] - target[key]) for key in DIFFICULTY_KEYS}


def evaluate_result(prepared, candidate, model, *, case_id, source_id,
                    rule_count=0, elapsed_seconds=None, failure_reason=None,
                    config=EvaluationConfig()):
    """1入力につき1出力を評価する。無出力もcandidate=Noneで必ず記録する。

    candidateは原譜のIDを継承した内部Barを渡す（出力トークンから再採番しない）。
    source_idは元の楽曲を識別し、後の曲単位マクロ平均で使用する。
    """
    config.validate()
    if any(not isinstance(value, str) or not value for value in (case_id, source_id)):
        raise ValueError('case_id・source_idは空でない文字列')
    if type(rule_count) is not int or rule_count < 0:
        raise ValueError('適用ルール数は非負整数')
    if elapsed_seconds is not None and (
        type(elapsed_seconds) not in (int, float) or not math.isfinite(elapsed_seconds) or elapsed_seconds < 0
    ):
        raise ValueError('実行秒数は非負の有限数またはNone')
    if failure_reason is not None and (not isinstance(failure_reason, str) or not failure_reason):
        raise ValueError('failure_reasonは空でない文字列またはNone')
    if candidate is not None and failure_reason is not None:
        raise ValueError('出力と出力失敗理由は同時に指定しない')
    baseline = model.score(prepared.original, prepared.original, prepared.target,
                           previous=prepared.previous, guards=prepared.guards)
    if not baseline.valid:
        raise ValueError('評価入力自体が不適格: ' + ', '.join(baseline.reasons))
    before_errors = _errors(baseline.difficulty, prepared.target)
    baseline_mae = mean(before_errors.values())
    original_skeleton, original_components, _ = skeleton_similarity(
        prepared.original, prepared.original, EVALUATION_SKELETON_WEIGHTS)
    baseline_target_success = max(before_errors.values()) <= config.target_tolerance + EPSILON
    baseline_joint = baseline_target_success and original_skeleton >= config.skeleton_threshold - EPSILON
    baseline_joint = baseline_joint and any(value is not None for value in original_components.values())
    row = dict(
        schema_version=1, case_id=case_id, source_id=source_id, evaluation_config=asdict(config),
        status='failed', reasons=[failure_reason or 'no_output'], elapsed_seconds=elapsed_seconds,
        rule_count=rule_count, target=dict(prepared.target), baseline_mae=baseline_mae,
        baseline_attribute_errors=before_errors, baseline_target_success=baseline_target_success,
        baseline_joint_success=baseline_joint, target_success=False, joint_success=False,
        # 不適格・無出力は正規化MAEの最大値1として全入力の分母に残す。
        failure_adjusted_mae=1., metrics=None,
    )
    if candidate is None:
        return row
    result = model.score(prepared.original, candidate, prepared.target, rule_count=rule_count,
                         previous=prepared.previous, guards=prepared.guards)
    if not result.valid:
        row.update(status='invalid', reasons=list(result.reasons))
        return row
    after_errors = _errors(result.difficulty, prepared.target)
    mae = mean(after_errors.values())
    requested = {key: prepared.target[key]-baseline.difficulty.normalized[key] for key in DIFFICULTY_KEYS}
    actual = {key: result.difficulty.normalized[key]-baseline.difficulty.normalized[key] for key in DIFFICULTY_KEYS}
    changed_attributes = [key for key in DIFFICULTY_KEYS if abs(requested[key]) > EPSILON]
    held_attributes = [key for key in DIFFICULTY_KEYS if abs(requested[key]) <= EPSILON]
    directions = {key: (actual[key] > EPSILON if requested[key] > 0 else actual[key] < -EPSILON)
                  for key in changed_attributes}
    progress = {key: after_errors[key] < before_errors[key]-EPSILON for key in changed_attributes}
    skeleton, components, effective = skeleton_similarity(
        prepared.original, candidate, EVALUATION_SKELETON_WEIGHTS)
    evidence = any(value is not None for value in components.values())
    target_success = max(after_errors.values()) <= config.target_tolerance + EPSILON
    joint_success = target_success and evidence and skeleton >= config.skeleton_threshold-EPSILON
    edit = musical_edit_distance(prepared.original, candidate)
    metrics = dict(
        attribute_errors=after_errors, difficulty=asdict(result.difficulty), mae=mae,
        max_attribute_error=max(after_errors.values()), exact_target=max(after_errors.values()) <= EPSILON,
        within_one_class=max(after_errors.values()) <= 1/7+EPSILON,
        mae_improvement=baseline_mae-mae, improved=mae < baseline_mae-EPSILON,
        requested_delta=requested, actual_delta=actual, direction_matches=directions,
        requested_attribute_progress=progress,
        direction_rate=mean(directions.values()) if directions else None,
        requested_progress_rate=mean(progress.values()) if progress else None,
        held_attribute_mae=mean(abs(actual[key]) for key in held_attributes) if held_attributes else None,
        skeleton_similarity=skeleton if evidence else None, skeleton_components=components,
        skeleton_effective_weights=effective, skeleton_evidence=evidence,
        edit_distance=edit, changed=edit > EPSILON,
        note_count_delta=len(candidate.notes)-len(prepared.original.notes),
        **onset_similarities(prepared.original, candidate),
    )
    row.update(status='valid', reasons=[], metrics=metrics, target_success=target_success,
               joint_success=joint_success, failure_adjusted_mae=mae)
    return row


def _average(values):
    values = [value for value in values if value is not None]
    return dict(mean=mean(values) if values else None, n=len(values))


def _rate(values):
    values = list(values)
    count = sum(bool(value) for value in values)
    return dict(value=count/len(values) if values else None, numerator=count, denominator=len(values))


def _spearman(targets, generated):
    """定数列・少数件は相関を捏造せず、理由付きの欠測値として返す。"""
    n = len(targets)
    reason = ('insufficient_samples' if n < 2 else
              'constant_target' if len(set(targets)) < 2 else
              'constant_generated' if len(set(generated)) < 2 else None)
    coefficient = None if reason else float(spearmanr(targets, generated).statistic)
    return dict(coefficient=coefficient, n=n, reason=reason)


def summarize_control_metrics(rows):
    """適格出力について属性制御と原譜保持を集計する。既存の選定指標は変更しない。

    ラベル距離はabs(7 * 正規化目標 - 生成ラベル)。連続目標も丸めない。
    類似度は小節ごとのF1を平均し、旧結果の未記録値・両方空は除外する。
    """
    valid = [row for row in rows if row['status'] == 'valid']
    distances = {key: [abs(7 * row['target'][key] - row['metrics']['difficulty']['classes'][key])
                       for row in valid] for key in DIFFICULTY_KEYS}
    return dict(
        attribute_spearman={key: _spearman(
            [row['target'][key] for row in valid],
            [row['metrics']['difficulty']['classes'][key] for row in valid]) for key in DIFFICULTY_KEYS},
        attribute_label_distance={key: _average(values) for key, values in distances.items()},
        label_distance=_average(mean(values) for values in zip(*distances.values())),
        rhythm_similarity=_average(row['metrics'].get('rhythm_similarity') for row in valid),
        hihat_similarity=_average(row['metrics'].get('hihat_similarity') for row in valid),
    )


def summarize_results(rows):
    """同じ評価基準・重複のない入力集合を集計。欠測の分母を項目ごとに保存する。"""
    rows = list(rows)
    if len({row['case_id'] for row in rows}) != len(rows):
        raise ValueError('同一集計内のcase_id重複。モデルごとに分けて集計する')
    if any(row['evaluation_config'] != rows[0]['evaluation_config'] for row in rows):
        raise ValueError('異なる評価閾値を同じ集計に混在させない')
    if any(row['status'] not in ('valid', 'invalid', 'failed') for row in rows):
        raise ValueError('未知の評価状態')
    valid = [row for row in rows if row['status'] == 'valid']
    metrics = [row['metrics'] for row in valid]
    times = [row['elapsed_seconds'] for row in rows if row['elapsed_seconds'] is not None]
    continuous = {name: _average(metric[name] for metric in metrics) for name in (
        'mae', 'max_attribute_error', 'mae_improvement', 'direction_rate', 'requested_progress_rate',
        'held_attribute_mae', 'skeleton_similarity', 'edit_distance', 'note_count_delta')}
    components = {key: _average(metric['skeleton_components'][key] for metric in metrics)
                  for key in ('kick', 'snare', 'accent', 'fill')}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['source_id']].append(row)
    source_rows = [dict(source_id=source, n=len(group),
                        valid_rate=mean(row['status'] == 'valid' for row in group),
                        joint_success_rate=mean(row['joint_success'] for row in group),
                        failure_adjusted_mae=mean(row['failure_adjusted_mae'] for row in group),
                        baseline_mae=mean(row['baseline_mae'] for row in group),
                        baseline_joint_success_rate=mean(row['baseline_joint_success'] for row in group))
                   for source, group in sorted(grouped.items())]
    return dict(
        n=len(rows), evaluation_config=rows[0]['evaluation_config'] if rows else None,
        status_counts={status: sum(row['status'] == status for row in rows)
                       for status in ('valid', 'invalid', 'failed')},
        valid_rate=_rate(row['status'] == 'valid' for row in rows),
        target_success_rate=_rate(row['target_success'] for row in rows),
        joint_success_rate=_rate(row['joint_success'] for row in rows),
        exact_target_rate=_rate(bool(row['metrics'] and row['metrics']['exact_target']) for row in rows),
        within_one_class_rate=_rate(bool(row['metrics'] and row['metrics']['within_one_class']) for row in rows),
        improvement_rate=_rate(bool(row['metrics'] and row['metrics']['improved']) for row in rows),
        unchanged_rate_valid=_rate(not metric['changed'] for metric in metrics),
        skeleton_evidence_rate_valid=_rate(metric['skeleton_evidence'] for metric in metrics),
        baseline_mae=_average(row['baseline_mae'] for row in rows),
        baseline_target_success_rate=_rate(row['baseline_target_success'] for row in rows),
        baseline_joint_success_rate=_rate(row['baseline_joint_success'] for row in rows),
        failure_adjusted_mae=_average(row['failure_adjusted_mae'] for row in rows),
        valid_output_means=continuous, skeleton_component_means=components,
        attribute_mae_valid={key: _average(metric['attribute_errors'][key] for metric in metrics)
                             for key in DIFFICULTY_KEYS},
        rule_count_valid=_average(row['rule_count'] for row in valid),
        runtime_seconds=dict(mean=mean(times) if times else None, median=median(times) if times else None,
                             maximum=max(times) if times else None, n=len(times)),
        reason_counts=dict(Counter(reason for row in rows for reason in row['reasons'])),
        source_macro={key: _average(row[key] for row in source_rows) for key in (
            'valid_rate', 'joint_success_rate', 'failure_adjusted_mae', 'baseline_mae', 'baseline_joint_success_rate')},
        sources=source_rows,
        **summarize_control_metrics(rows),
    )
