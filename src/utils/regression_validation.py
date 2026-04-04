from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Any


@dataclass(frozen=True)
class RegressionMetricResult:
    metric: str
    baseline: float
    current: float
    relative_diff: float
    threshold: float
    passed: bool


def _to_float_dict(values: Mapping[str, Any], context: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in values.items():
        try:
            out[key] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric value for {context}.{key}: {value!r}") from exc
    return out


def evaluate_regression_thresholds(
    baseline_metrics: Mapping[str, Any],
    current_metrics: Mapping[str, Any],
    threshold_by_metric: Mapping[str, Any],
    epsilon: float = 1e-8,
) -> Dict[str, Any]:
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    baseline = _to_float_dict(baseline_metrics, "baseline_metrics")
    current = _to_float_dict(current_metrics, "current_metrics")
    thresholds = _to_float_dict(threshold_by_metric, "threshold_by_metric")

    missing_baseline = [k for k in thresholds if k not in baseline]
    missing_current = [k for k in thresholds if k not in current]
    if missing_baseline or missing_current:
        raise ValueError(
            "Regression threshold evaluation requires matching metric keys. "
            f"missing_baseline={missing_baseline}, missing_current={missing_current}"
        )

    metric_results: Dict[str, RegressionMetricResult] = {}
    for metric in sorted(thresholds.keys()):
        base = baseline[metric]
        curr = current[metric]
        threshold = thresholds[metric]
        if threshold < 0:
            raise ValueError(f"Threshold must be non-negative for metric={metric}, got {threshold}")

        denom = max(abs(base), epsilon)
        relative_diff = abs(curr - base) / denom
        passed = relative_diff <= threshold
        metric_results[metric] = RegressionMetricResult(
            metric=metric,
            baseline=base,
            current=curr,
            relative_diff=relative_diff,
            threshold=threshold,
            passed=passed,
        )

    failed_metrics = [m for m, r in metric_results.items() if not r.passed]
    return {
        "passed": len(failed_metrics) == 0,
        "failed_metrics": failed_metrics,
        "metrics": {
            metric: {
                "baseline": result.baseline,
                "current": result.current,
                "relative_diff": result.relative_diff,
                "threshold": result.threshold,
                "passed": result.passed,
            }
            for metric, result in metric_results.items()
        },
    }


def build_mode_acceptance_report(
    mode: str,
    requirement_results: Mapping[str, Any],
    regression_result: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    report = {
        "mode": mode,
        "requirements": dict(sorted(requirement_results.items())),
        "regression": regression_result,
        "metadata": dict(sorted((metadata or {}).items())),
    }
    report["passed"] = bool(
        report["regression"].get("passed", False)
        and all(bool(v) for v in report["requirements"].values())
    )
    return report


def write_acceptance_report(report_path: str, report: Mapping[str, Any]) -> str:
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
    return str(path)
