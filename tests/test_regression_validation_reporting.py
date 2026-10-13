import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.regression_validation import (
    build_mode_acceptance_report,
    evaluate_regression_thresholds,
    write_acceptance_report,
)


class RegressionValidationReportingTests(unittest.TestCase):
    def test_threshold_passes_when_within_limit(self):
        baseline = {"recons_loss": 1.0, "kldiv_raw": 0.5}
        current = {"recons_loss": 1.05, "kldiv_raw": 0.52}
        threshold = {"recons_loss": 0.06, "kldiv_raw": 0.05}

        result = evaluate_regression_thresholds(baseline, current, threshold)
        self.assertTrue(result["passed"])
        self.assertEqual(result["failed_metrics"], [])

    def test_threshold_fails_when_exceeds_limit(self):
        baseline = {"recons_loss": 1.0, "kldiv_raw": 0.5}
        current = {"recons_loss": 1.2, "kldiv_raw": 0.52}
        threshold = {"recons_loss": 0.1, "kldiv_raw": 0.1}

        result = evaluate_regression_thresholds(baseline, current, threshold)
        self.assertFalse(result["passed"])
        self.assertIn("recons_loss", result["failed_metrics"])

    def test_acceptance_report_includes_mode_and_requirements(self):
        regression = {
            "passed": True,
            "failed_metrics": [],
            "metrics": {
                "recons_loss": {
                    "baseline": 1.0,
                    "current": 1.01,
                    "relative_diff": 0.01,
                    "threshold": 0.05,
                    "passed": True,
                }
            },
        }
        report = build_mode_acceptance_report(
            mode="remi",
            requirement_results={"7.1": True, "7.2": True, "7.3": True, "7.4": True},
            regression_result=regression,
            metadata={"seed": 42, "suite": "training-extension-drum-difficulty-conditioning"},
        )

        self.assertEqual(report["mode"], "remi")
        self.assertTrue(report["passed"])
        self.assertIn("requirements", report)
        self.assertIn("regression", report)

    def test_acceptance_report_write_is_deterministic(self):
        regression = {
            "passed": True,
            "failed_metrics": [],
            "metrics": {
                "recons_loss": {
                    "baseline": 1.0,
                    "current": 1.0,
                    "relative_diff": 0.0,
                    "threshold": 0.05,
                    "passed": True,
                }
            },
        }
        report = build_mode_acceptance_report(
            mode="cp_limb_v1",
            requirement_results={"7.1": True, "7.2": True, "7.3": True, "7.4": True},
            regression_result=regression,
            metadata={"seed": 42},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = os.path.join(tmpdir, "report_1.json")
            p2 = os.path.join(tmpdir, "report_2.json")
            write_acceptance_report(p1, report)
            write_acceptance_report(p2, report)

            with open(p1, "r", encoding="utf-8") as f:
                c1 = f.read()
            with open(p2, "r", encoding="utf-8") as f:
                c2 = f.read()

            self.assertEqual(c1, c2)
            # also assert valid JSON payload
            parsed = json.loads(c1)
            self.assertEqual(parsed["mode"], "cp_limb_v1")


if __name__ == "__main__":
    unittest.main()
