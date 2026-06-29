#!/usr/bin/env python3
"""
Collect evaluation metrics from results.json files across checkpoint directories
and save as a spreadsheet-compatible CSV file.

Usage:
    python scripts/analysis/collect_eval_results.py \
        --checkpoints checkpoints_drum_standard-lakh1-kl6 checkpoints_drum_standard-lakh2-kl6 checkpoints_drum_standard-lakh3-kl6 checkpoints_drum_standard-lakh4-kl6 checkpoints_drum_standard-lakh5-kl6 \
        --metrics hihat_similarity rhythm_similarity token_iou Spearman_control \
        --output results_table.csv

Supported metric names:
    hihat_similarity     -> mean_hihat_similarity
    rhythm_similarity    -> mean_rhythm_similarity
    groove_similarity    -> mean_groove_similarity
    token_iou            -> mean_token_iou
    Spearman_control     -> expands to all 5 spearman_control_* correlations as separate columns
    <spearman key>       -> correlation value of that key in spearman dict
    <top-level key>      -> direct value from results.json
"""

import argparse
import csv
import json
from pathlib import Path

EVAL_DIR = Path(__file__).parent.parent.parent / "output" / "eval-new"

METRIC_ALIASES = {
    "hihat_similarity": "mean_hihat_similarity",
    "rhythm_similarity": "mean_rhythm_similarity",
    "groove_similarity": "mean_groove_similarity",
    "token_iou": "mean_token_iou",
}

SPEARMAN_CONTROL_KEYS = [
    "spearman_control_s_tech_cls",
    "spearman_control_s_indep_cls",
    "spearman_control_s_hand_cls",
    "spearman_control_s_foot_cls",
    "spearman_control_s_move_cls",
]


def expand_metrics(metrics):
    """Expand Spearman_control into individual spearman_control_* columns."""
    expanded = []
    for m in metrics:
        if m.lower() == "spearman_control":
            expanded.extend(SPEARMAN_CONTROL_KEYS)
        else:
            expanded.append(m)
    return expanded


def extract_metric(data, metric_name):
    if metric_name in data:
        return data[metric_name]

    alias = METRIC_ALIASES.get(metric_name)
    if alias and alias in data:
        return data[alias]

    mean_key = f"mean_{metric_name}"
    if mean_key in data:
        return data[mean_key]

    spearman = data.get("spearman", {})
    if metric_name in spearman:
        entry = spearman[metric_name]
        return entry.get("correlation", entry)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Collect evaluation metrics from results.json and save as CSV."
    )
    parser.add_argument(
        "--checkpoints", "-c",
        nargs="+",
        required=True,
        help="Checkpoint directory names under output/eval-new/.",
    )
    parser.add_argument(
        "--metrics", "-m",
        nargs="+",
        default=["hihat_similarity", "rhythm_similarity", "token_iou", "Spearman_control"],
        help="Metric column names to extract.",
    )
    parser.add_argument(
        "--output", "-o",
        default="results_table.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--eval-dir",
        default=str(EVAL_DIR),
        help="Base directory containing checkpoint evaluation directories.",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    metrics = expand_metrics(args.metrics)
    rows = []

    for checkpoint in args.checkpoints:
        results_path = eval_dir / checkpoint / "results.json"
        row = {"checkpoint": checkpoint}

        if not results_path.exists():
            print(f"Warning: {results_path} not found, skipping.")
            row.update({m: "" for m in metrics})
            rows.append(row)
            continue

        with open(results_path) as f:
            data = json.load(f)

        for metric in metrics:
            value = extract_metric(data, metric)
            if value is None:
                print(f"Warning: metric '{metric}' not found in {results_path}")
                row[metric] = ""
            else:
                row[metric] = round(value, 4) if isinstance(value, float) else value
        rows.append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["checkpoint"] + metrics
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output_path}")

    col_widths = [
        max(len(str(r.get(col, ""))) for r in [dict(zip(fieldnames, fieldnames))] + rows)
        for col in fieldnames
    ]
    sep = "-+-".join("-" * w for w in col_widths)
    print(" | ".join(f"{col:<{w}}" for col, w in zip(fieldnames, col_widths)))
    print(sep)
    for row in rows:
        vals = [str(row.get(col, "")) for col in fieldnames]
        print(" | ".join(f"{v:<{w}}" for v, w in zip(vals, col_widths)))


if __name__ == "__main__":
    main()
