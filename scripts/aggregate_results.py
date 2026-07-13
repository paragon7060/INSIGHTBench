"""Aggregate per-task JSON result files into a summary table.

Usage:
    python scripts/aggregate_results.py --results_dir outputs/results/pi0
    python scripts/aggregate_results.py --results_dir outputs/results/pi0 --save_csv
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from insightbench.utils.results import aggregate_results

parser = argparse.ArgumentParser(description="Aggregate InsightBench evaluation results.")
parser.add_argument("--results_dir", type=str, required=True)
parser.add_argument("--save_csv",    action="store_true", help="Save summary as CSV next to results_dir.")


def main() -> None:
    args = parser.parse_args()
    summary = aggregate_results(args.results_dir)

    if not summary:
        return

    if args.save_csv:
        import csv
        csv_path = os.path.join(args.results_dir, "summary.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["object", "successes", "attempts", "rate"])
            for obj, s in summary["per_object"].items():
                w.writerow([obj, s["successes"], s["attempts"], s["rate"]])
            w.writerow(["TOTAL", summary["total"]["successes"], summary["total"]["attempts"], summary["total"]["rate"]])
        print(f"[Results] Summary saved to {csv_path}")


if __name__ == "__main__":
    main()
