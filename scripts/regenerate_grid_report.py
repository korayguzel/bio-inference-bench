#!/usr/bin/env python3
"""Regenerate the grid report from existing raw JSON artifacts.

Uses the same canonical reporting logic as run_grid.py
(bio_inference_bench.grid_report). Does NOT rerun any benchmarks.

Usage:
    python scripts/regenerate_grid_report.py
    python scripts/regenerate_grid_report.py --notes "Known issue: ..."
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from bio_inference_bench.grid_report import aggregate_group, generate_grid_report
from bio_inference_bench.profiler import get_gpu_info
from bio_inference_bench.utils import timestamp


def load_all_grid_runs(raw_dir: Path) -> dict[str, list[dict]]:
    """Load all grid JSON files, grouped by (model, prompt, max_new)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(raw_dir.glob("grid_*.json")):
        data = json.loads(path.read_text())
        pr = data["primary_result"]
        key = f"{pr['model_name']}|{pr['prompt_token_length']}|{pr['max_new_tokens']}"
        groups[key].append(data)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate grid report from saved artifacts")
    parser.add_argument("--notes", nargs="*", help="Additional notes to include in the report")
    args = parser.parse_args()

    raw_dir = Path("results/raw")
    groups = load_all_grid_runs(raw_dir)

    if not groups:
        print("No grid_*.json files found in results/raw/")
        return

    print(f"Loaded {sum(len(v) for v in groups.values())} runs across {len(groups)} configs")

    gpu_info = get_gpu_info()
    summaries = [aggregate_group(runs) for _, runs in sorted(groups.items())]

    ts = timestamp()

    summary_path = Path("results/summaries") / f"grid_summary_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    report_path = Path("results/summaries") / f"grid_report_{ts}.md"
    generate_grid_report(summaries, gpu_info, report_path, metadata_notes=args.notes)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
