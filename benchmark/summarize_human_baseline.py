from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"


def load_human_runs(results_dir: Path) -> list[dict]:
    runs = []
    paths = sorted(
        set(results_dir.glob("human_baseline_*.json"))
        | set(results_dir.glob("human_t*.json"))
    )
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("agent_name") == "human_baseline" and payload.get("results"):
            payload["_path"] = path.name
            runs.append(payload)
    return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize stdin-driven human baseline runs.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs = load_human_runs(args.results_dir)
    if not runs:
        print(f"No human_baseline_*.json or human_t*.json files found in {args.results_dir}")
        return 1

    total_passed = 0
    total_tasks = 0
    print("Human baseline results")
    print("----------------------")
    for run in runs:
        results = run["results"]
        passed = sum(1 for r in results if r.get("passed"))
        total = len(results)
        total_passed += passed
        total_tasks += total
        label = run.get("trial_id") or run["_path"]
        print(f"{label}: {passed}/{total} ({passed / total:.1%})")

    mean_pct = total_passed / total_tasks if total_tasks else 0.0
    print(f"Human baseline: {total_passed}/{total_tasks} ({mean_pct:.1%})")
    print()
    print("LaTeX sentence:")
    print(f"The human baseline is \\\\textbf{{{total_passed}/{total_tasks}}} ({mean_pct * 100:.1f}\\\\%) over {total_tasks} task runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
