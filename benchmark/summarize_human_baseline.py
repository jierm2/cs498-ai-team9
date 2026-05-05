from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"


def load_human_runs(results_dir: Path) -> list[dict]:
    runs = []
    for path in sorted(results_dir.glob("human_baseline_*.json")):
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
        print(f"No human_baseline_*.json files found in {args.results_dir}")
        return 1

    by_author: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        author = run.get("human_author") or run.get("trial_id") or run["_path"]
        by_author[author].extend(run["results"])

    total_passed = 0
    total_tasks = 0
    print("Human baseline results")
    print("----------------------")
    for author, results in sorted(by_author.items()):
        passed = sum(1 for r in results if r.get("passed"))
        total = len(results)
        total_passed += passed
        total_tasks += total
        print(f"{author}: {passed}/{total} ({passed / total:.1%})")

    mean_pct = total_passed / total_tasks if total_tasks else 0.0
    label = "Human baseline" if len(by_author) == 1 else "Joint"
    print(f"{label}: {total_passed}/{total_tasks} ({mean_pct:.1%})")
    print()
    print("LaTeX sentence:")
    if len(by_author) == 1:
        print(f"The human baseline is \\\\textbf{{{total_passed}/{total_tasks}}} ({mean_pct * 100:.1f}\\\\%) over five task runs.")
    else:
        author_parts = [
            f"{author} scored \\\\textbf{{{sum(1 for r in results if r.get('passed'))}/{len(results)}}}"
            for author, results in sorted(by_author.items())
        ]
        print(
            "; ".join(author_parts)
            + f"; joint mean \\\\textbf{{{mean_pct * 100:.1f}\\\\%}} (n={total_tasks})."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
