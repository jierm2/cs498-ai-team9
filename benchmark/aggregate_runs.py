"""Aggregate per-trial result JSONs into a single accuracy summary.

Reads every benchmark result file in `benchmark/results/`, groups by
`agent_name`, and reports mean and std accuracy across trials. Also emits a
per-task PASS/FAIL matrix and a markdown table you can paste into the paper.

Usage:
    python benchmark/aggregate_runs.py
    python benchmark/aggregate_runs.py --results-dir benchmark/results
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"


def load_run(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_runs(results_dir: Path) -> list[dict]:
    runs = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name.startswith("."):
            continue
        try:
            payload = load_run(path)
        except json.JSONDecodeError:
            continue
        agent_name = payload.get("agent_name") or _infer_agent_from_filename(path.name)
        trial_id = payload.get("trial_id") or _infer_trial_from_filename(path.name)
        runs.append({
            "path": path,
            "agent_name": agent_name,
            "trial_id": trial_id,
            "summary": payload.get("summary", {}),
            "results": payload.get("results", []),
        })
    return runs


def _infer_agent_from_filename(name: str) -> str:
    stem = name.removesuffix(".json")
    if "no_recap" in stem or "ablation" in stem:
        return "agent_no_recap"
    if "baseline" in stem:
        return "baseline"
    if "agent" in stem:
        return "three_phase_agent"
    return "unknown"


def _infer_trial_from_filename(name: str) -> str:
    for tag in ("t1", "t2", "t3"):
        if f"_{tag}" in name:
            return tag
    return ""


def summarize(runs: list[dict]) -> dict[str, list[dict]]:
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        if not run["agent_name"]:
            continue
        results = run["results"]
        if not results:
            continue
        passed = sum(1 for r in results if r.get("passed"))
        total = len(results)
        accuracy = passed / total if total else 0.0
        by_agent[run["agent_name"]].append({
            "trial_id": run["trial_id"],
            "passed": passed,
            "total": total,
            "accuracy": accuracy,
            "path": run["path"].name,
        })
    return by_agent


def per_task_matrix(runs: list[dict], agent_name: str) -> dict[str, dict[str, bool]]:
    matrix: dict[str, dict[str, bool]] = defaultdict(dict)
    for run in runs:
        if run["agent_name"] != agent_name:
            continue
        trial = run["trial_id"] or run["path"].name
        for r in run["results"]:
            tid = r.get("task_id", "?")
            matrix[tid][trial] = bool(r.get("passed"))
    return matrix


def print_summary(by_agent: dict[str, list[dict]]) -> None:
    print("=" * 78)
    print("ACCURACY BY AGENT (across trials)")
    print("=" * 78)
    print(f"{'agent':<25} {'trials':>8} {'mean':>10} {'std':>10}    per-trial")
    print("-" * 78)
    for agent, trials in sorted(by_agent.items()):
        accs = [t["accuracy"] for t in trials]
        mean = statistics.mean(accs) if accs else 0.0
        std = statistics.stdev(accs) if len(accs) > 1 else 0.0
        per_trial = "  ".join(
            f"{t['trial_id'] or '?'}={t['passed']}/{t['total']}" for t in trials
        )
        print(f"{agent:<25} {len(trials):>8} {mean*100:>9.1f}% {std*100:>9.1f}%    {per_trial}")
    print()


def print_markdown_table(by_agent: dict[str, list[dict]]) -> None:
    print("=" * 78)
    print("MARKDOWN TABLE (paste into paper)")
    print("=" * 78)
    rows = []
    for agent, trials in sorted(by_agent.items()):
        accs = [t["accuracy"] for t in trials]
        mean = statistics.mean(accs) if accs else 0.0
        std = statistics.stdev(accs) if len(accs) > 1 else 0.0
        per_trial = " / ".join(f"{t['passed']}/{t['total']}" for t in trials)
        rows.append((agent, per_trial, mean, std))
    print()
    print("| Condition | Per-trial | Mean | Std |")
    print("|---|---|---|---|")
    for agent, per_trial, mean, std in rows:
        print(f"| {agent} | {per_trial} | {mean*100:.1f}% | {std*100:.1f} |")
    print()


def print_per_task_table(runs: list[dict], by_agent: dict[str, list[dict]]) -> None:
    print("=" * 78)
    print("PER-TASK PASS/FAIL MATRIX (✓ = PASS, ✗ = FAIL, blank = not run)")
    print("=" * 78)
    all_task_ids = set()
    for run in runs:
        for r in run["results"]:
            tid = r.get("task_id")
            if tid:
                all_task_ids.add(tid)
    task_ids = sorted(all_task_ids)
    agents = sorted(by_agent.keys())

    matrices = {a: per_task_matrix(runs, a) for a in agents}

    header = f"{'task':<60}"
    for a in agents:
        header += f" {a[:18]:<18}"
    print(header)
    print("-" * len(header))

    for tid in task_ids:
        row = f"{tid:<60}"
        for a in agents:
            cell = matrices[a].get(tid, {})
            if not cell:
                row += f" {'':<18}"
            else:
                marks = "".join("✓" if v else "✗" for _, v in sorted(cell.items()))
                row += f" {marks:<18}"
        print(row)
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")

    runs = collect_runs(args.results_dir)
    if not runs:
        print(f"No result JSONs found in {args.results_dir}")
        return 1

    by_agent = summarize(runs)
    print_summary(by_agent)
    print_markdown_table(by_agent)
    print_per_task_table(runs, by_agent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
