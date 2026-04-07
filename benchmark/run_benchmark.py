from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_TASK_PATH = PROJECT_ROOT / "benchmark" / "benchmark.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmark" / "results"


def load_tasks(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_task(task: dict) -> dict:
    normalized = dict(task)
    misconception_text = (
        normalized.get("misconception")
        or normalized.get("simulator_prompt")
        or normalized.get("student_initial_reasoning")
        or normalized.get("misconception_label")
        or "Unknown misconception"
    )

    normalized["misconception"] = misconception_text
    normalized["max_turns"] = int(normalized.get("max_turns", 8))
    return normalized


def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("\u2212", "-")
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(".?!")
    return text


def _extract_numeric(text: str) -> str | None:
    cleaned = text.strip().lower().replace("\u2212", "-")
    mixed = re.search(r"-?\d+\s+\d+/\d+", cleaned)
    if mixed:
        return mixed.group(0)
    frac = re.search(r"-?\d+/\d+", cleaned)
    if frac:
        return frac.group(0)
    num = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if num:
        return num.group(0)
    return None


def _to_fraction(value: str) -> Fraction | None:
    v = value.strip().lower().replace("\u2212", "-")
    v = re.sub(r"\s+", " ", v)

    mixed_match = re.fullmatch(r"(-?\d+)\s+(\d+)/(\d+)", v)
    if mixed_match:
        whole = int(mixed_match.group(1))
        num = int(mixed_match.group(2))
        den = int(mixed_match.group(3))
        if den == 0:
            return None
        frac = Fraction(num, den)
        return Fraction(whole, 1) + frac if whole >= 0 else Fraction(whole, 1) - frac

    try:
        return Fraction(v)
    except Exception:
        return None


def grade_transfer_answer(task: dict, student_answer: str) -> tuple[bool, str]:
    accepted = task.get("accepted_answers") or [task.get("correct_answer", "")]
    accepted = [str(a) for a in accepted if str(a).strip()]

    answer_norm = _normalize_text(student_answer)
    accepted_norm = [_normalize_text(a) for a in accepted]

    if answer_norm in accepted_norm:
        return True, "exact"

    if any(a in answer_norm for a in accepted_norm if len(a) >= 2):
        return True, "substring"

    answer_num = _extract_numeric(student_answer)
    answer_frac = _to_fraction(answer_num) if answer_num else None
    if answer_frac is not None:
        for candidate in accepted:
            candidate_num = _extract_numeric(candidate)
            candidate_frac = _to_fraction(candidate_num) if candidate_num else None
            if candidate_frac is not None and candidate_frac == answer_frac:
                return True, "numeric-equivalent"

    return False, "no-match"


def run_single_task(agent, task: dict, simulate_fn, verbose: bool = False) -> dict:
    sim_history: list[dict] = []
    initial_reasoning_used = False

    def get_student_response(prompt_text: str) -> str:
        nonlocal initial_reasoning_used

        sim_history.append({"role": "user", "text": prompt_text})
        is_transfer = prompt_text.lower().startswith("now answer this question")

        if not initial_reasoning_used and task.get("student_initial_reasoning") and not is_transfer:
            student_response = str(task["student_initial_reasoning"])
            initial_reasoning_used = True
        else:
            student_response = simulate_fn(task, sim_history)

        sim_history.append({"role": "model", "text": student_response})
        return student_response

    session = agent.run_session(task, get_student_response=get_student_response, verbose=verbose)
    passed, grading_mode = grade_transfer_answer(task, session.student_transfer_answer)

    final_state = session.final_tracker_state or {}
    return {
        "task_id": task.get("id"),
        "topic": task.get("topic", "unknown"),
        "passed": passed,
        "grading_mode": grading_mode,
        "transfer_question": session.transfer_question,
        "correct_answer": task.get("correct_answer", ""),
        "accepted_answers": task.get("accepted_answers", []),
        "student_transfer_answer": session.student_transfer_answer,
        "turns_taken": session.turns_taken,
        "tracker": final_state,
        "diagnosed": bool(final_state.get("misconception_identified", False)),
        "counter_example_shown": bool(final_state.get("counter_example_shown", False)),
        "confirmed_correction": bool(final_state.get("confirmed_correction", False)),
    }


def aggregate_summary(task_results: list[dict]) -> dict:
    total = len(task_results)
    passed = sum(1 for r in task_results if r["passed"])
    accuracy = (passed / total) if total else 0.0

    by_topic: dict[str, dict] = {}
    for result in task_results:
        topic = result["topic"]
        topic_bucket = by_topic.setdefault(topic, {"total": 0, "passed": 0})
        topic_bucket["total"] += 1
        if result["passed"]:
            topic_bucket["passed"] += 1

    for topic, values in by_topic.items():
        values["accuracy"] = values["passed"] / values["total"] if values["total"] else 0.0

    turns = [r["turns_taken"] for r in task_results]
    diagnosed_rate = mean([1 if r["diagnosed"] else 0 for r in task_results]) if task_results else 0.0
    counter_example_rate = mean([1 if r["counter_example_shown"] else 0 for r in task_results]) if task_results else 0.0
    confirmed_rate = mean([1 if r["confirmed_correction"] else 0 for r in task_results]) if task_results else 0.0

    return {
        "total_tasks": total,
        "passed_tasks": passed,
        "accuracy": accuracy,
        "avg_turns_taken": mean(turns) if turns else 0.0,
        "diagnosed_rate": diagnosed_rate,
        "counter_example_shown_rate": counter_example_rate,
        "confirmed_correction_rate": confirmed_rate,
        "by_topic": by_topic,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tutoring-agent benchmark against the student simulator.")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASK_PATH, help="Path to benchmark tasks JSON.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where result JSON files will be written.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N tasks (0 = all tasks).")
    parser.add_argument("--task-id", type=str, default="", help="Run only a specific task ID.")
    parser.add_argument("--verbose", action="store_true", help="Print turn-by-turn tutor/student dialogue.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from agent import TutoringAgent
        from src.simulator.student_simulator import simulate_student
    except ImportError as exc:
        raise ImportError(
            "Missing project dependencies. Activate your project environment and run: pip install -r requirements.txt"
        ) from exc

    if not args.tasks.exists():
        raise FileNotFoundError(f"Task file not found: {args.tasks}")

    tasks = [normalize_task(t) for t in load_tasks(args.tasks)]

    if args.task_id:
        tasks = [t for t in tasks if t.get("id") == args.task_id]

    if args.limit and args.limit > 0:
        tasks = tasks[: args.limit]

    if not tasks:
        raise ValueError("No tasks selected. Check --task-id/--limit settings.")

    agent = TutoringAgent()
    task_results: list[dict] = []

    for i, task in enumerate(tasks, start=1):
        print(f"[{i}/{len(tasks)}] Running task: {task.get('id')}")
        try:
            result = run_single_task(agent, task, simulate_student, verbose=args.verbose)
            task_results.append(result)
            status = "PASS" if result["passed"] else "FAIL"
            print(f"  -> {status} | transfer_answer={result['student_transfer_answer']}")
        except Exception as exc:
            error_result = {
                "task_id": task.get("id"),
                "topic": task.get("topic", "unknown"),
                "passed": False,
                "error": str(exc),
            }
            task_results.append(error_result)
            print(f"  -> ERROR | {exc}")

    valid_results = [r for r in task_results if "turns_taken" in r]
    summary = aggregate_summary(valid_results)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"benchmark_results_{run_ts}.json"

    payload = {
        "run_at": datetime.now().isoformat(),
        "task_file": str(args.tasks),
        "total_requested_tasks": len(tasks),
        "summary": summary,
        "results": task_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n=== Benchmark Summary ===")
    print(f"Accuracy: {summary['passed_tasks']}/{summary['total_tasks']} ({summary['accuracy']:.1%})")
    print(f"Avg turns: {summary['avg_turns_taken']:.2f}")
    print(f"Diagnosed rate: {summary['diagnosed_rate']:.1%}")
    print(f"Counter-example rate: {summary['counter_example_shown_rate']:.1%}")
    print(f"Confirmed correction rate: {summary['confirmed_correction_rate']:.1%}")
    print(f"Saved results to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
