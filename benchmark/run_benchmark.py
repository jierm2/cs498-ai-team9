from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_TASK_PATH = PROJECT_ROOT / "benchmark" / "benchmark.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmark" / "results"


def _task_file_for_record(tasks_path: Path) -> str:
    """Store task file as a path relative to repo root when possible."""
    resolved = tasks_path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        try:
            return resolved.relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return resolved.as_posix()


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


def _extract_final_answer(text: str) -> str:
    """Extract the student's final answer from potentially verbose responses.
    Looks for explicit answer patterns first, then falls back to the last
    numeric/math expression found."""
    text = text.strip()

    # Pattern 1: "my answer is X", "the answer is X", "I get X", "= X"
    answer_patterns = [
        r"(?:my |the |final )?answer\s*(?:is|:)\s*(.+?)(?:\.|$)",
        r"(?:i get|i got|that gives|result is|equals?)\s*(.+?)(?:\.|$)",
        r"=\s*(.+?)(?:\.|$)",
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip().rstrip(".!?,")
            if candidate:
                return candidate

    # Pattern 2: if the whole response is short (< 30 chars), treat it as the answer
    if len(text) < 30:
        return text

    # Pattern 3: extract the last numeric/fraction expression
    # Search from the end of the text
    all_nums = list(re.finditer(r"-?\d+(?:\s+\d+/\d+|\.\d+|/\d+)?", text))
    if all_nums:
        return all_nums[-1].group(0)

    # Fallback: return last line
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else text


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


def _is_pure_numeric(text: str) -> bool:
    """True if `text` is a single number/fraction with no algebraic variables,
    word qualifiers, or extra terms. Used to gate numeric-equivalence grading
    so that 'x^2 + 4' isn't graded equal to 'x^2 + 4x + 4'."""
    if not text:
        return False
    cleaned = text.strip().lower().replace("\u2212", "-")
    # Strip a single trailing unit word like "feet", "units", "cubic units"
    cleaned = re.sub(r"\s*(cubic\s+units?|units?|feet|ft|meters?|m|cm|inches?|in|degrees?|\u00b0)\s*$", "", cleaned)
    cleaned = cleaned.strip()
    # Allowed: optional sign, digits, optional decimal, optional fraction or mixed
    pure_patterns = [
        r"^-?\d+\s+\d+/\d+$",       # mixed: "1 2/3"
        r"^-?\d+/\d+$",              # fraction: "7/12"
        r"^-?\d+(?:\.\d+)?$",         # int/decimal: "12", "-3", "1.5"
    ]
    return any(re.fullmatch(p, cleaned) for p in pure_patterns)


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
    """Grade the student's transfer answer against accepted answers.
    Uses strict matching: exact text match or numeric equivalence only.
    NO substring matching — this was causing false positives."""

    accepted = task.get("accepted_answers") or [task.get("correct_answer", "")]
    accepted = [str(a) for a in accepted if str(a).strip()]

    # Step 1: extract the core answer from potentially verbose response
    extracted = _extract_final_answer(student_answer)
    extracted_norm = _normalize_text(extracted)
    full_norm = _normalize_text(student_answer)

    accepted_norm = [_normalize_text(a) for a in accepted]

    # Check exact match on extracted answer or full text
    if extracted_norm in accepted_norm or full_norm in accepted_norm:
        return True, "exact"

    # Numeric/fraction equivalence: ONLY apply when both the student's extracted
    # answer AND the candidate are purely numeric (no algebraic variables or
    # extra terms). Otherwise "x^2 + 4" would grade equal to "x^2 + 4x + 4"
    # because _extract_numeric pulls just the first number it finds.
    if _is_pure_numeric(extracted):
        answer_num = _extract_numeric(extracted)
        answer_frac = _to_fraction(answer_num) if answer_num else None
        if answer_frac is not None:
            for candidate in accepted:
                if not _is_pure_numeric(candidate):
                    continue
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
        is_transfer = "now answer this question" in prompt_text.lower()

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
        "student_extracted_answer": _extract_final_answer(session.student_transfer_answer),
        "turns_taken": session.turns_taken,
        "dialogue_history": session.dialogue_history,
        "tracker": final_state,
        "tracker_parse_failures": getattr(session, "tracker_parse_failures", 0),
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
    total_parse_failures = sum(r.get("tracker_parse_failures", 0) for r in task_results)

    return {
        "total_tasks": total,
        "passed_tasks": passed,
        "accuracy": accuracy,
        "avg_turns_taken": mean(turns) if turns else 0.0,
        "diagnosed_rate": diagnosed_rate,
        "counter_example_shown_rate": counter_example_rate,
        "confirmed_correction_rate": confirmed_rate,
        "total_tracker_parse_failures": total_parse_failures,
        "by_topic": by_topic,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tutoring-agent benchmark against the student simulator.")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASK_PATH, help="Path to benchmark tasks JSON.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for result JSON files.")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N tasks (0 = all).")
    parser.add_argument("--task-id", type=str, default="", help="Run only a specific task ID.")
    parser.add_argument("--verbose", action="store_true", help="Print turn-by-turn dialogue.")
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        metavar="N",
        help="Concurrent tasks (Vertex QPM ~60–300; default 3 to limit 429s).",
    )
    parser.add_argument(
        "--no-teaching-move",
        action="store_true",
        help="Ablation: disable the teaching move in the planner.",
    )
    parser.add_argument(
        "--no-transfer-recap",
        action="store_true",
        help="Ablation: disable the transfer-question recap (cold-start transfer prompt).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Optional label appended to the result filename (e.g. 'ablation_no_teaching').",
    )
    parser.add_argument(
        "--trial-id",
        type=str,
        default="",
        help="Trial identifier (e.g. 'trial1') stamped on every result for cross-trial aggregation.",
    )
    parser.add_argument(
        "--agent-name",
        type=str,
        default="three_phase",
        help="Agent identifier stamped on every result (e.g. 'three_phase', 'ablation_no_teaching').",
    )
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

    agent = TutoringAgent(
        enable_teaching_move=not args.no_teaching_move,
        enable_transfer_recap=not args.no_transfer_recap,
    )
    task_results: list[dict] = []

    workers = max(1, min(args.workers, len(tasks)))
    print(f"Running {len(tasks)} task(s) with up to {workers} concurrent worker(s) (Vertex QPM limits apply).")

    def run_task_wrapper(task: dict) -> dict:
        result = run_single_task(agent, task, simulate_student, verbose=args.verbose)
        result["trial_id"] = args.trial_id
        result["agent_name"] = args.agent_name
        return result

    PER_TASK_TIMEOUT_S = 300  # 5 min hard cap per task; record ERROR and continue

    pool = ThreadPoolExecutor(max_workers=workers)
    futures = {pool.submit(run_task_wrapper, t): t for t in tasks}
    pending = set(futures.keys())
    while pending:
        done, pending = concurrent.futures.wait(
            pending, timeout=PER_TASK_TIMEOUT_S,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        if not done:
            # Nothing finished in the last window — declare oldest pending as hung.
            stuck = next(iter(pending))
            stuck_task = futures[stuck]
            tid = stuck_task.get("id")
            error_result = {
                "task_id": tid,
                "topic": stuck_task.get("topic", "unknown"),
                "passed": False,
                "error": f"task hung past {PER_TASK_TIMEOUT_S}s timeout",
                "trial_id": args.trial_id,
                "agent_name": args.agent_name,
            }
            task_results.append(error_result)
            print(f"  -> TIMEOUT | {tid} | exceeded {PER_TASK_TIMEOUT_S}s", flush=True)
            stuck.cancel()
            pending.discard(stuck)
            continue
        for future in done:
            task = futures[future]
            tid = task.get("id")
            try:
                result = future.result(timeout=1)
                task_results.append(result)
                status = "PASS" if result["passed"] else "FAIL"
                print(
                    f"  -> {status} ({result['grading_mode']}) | {tid} | "
                    f"extracted=\"{result['student_extracted_answer']}\" | "
                    f"correct=\"{result['correct_answer']}\"",
                    flush=True,
                )
            except Exception as exc:
                error_result = {
                    "task_id": tid,
                    "topic": task.get("topic", "unknown"),
                    "passed": False,
                    "error": str(exc),
                    "trial_id": args.trial_id,
                    "agent_name": args.agent_name,
                }
                task_results.append(error_result)
                print(f"  -> ERROR | {tid} | {exc}", flush=True)
    pool.shutdown(wait=False, cancel_futures=True)

    valid_results = [r for r in task_results if "turns_taken" in r]
    summary = aggregate_summary(valid_results)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    label_suffix = f"_{args.label}" if args.label else ""
    output_path = args.output_dir / f"benchmark_results_{run_ts}{label_suffix}.json"

    payload = {
        "run_at": datetime.now().isoformat(),
        "task_file": _task_file_for_record(args.tasks),
        "total_requested_tasks": len(tasks),
        "trial_id": args.trial_id,
        "agent_name": args.agent_name,
        "agent_config": {
            "enable_teaching_move": not args.no_teaching_move,
            "enable_transfer_recap": not args.no_transfer_recap,
            "label": args.label,
        },
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
    print(f"Tracker parse failures: {summary['total_tracker_parse_failures']}")
    print(f"Saved results to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())