from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_TASK_PATH = PROJECT_ROOT / "benchmark" / "benchmark.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "benchmark" / "results"
DEFAULT_FALLBACK_MODEL = "gemini-3.1-pro-preview"


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


def _infer_default_agent_model() -> str:
    """Infer AGENT_MODEL from agent.py without importing runtime-heavy modules."""
    agent_path = PROJECT_ROOT / "agent.py"
    try:
        text = agent_path.read_text(encoding="utf-8")
    except OSError:
        return DEFAULT_FALLBACK_MODEL

    match = re.search(r'^\s*AGENT_MODEL\s*=\s*["\']([^"\']+)["\']\s*$', text, re.MULTILINE)
    if match:
        return match.group(1)
    return DEFAULT_FALLBACK_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a non-agentic tutoring baseline: the same LLM used by the tutor model "
            "conducts multi-turn tutoring with the student simulator (no planner/tracker)."
        )
    )
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASK_PATH, help="Path to benchmark tasks JSON.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for result JSON files.")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N tasks (0 = all).")
    parser.add_argument("--task-id", type=str, default="", help="Run only a specific task ID.")
    parser.add_argument(
        "--model",
        type=str,
        default=_infer_default_agent_model(),
        help="Model to benchmark (defaults to AGENT_MODEL found in agent.py).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        metavar="N",
        help="Concurrent tasks (default 3).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-task prompt/answer details.")
    return parser.parse_args()


def _format_history(dialogue_history: list[dict]) -> str:
    if not dialogue_history:
        return "(no turns yet)"
    lines = []
    for turn in dialogue_history:
        role = "Tutor" if turn["role"] == "tutor" else "Student"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def _generate_tutor_response(model: str, task: dict, dialogue_history: list[dict], client) -> str:
    history_text = _format_history(dialogue_history)
    prompt = f"""<OBJECTIVE>
Write ONLY your next tutor message as plain dialogue (no role labels or prefixes).
</OBJECTIVE>

<CONTEXT>
You are tutoring this misconception:
{task.get('misconception', '')}

Conversation so far:
{history_text if history_text != '(no turns yet)' else '(session just started)'}
</CONTEXT>

<CONSTRAINTS>
Keep the message concise (2–4 sentences).
Ask at least one question that advances understanding.
Do not give away the final answer directly.
Do not repeat the exact same prompt as the previous turn.
Output only the tutor's next message.
</CONSTRAINTS>"""

    response = client.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
        config={"temperature": 1.0},
    )
    return (response.text or "").strip() or "Can you explain your reasoning step by step?"


def run_single_task(model: str, task: dict, client, verbose: bool = False) -> dict:
    from benchmark.run_benchmark import grade_transfer_answer, _extract_final_answer
    from src.simulator.student_simulator import simulate_student

    sim_history: list[dict] = []
    dialogue_history: list[dict] = []
    initial_reasoning_used = False
    max_turns = int(task.get("max_turns", 8))

    def get_student_response(prompt_text: str) -> str:
        nonlocal initial_reasoning_used

        sim_history.append({"role": "user", "text": prompt_text})
        is_transfer = "answer the following question" in prompt_text.lower() or "transfer" in prompt_text.lower()

        if not initial_reasoning_used and task.get("student_initial_reasoning") and not is_transfer:
            student_response = str(task["student_initial_reasoning"])
            initial_reasoning_used = True
        else:
            student_response = simulate_student(task, sim_history)

        sim_history.append({"role": "model", "text": student_response})
        return student_response

    for turn_num in range(1, max_turns + 1):
        tutor_message = _generate_tutor_response(model, task, dialogue_history, client)
        if verbose:
            print(f"\n[Turn {turn_num}/{max_turns}]")
            print(f"Tutor: {tutor_message}")

        student_response = get_student_response(tutor_message)
        if verbose:
            print(f"Student: {student_response}")

        dialogue_history.append({"role": "tutor", "content": tutor_message})
        dialogue_history.append({"role": "student", "content": student_response})

    transfer_prompt = (
        "<OBJECTIVE>\n"
        "Answer the following question on your own.\n"
        "</OBJECTIVE>\n\n"
        "<CONTEXT>\n"
        f"{task.get('transfer_question', '')}\n"
        "</CONTEXT>\n\n"
        "<CONSTRAINTS>\n"
        "Give ONLY your final answer (a number, expression, or short phrase). No explanation.\n"
        "</CONSTRAINTS>"
    )

    if verbose:
        print(f"\n[Transfer Test] {task.get('transfer_question', '')}")

    llm_answer = get_student_response(transfer_prompt).strip()

    passed, grading_mode = grade_transfer_answer(task, llm_answer)

    if verbose:
        print(f"\nTask: {task.get('id')} | model={model}")
        print(f"Student transfer answer: {llm_answer}")

    return {
        "task_id": task.get("id"),
        "topic": task.get("topic", "unknown"),
        "passed": passed,
        "grading_mode": grading_mode,
        "transfer_question": task.get("transfer_question", ""),
        "correct_answer": task.get("correct_answer", ""),
        "accepted_answers": task.get("accepted_answers", []),
        "llm_answer": llm_answer,
        "llm_extracted_answer": _extract_final_answer(llm_answer),
        "turns_taken": max_turns,
        "dialogue_history": dialogue_history,
        "model": model,
    }


def aggregate_summary(task_results: list[dict]) -> dict:
    total = len(task_results)
    passed = sum(1 for r in task_results if r.get("passed"))
    accuracy = (passed / total) if total else 0.0

    by_topic: dict[str, dict] = {}
    for result in task_results:
        topic = result.get("topic", "unknown")
        topic_bucket = by_topic.setdefault(topic, {"total": 0, "passed": 0})
        topic_bucket["total"] += 1
        if result.get("passed"):
            topic_bucket["passed"] += 1

    for values in by_topic.values():
        values["accuracy"] = values["passed"] / values["total"] if values["total"] else 0.0

    turns = [r["turns_taken"] for r in task_results if "turns_taken" in r]

    return {
        "total_tasks": total,
        "passed_tasks": passed,
        "accuracy": accuracy,
        "avg_turns_taken": mean(turns) if turns else 0.0,
        "by_topic": by_topic,
    }


def main() -> int:
    args = parse_args()

    from benchmark.run_benchmark import load_tasks, normalize_task
    from src.utils.gemini_client import get_client

    if not args.tasks.exists():
        raise FileNotFoundError(f"Task file not found: {args.tasks}")

    tasks = [normalize_task(t) for t in load_tasks(args.tasks)]

    if args.task_id:
        tasks = [t for t in tasks if t.get("id") == args.task_id]
    if args.limit and args.limit > 0:
        tasks = tasks[: args.limit]
    if not tasks:
        raise ValueError("No tasks selected. Check --task-id/--limit settings.")

    client = get_client()
    workers = max(1, min(args.workers, len(tasks)))

    print(f"Running LLM tutoring baseline on {len(tasks)} task(s) with up to {workers} worker(s). Model: {args.model}")

    task_results: list[dict] = []

    def run_task_wrapper(task: dict) -> dict:
        return run_single_task(args.model, task, client, verbose=args.verbose)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_task_wrapper, t): t for t in tasks}
        for future in as_completed(futures):
            task = futures[future]
            tid = task.get("id")
            try:
                result = future.result()
                task_results.append(result)
                status = "PASS" if result["passed"] else "FAIL"
                print(
                    f"  -> {status} ({result['grading_mode']}) | {tid} | "
                    f"extracted=\"{result['llm_extracted_answer']}\" | "
                    f"correct=\"{result['correct_answer']}\""
                )
            except Exception as exc:
                error_result = {
                    "task_id": tid,
                    "topic": task.get("topic", "unknown"),
                    "passed": False,
                    "error": str(exc),
                    "model": args.model,
                }
                task_results.append(error_result)
                print(f"  -> ERROR | {tid} | {exc}")

    valid_results = [r for r in task_results if "llm_answer" in r]
    summary = aggregate_summary(valid_results)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"llm_baseline_results_{run_ts}.json"

    payload = {
        "run_at": datetime.now().isoformat(),
        "benchmark_type": "llm_direct_baseline",
        "task_file": _task_file_for_record(args.tasks),
        "total_requested_tasks": len(tasks),
        "model": args.model,
        "summary": summary,
        "results": task_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n=== LLM Baseline Summary ===")
    print(f"Accuracy: {summary['passed_tasks']}/{summary['total_tasks']} ({summary['accuracy']:.1%})")
    print(f"Avg turns: {summary['avg_turns_taken']:.2f}")
    print(f"Saved results to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
