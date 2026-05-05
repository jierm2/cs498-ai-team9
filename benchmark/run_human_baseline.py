from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.run_benchmark import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TASK_PATH,
    _extract_final_answer,
    _task_file_for_record,
    aggregate_summary,
    grade_transfer_answer,
    load_tasks,
    normalize_task,
)
from src.simulator.student_simulator import simulate_student  # noqa: E402

DEFAULT_HUMAN_TASK_IDS = [
    "fractions_01_add_num_den",
    "negative_03_subtracting_negative_same_as_subtracting_positive",
    "algebra_03_division_distributes_over_sum_in_numerator_termwise_to_denominator",
    "geometry_03_bigger_area_means_bigger_perimeter",
    "negative_05_distribute_negative_sign_incorrectly",
]


@dataclass
class HumanSessionResult:
    task_id: str
    turns_taken: int
    transfer_question: str
    student_transfer_answer: str
    dialogue_history: list[dict] = field(default_factory=list)
    final_tracker_state: dict = field(default_factory=dict)
    tracker_parse_failures: int = 0


def _prompt_human(turn: int, max_turns: int) -> str:
    while True:
        msg = input(f"\nHuman tutor turn {turn}/{max_turns} (/done for transfer): ").strip()
        if msg:
            return msg


def run_human_task(task: dict, verbose: bool = False) -> dict:
    sim_history: list[dict] = []
    dialogue_history: list[dict] = []

    print("\n" + "=" * 88)
    print(f"Task: {task['id']} | topic={task.get('topic')} | difficulty={task.get('difficulty')}")
    print(f"Problem: {task.get('problem', '')}")
    print(f"Student starts: {task.get('student_initial_reasoning', '')}")
    print(f"Transfer question (do not solve during tutoring): {task.get('transfer_question', '')}")
    print("=" * 88)

    turns_taken = 0
    max_turns = int(task.get("max_turns", 8))
    for turn in range(1, max_turns + 1):
        tutor_message = _prompt_human(turn, max_turns)
        if tutor_message.lower() == "/done":
            break

        turns_taken += 1
        sim_history.append({"role": "user", "text": tutor_message})

        if turns_taken == 1 and task.get("student_initial_reasoning"):
            student_response = str(task["student_initial_reasoning"])
        else:
            student_response = simulate_student(task, sim_history)

        sim_history.append({"role": "model", "text": student_response})
        dialogue_history.append({"role": "tutor", "content": tutor_message})
        dialogue_history.append({"role": "student", "content": student_response})

        print(f"\nStudent: {student_response}")
        if verbose:
            print(f"[sim turns stored: {len(sim_history)}]")

    transfer_prompt = (
        "<OBJECTIVE>\n"
        "Now answer this question on your own, using what you learned in the tutoring dialogue.\n"
        "</OBJECTIVE>\n\n"
        "<CONTEXT>\n"
        f"{task['transfer_question']}\n"
        "</CONTEXT>\n\n"
        "<CONSTRAINTS>\n"
        "Give ONLY your final answer (a number, expression, or short phrase). No explanation.\n"
        "</CONSTRAINTS>"
    )
    sim_history.append({"role": "user", "text": transfer_prompt})
    student_transfer_answer = simulate_student(task, sim_history)
    sim_history.append({"role": "model", "text": student_transfer_answer})

    passed, grading_mode = grade_transfer_answer(task, student_transfer_answer)
    session = HumanSessionResult(
        task_id=task["id"],
        turns_taken=turns_taken,
        transfer_question=task["transfer_question"],
        student_transfer_answer=student_transfer_answer,
        dialogue_history=dialogue_history,
    )

    result = {
        "task_id": task.get("id"),
        "topic": task.get("topic", "unknown"),
        "difficulty": task.get("difficulty", "unknown"),
        "passed": passed,
        "grading_mode": grading_mode,
        "transfer_question": session.transfer_question,
        "correct_answer": task.get("correct_answer", ""),
        "accepted_answers": task.get("accepted_answers", []),
        "student_transfer_answer": session.student_transfer_answer,
        "student_extracted_answer": _extract_final_answer(session.student_transfer_answer),
        "turns_taken": session.turns_taken,
        "dialogue_history": session.dialogue_history,
        "tracker": {},
        "tracker_parse_failures": 0,
        "diagnosed": False,
        "counter_example_shown": False,
        "confirmed_correction": False,
    }

    status = "PASS" if passed else "FAIL"
    print(
        f"\nTransfer answer: {student_transfer_answer}\n"
        f"Extracted: {result['student_extracted_answer']} | correct: {result['correct_answer']} | {status}"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a stdin-driven human tutor baseline.")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASK_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--author", type=str, required=True, help="Human identifier, e.g. human_tutor.")
    parser.add_argument("--trial-id", type=str, default="", help="Defaults to the author id.")
    parser.add_argument("--task-id", action="append", default=[], help="Task id to run. Repeatable.")
    parser.add_argument("--all-20", action="store_true", help="Run all benchmark tasks instead of the 5-task sample.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = [normalize_task(t) for t in load_tasks(args.tasks)]
    selected_ids = [] if args.all_20 else (args.task_id or DEFAULT_HUMAN_TASK_IDS)
    if selected_ids:
        wanted = set(selected_ids)
        tasks = [t for t in tasks if t.get("id") in wanted]
        missing = wanted - {t.get("id") for t in tasks}
        if missing:
            raise ValueError(f"Unknown task id(s): {', '.join(sorted(missing))}")
    if not tasks:
        raise ValueError("No tasks selected.")

    task_results = []
    for task in tasks:
        result = run_human_task(task, verbose=args.verbose)
        result["trial_id"] = args.trial_id or args.author
        result["agent_name"] = "human_baseline"
        result["human_author"] = args.author
        task_results.append(result)

    summary = aggregate_summary(task_results)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_author = args.author.replace(" ", "_").lower()
    output_path = args.output_dir / f"human_baseline_{safe_author}_{run_ts}.json"

    payload = {
        "run_at": datetime.now().isoformat(),
        "task_file": _task_file_for_record(args.tasks),
        "total_requested_tasks": len(tasks),
        "trial_id": args.trial_id or args.author,
        "agent_name": "human_baseline",
        "human_author": args.author,
        "summary": summary,
        "results": task_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n=== Human Baseline Summary ===")
    print(f"Accuracy: {summary['passed_tasks']}/{summary['total_tasks']} ({summary['accuracy']:.1%})")
    print(f"Avg turns: {summary['avg_turns_taken']:.2f}")
    print(f"Saved results to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
