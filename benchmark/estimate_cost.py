"""Estimate Vertex AI cost per condition based on existing result JSONs.

Uses the official Vertex `count_tokens` API to count tokens for each
turn-level message we have in `dialogue_history`, then multiplies by Vertex
Gemini pricing. Counts:

- Tutor LLM output (every tutor turn)
- Student/simulator LLM output (every student turn + transfer answer)
- Tracker LLM output (estimated 1 call per tutor turn for agent runs)
- Judge LLM output (estimated 1 call per simulator turn for agent + ablation,
  also for baseline since simulator runs the judge)

Input tokens are counted only for the messages we have. Tutor and simulator
*system prompts* are short and applied identically across conditions, so we
add a fixed per-call overhead estimate. Tracker/judge prompts include the full
running dialogue history, which we approximate by summing all prior dialogue
tokens at each call point.

Pricing reference (Vertex AI Gemini standard pricing, May 2026):
- Gemini 3 Flash Preview: $0.50 / 1M text input, $3.00 / 1M text output
- Gemini 3.1 Flash-Lite Preview: $0.25 / 1M text input, $1.50 / 1M text output

Usage:
    python benchmark/estimate_cost.py
    python benchmark/estimate_cost.py --use-api    # use count_tokens (slower, accurate)
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"

# Vertex AI Gemini pricing (USD per million tokens)
PRICING = {
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
    "gemini-3.1-flash-lite-preview": {"input": 0.25, "output": 1.50},
    "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},  # for reference, <=200K input tokens
}

TUTOR_MODEL = "gemini-3-flash-preview"
TRACKER_MODEL = "gemini-3-flash-preview"
SIMULATOR_MODEL = "gemini-3.1-flash-lite-preview"
JUDGE_MODEL = "gemini-3.1-flash-lite-preview"

# Fixed prompt overhead (system prompt + scaffolding, per call)
TUTOR_SYSTEM_OVERHEAD = 200      # tutor system prompt + move instructions
SIMULATOR_SYSTEM_OVERHEAD = 350  # simulator system prompt + tactic + belief lines
TRACKER_PROMPT_OVERHEAD = 300    # tracker schema + instructions
JUDGE_PROMPT_OVERHEAD = 400      # judge schema + instructions


def heuristic_token_count(text: str) -> int:
    """Approximate token count: ~4 chars per token for English."""
    if not text:
        return 0
    return max(1, len(text) // 4)


_token_counter = None


def api_token_count(text: str, model: str) -> int:
    """Use Vertex count_tokens API. Lazy-loads client to avoid import cost."""
    global _token_counter
    if _token_counter is None:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.utils.gemini_client import get_client
        _token_counter = get_client()
    if not text:
        return 0
    try:
        resp = _token_counter.models._models.count_tokens(
            model=model,
            contents=[{"role": "user", "parts": [{"text": text}]}],
        )
        return int(resp.total_tokens)
    except Exception:
        return heuristic_token_count(text)


def estimate_run_cost(run_path: Path, count_fn) -> dict:
    """Walk one run's dialogue_history and tally tokens × price per role."""
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    agent_name = payload.get("agent_name", "unknown")
    is_baseline = agent_name == "baseline"
    is_no_recap = agent_name == "agent_no_recap"

    # Per-call counters
    counts = {
        "tutor_input_tok": 0,
        "tutor_output_tok": 0,
        "tracker_input_tok": 0,
        "tracker_output_tok": 0,
        "simulator_input_tok": 0,
        "simulator_output_tok": 0,
        "judge_input_tok": 0,
        "judge_output_tok": 0,
        "n_tutor_calls": 0,
        "n_tracker_calls": 0,
        "n_simulator_calls": 0,
        "n_judge_calls": 0,
    }

    for task_result in payload.get("results", []):
        history = task_result.get("dialogue_history", [])
        running_history_tokens = 0  # accumulates as dialogue grows

        # Per-turn cost. The dialogue_history alternates tutor / student.
        for i, turn in enumerate(history):
            role = turn.get("role")
            content = turn.get("content", "")
            tok_out = count_fn(content, TUTOR_MODEL if role == "tutor" else SIMULATOR_MODEL)

            if role == "tutor":
                # Tutor call: input = system + history-so-far, output = this turn
                counts["tutor_input_tok"] += TUTOR_SYSTEM_OVERHEAD + running_history_tokens
                counts["tutor_output_tok"] += tok_out
                counts["n_tutor_calls"] += 1
                # Tracker runs after tutor turn (only for agent / agent_no_recap, not baseline)
                if not is_baseline:
                    counts["tracker_input_tok"] += TRACKER_PROMPT_OVERHEAD + running_history_tokens + tok_out
                    counts["tracker_output_tok"] += 60  # short JSON output
                    counts["n_tracker_calls"] += 1
            elif role == "student":
                # Simulator call: input = system + history-so-far, output = this turn
                counts["simulator_input_tok"] += SIMULATOR_SYSTEM_OVERHEAD + running_history_tokens
                counts["simulator_output_tok"] += tok_out
                counts["n_simulator_calls"] += 1
                # Judge call: input = full history-so-far, output = short JSON
                counts["judge_input_tok"] += JUDGE_PROMPT_OVERHEAD + running_history_tokens + tok_out
                counts["judge_output_tok"] += 40
                counts["n_judge_calls"] += 1

            running_history_tokens += tok_out

        # Transfer test = one extra simulator call. Input = full history + transfer prompt; output = answer.
        transfer_answer = task_result.get("student_transfer_answer") or task_result.get("llm_answer", "")
        if transfer_answer:
            counts["simulator_input_tok"] += SIMULATOR_SYSTEM_OVERHEAD + running_history_tokens + 100
            counts["simulator_output_tok"] += count_fn(transfer_answer, SIMULATOR_MODEL)
            counts["n_simulator_calls"] += 1

    # Compute costs
    flash = PRICING[TUTOR_MODEL]
    flash_lite = PRICING[SIMULATOR_MODEL]
    cost_usd = (
        counts["tutor_input_tok"] * flash["input"] / 1e6
        + counts["tutor_output_tok"] * flash["output"] / 1e6
        + counts["tracker_input_tok"] * flash["input"] / 1e6
        + counts["tracker_output_tok"] * flash["output"] / 1e6
        + counts["simulator_input_tok"] * flash_lite["input"] / 1e6
        + counts["simulator_output_tok"] * flash_lite["output"] / 1e6
        + counts["judge_input_tok"] * flash_lite["input"] / 1e6
        + counts["judge_output_tok"] * flash_lite["output"] / 1e6
    )

    return {
        "file": run_path.name,
        "agent_name": agent_name,
        "trial_id": payload.get("trial_id", ""),
        "n_tasks": len(payload.get("results", [])),
        "counts": counts,
        "estimated_cost_usd": round(cost_usd, 4),
        "cost_per_task_usd": round(cost_usd / max(1, len(payload.get("results", []))), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--use-api", action="store_true",
                        help="Use Vertex count_tokens API for accurate counts (slower, costs ~$0.10).")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "cost_estimate.json")
    args = parser.parse_args()

    count_fn = api_token_count if args.use_api else (lambda text, model: heuristic_token_count(text))

    files = sorted(p for p in args.results_dir.glob("*.json")
                   if p.name not in {"aggregated_summary.json", "cost_estimate.json"}
                   and not p.is_dir())

    per_run = []
    for path in files:
        try:
            per_run.append(estimate_run_cost(path, count_fn))
        except Exception as e:
            print(f"  skipped {path.name}: {e}")

    # Group by condition
    by_condition: dict[str, list[float]] = defaultdict(list)
    for run in per_run:
        by_condition[run["agent_name"]].append(run["estimated_cost_usd"])

    # Print summary
    print("=" * 78)
    print(f"COST ESTIMATE  ({'count_tokens API' if args.use_api else 'heuristic 4 chars/token'})")
    print("=" * 78)
    print(f"{'condition':<25} {'trials':>7} {'mean / trial':>14} {'total':>10} {'per task':>10}")
    print("-" * 78)
    for cond, costs in sorted(by_condition.items()):
        mean = statistics.mean(costs)
        total = sum(costs)
        per_task = mean / 20
        print(f"{cond:<25} {len(costs):>7} {f'${mean:.4f}':>14} {f'${total:.4f}':>10} {f'${per_task:.4f}':>10}")
    grand_total = sum(r["estimated_cost_usd"] for r in per_run)
    print("-" * 78)
    print(f"{'TOTAL':<25} {len(per_run):>7} {'':>14} {f'${grand_total:.4f}':>10}")
    print()

    payload = {
        "method": "vertex_count_tokens_api" if args.use_api else "heuristic_chars_div_4",
        "pricing_usd_per_million": {
            "gemini-3-flash-preview": PRICING[TUTOR_MODEL],
            "gemini-3.1-flash-lite-preview": PRICING[SIMULATOR_MODEL],
        },
        "by_condition": {
            cond: {
                "n_trials": len(costs),
                "mean_cost_per_trial_usd": round(statistics.mean(costs), 4),
                "total_cost_usd": round(sum(costs), 4),
                "mean_cost_per_task_usd": round(statistics.mean(costs) / 20, 4),
            }
            for cond, costs in by_condition.items()
        },
        "grand_total_usd": round(grand_total, 4),
        "per_run": per_run,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote cost estimate to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
