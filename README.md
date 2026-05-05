# Misconception Tutoring — Benchmark + Three-Phase Agent

A class-project benchmark and tutoring agent for CS 498 (AI Agents).

**Problem.** Can an LLM-driven agent help a simulated K–12 student abandon a
specific math misconception and answer a transfer question correctly — better
than the same LLM run as a single-prompt tutor in a loop?

**Contribution.**

1. **A 20-task benchmark** of math misconceptions across fractions, negative
   numbers, algebra, and geometry, with an LLM-judged student simulator that
   resists correction until presented with semantically meaningful evidence.
2. **A three-phase tutoring agent** (probe → confront → teach → confirm) with
   an LLM tracker, a state-machine planner, and a transfer-question recap that
   reaches **96.0%** accuracy across 5 trials, vs **49.0%** for a non-agentic
   LLM baseline using the same model and turn budget.
3. **An ablation** showing the transfer recap is the load-bearing component:
   removing it drops the agent to **41.0%**, below baseline.
4. **Per-task cost: ~1¢.** The full agent costs $0.0135 per task on average
   vs $0.0100 for baseline — a 35% cost increase that buys 47 absolute points
   of accuracy.

## Repo layout

```
agent.py                       # TutoringAgent (planner + tracker + tutor LLM)
src/
├── simulator/
│   └── student_simulator.py   # student LLM + LLM-judged correction gate
└── utils/
    └── gemini_client.py       # Vertex AI / Gemini client setup
benchmark/
├── benchmark.json             # 20 task specifications
├── run_benchmark.py           # run the agent against the simulator
├── run_llm_baseline.py        # non-agentic LLM-in-loop baseline
├── run_human_baseline.py      # stdin-driven 5-task human reference run
├── summarize_human_baseline.py # summarize human reference JSONs
├── aggregate_runs.py          # mean/std accuracy across trials, per-task table
├── estimate_cost.py           # per-condition USD cost via Vertex count_tokens
├── README.md                  # benchmark protocol + task schema (Assignment 6)
└── results/                   # 15 final-experiment JSONs (3 conditions × 5 trials)
    ├── aggregated_summary.json    # canonical accuracy summary
    ├── cost_estimate.json         # canonical cost summary
    └── archive/               # exploratory / pre-final-experiment runs
archive/                       # connectivity demos and an early task draft
requirements.txt
```

## Setup

### 1. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Vertex AI credentials

This project uses **Vertex AI** (not the Gemini Developer API) because the team
account is on GCP. You need a service account JSON key with the **Vertex AI
User** role.

**Steps:**

1. In the GCP console, open *IAM & Admin → Service Accounts*, pick (or create)
   a service account, then *Keys → Add Key → JSON*. Download the key file.
2. Place the key somewhere local (e.g. project root). **Do not commit it.** The
   `.gitignore` already excludes `noble-operation-*.json` and similar names.
3. Create `.env` in the project root:

```bash
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_APPLICATION_CREDENTIALS=your-service-account.json
GOOGLE_CLOUD_LOCATION=global
# GOOGLE_CLOUD_PROJECT is optional — the service account's project_id is used
```

`src/utils/gemini_client.py` reads these env vars, prefers the service
account's `project_id` over any stale `GOOGLE_CLOUD_PROJECT` exported in your
shell, and resolves a relative `GOOGLE_APPLICATION_CREDENTIALS` from the repo
root.

### 3. Smoke test

```bash
python3 benchmark/run_benchmark.py --task-id fractions_01_add_num_den --label smoke
```

Should print one PASS line and an accuracy summary in ~1–2 minutes.

## Reproduce the experiment

The headline result is 5 trials × 3 conditions = 15 runs of 20 tasks each.

> **Run these commands one at a time, sequentially.** Launching them in
> parallel can stress the Vertex API and cause individual runs to hang. The
> per-task timeout in `run_benchmark.py` will catch hangs, but it's still
> faster (and friendlier to the API) to run sequentially.

```bash
# Full agent (5 trials)
for t in trial1 trial2 trial3 trial4 trial5; do
  python3 benchmark/run_benchmark.py --workers 5 --trial-id $t --agent-name three_phase_agent --label agent_${t#trial}
done

# Non-agentic baseline (5 trials)
for t in trial1 trial2 trial3 trial4 trial5; do
  python3 benchmark/run_llm_baseline.py --workers 5 --trial-id $t --agent-name baseline --label baseline_${t#trial}
done

# Ablation: agent without transfer-question recap (5 trials)
for t in trial1 trial2 trial3 trial4 trial5; do
  python3 benchmark/run_benchmark.py --workers 5 --trial-id $t --agent-name agent_no_recap --label no_recap_${t#trial} --no-transfer-recap
done

# Aggregate accuracy and estimate cost
python3 benchmark/aggregate_runs.py
python3 benchmark/estimate_cost.py --use-api
```

## Headline results

Accuracy (5 trials × 20 tasks each):

| Condition                         | Per-trial accuracies                          | Mean      | Std  |
|-----------------------------------|------------------------------------------------|-----------|------|
| Three-phase agent                 | 20/20, 20/20, 18/20, 18/20, 20/20             | **96.0%** | 5.5  |
| LLM-in-loop baseline              | 14/20, 8/20, 10/20, 11/20, 6/20               | 49.0%     | 15.2 |
| Agent − transfer recap (ablation) | 11/20, 6/20, 6/20, 7/20, 11/20                | 41.0%     | 12.9 |

Cost (Vertex `count_tokens` API, USD):

| Condition                         | Mean / trial | Total (5 trials) | Per task |
|-----------------------------------|--------------|------------------|----------|
| Three-phase agent                 | $0.271       | $1.353           | $0.0135  |
| LLM-in-loop baseline              | $0.200       | $1.002           | $0.0100  |
| Agent − transfer recap (ablation) | $0.278       | $1.388           | $0.0139  |
| **Total experiment**              | —            | **$3.74**        | —        |

All conditions: tutor and baseline use Gemini-3 Flash at temperature 1.0; the
student simulator and judge use Gemini-3.1 Flash-Lite. Same task set, same turn
budget (8), same evaluation grader. Differences are exactly the agent
scaffolding (planner + tracker + recap).

## Key files for graders

| File | What it is |
|---|---|
| `agent.py` | Read `TutoringAgent.run_session` for the dialogue loop and `DialoguePlanner.select_move` for the move FSM |
| `src/simulator/student_simulator.py` | `_correction_unlocked` is the LLM-judged gate; `simulate_student` runs the student LLM |
| `benchmark/benchmark.json` | All 20 task specifications |
| `benchmark/README.md` | Benchmark protocol, task schema, evaluation rules, validation, limitations |
| `benchmark/results/aggregated_summary.json` | Canonical accuracy table (mean/std + per-task matrix) — regenerate with `aggregate_runs.py` |
| `benchmark/results/cost_estimate.json` | Canonical cost table — regenerate with `estimate_cost.py --use-api` |
| `benchmark/results/agent_t*.json`, `baseline_t*.json`, `ablation_no_recap_t*.json` | Per-trial result JSONs (15 files) the headline agent numbers come from |
| `benchmark/results/human_t*.json` | Five full-20-task human reference runs used in the benchmark paper |
| `benchmark/run_human_baseline.py` | stdin-driven human reference runner; default 5-task sample, `--all-20` for full trials |

## Useful commands

```bash
# Run a specific task with verbose dialogue printing
python3 benchmark/run_benchmark.py --task-id geometry_03_bigger_area_means_bigger_perimeter --verbose

# Run only fractions tasks (first 5)
python3 benchmark/run_benchmark.py --limit 5

# Disable both ablation knobs at once (architecture-stripped)
python3 benchmark/run_benchmark.py --no-teaching-move --no-transfer-recap --label both_off

# Regenerate the headline table
python3 benchmark/aggregate_runs.py
```

## Notes

- Benchmark results write to `benchmark/results/<label>.json` with a stable
  schema (see `benchmark/README.md` for the schema).
- `archive/` contains connectivity demos (`example.py`, `vertex_example.py`)
  and an early task-format draft (`data/tasks.json`); none of them are
  imported by the live code.
- `benchmark/results/archive/` contains exploratory runs from before we settled
  on the final 5-trial × 3-condition design.

## License / disclaimer

This is a class project. The benchmark is small (20 tasks) and the simulator
is itself an LLM, so results should not be taken as evidence of general
tutoring competence; see `benchmark/README.md` for limitations.
