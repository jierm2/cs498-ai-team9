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
   reaches **96.7%** accuracy across 3 trials, vs **53.3%** for a non-agentic
   LLM baseline using the same model and turn budget.
3. **An ablation** showing the transfer recap is the load-bearing component:
   removing it drops the agent to **38.3%**, below baseline.

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
├── aggregate_runs.py          # mean/std accuracy across trials, per-task table
├── README.md                  # benchmark protocol + task schema (Assignment 6)
└── results/                   # 9 final-experiment JSONs (3 conditions × 3 trials)
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

The headline result is 3 trials × 3 conditions = 9 runs of 20 tasks each.

> **Run these commands one at a time, sequentially.** Launching them in
> parallel can stress the Vertex API and cause individual runs to hang. The
> per-task timeout in `run_benchmark.py` will catch hangs, but it's still
> faster (and friendlier to the API) to run sequentially.

```bash
# Full agent (3 trials)
python3 benchmark/run_benchmark.py --workers 5 --trial-id trial1 --agent-name three_phase_agent --label agent_t1
python3 benchmark/run_benchmark.py --workers 5 --trial-id trial2 --agent-name three_phase_agent --label agent_t2
python3 benchmark/run_benchmark.py --workers 5 --trial-id trial3 --agent-name three_phase_agent --label agent_t3

# Non-agentic baseline (3 trials)
python3 benchmark/run_llm_baseline.py --workers 5 --trial-id trial1 --agent-name baseline --label baseline_t1
python3 benchmark/run_llm_baseline.py --workers 5 --trial-id trial2 --agent-name baseline --label baseline_t2
python3 benchmark/run_llm_baseline.py --workers 5 --trial-id trial3 --agent-name baseline --label baseline_t3

# Ablation: agent without the transfer-question recap (3 trials)
python3 benchmark/run_benchmark.py --workers 5 --trial-id trial1 --agent-name agent_no_recap --label no_recap_t1 --no-transfer-recap
python3 benchmark/run_benchmark.py --workers 5 --trial-id trial2 --agent-name agent_no_recap --label no_recap_t2 --no-transfer-recap
python3 benchmark/run_benchmark.py --workers 5 --trial-id trial3 --agent-name agent_no_recap --label no_recap_t3 --no-transfer-recap

# Aggregate (reads everything in benchmark/results/)
python3 benchmark/aggregate_runs.py
```

## Headline results

| Condition                         | Trials              | Mean accuracy | Std  |
|-----------------------------------|---------------------|---------------|------|
| Three-phase agent                 | 20/20, 20/20, 18/20 | **96.7%**     | 5.8  |
| LLM-in-loop baseline              | 14/20, 8/20, 10/20  | 53.3%         | 15.3 |
| Agent − transfer recap (ablation) | 11/20, 6/20, 6/20   | 38.3%         | 14.4 |

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
| `benchmark/results/` | The 9 final-experiment result JSONs the headline numbers come from |

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
  on the final 3-trial × 3-condition design.

## License / disclaimer

This is a class project. The benchmark is small (20 tasks) and the simulator
is itself an LLM, so results should not be taken as evidence of general
tutoring competence; see `benchmark/README.md` for limitations.
