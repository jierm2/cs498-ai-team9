# Misconception-Tutoring Benchmark

A benchmark for evaluating tutoring agents on K–12 math misconceptions. Each task
defines a *misconception* the simulated student holds and a *transfer question*
the student must answer correctly *after* dialogue. An agent passes a task only
if the student gives the correct transfer answer at the end of the session.

## What this benchmark measures

The agent's job is to drive a multi-turn dialogue that helps a simulated student
abandon a specific math misconception and answer a transfer question correctly.
This is harder than single-shot Q&A because:

- The simulated student is *committed* to the misconception until shown
  pedagogically meaningful evidence (counter-examples + invitations to reconsider).
- The agent must produce a *transferable* method, not just walk through the
  exact transfer problem.
- The session has a limited turn budget (default 8 turns).

## Task suite

20 tasks across four topics, 5 each:

- **Fractions** — adding numerators and denominators separately, bigger denominator
  means bigger fraction, etc.
- **Negative numbers** — `-x · -y < 0`, ordering by absolute value, etc.
- **Algebra** — `(x+a)² = x² + a²`, combining unlike terms, etc.
- **Geometry** — perimeter/area confusion, diameter/radius confusion, etc.

Full task list: `benchmark.json`.

## Task schema

Each task is a JSON object with these fields:

| Field | Type | Purpose |
|---|---|---|
| `id` | string | unique task identifier |
| `topic` | string | one of fractions / negative_numbers / algebra / geometry |
| `difficulty` | string | informal label: easy / medium / hard |
| `misconception_label` | string | machine-readable name of the error |
| `misconception` | string | natural-language description shown to the simulator |
| `correction_label` | string | machine-readable name of the correct rule |
| `problem` | string | the situation the tutor is facing |
| `student_initial_reasoning` | string | the student's first turn in the dialogue |
| `student_belief_rules` | list[string] | extra constraints on the simulator's persona |
| `allowed_error_patterns` | list[string] | mistake patterns the simulator may exhibit |
| `transfer_question` | string | post-dialogue assessment problem |
| `correct_answer` | string | canonical correct answer |
| `accepted_answers` | list[string] | grader-accepted variants |
| `grading_rule` | string | human-readable grading note |
| `max_turns` | int | dialogue budget (default 8) |

## Evaluation protocol

1. **Setup.** Load a task. Initialize the agent and a fresh student simulator.
2. **Dialogue.** Run up to `max_turns` tutor↔student exchanges. The agent ends
   the session early when its tracker reports correction confirmed.
3. **Transfer test.** The simulator answers the `transfer_question` once, with
   the rest of the dialogue history visible to it. The agent has no chance to
   intervene during the transfer test.
4. **Grading.** The student's answer is compared against `accepted_answers`
   using strict matching:
   - Whitespace and unicode normalization
   - Exact text match, *or*
   - Numeric equivalence (both sides reduce to the same `Fraction`) — only when
     both sides parse as a single number/fraction (no algebraic variables)
   - **No substring matching** — `"x²+4"` is not graded equal to `"x²+4x+4"`
5. **Score.** Accuracy = `tasks_passed / tasks_total`. Report mean ± std across
   ≥3 trials because the simulator runs at temperature 1.0 and is stochastic.

## Student simulator design

The simulator uses an LLM (Gemini-3.1-Flash-Lite) and a *correction gate*.
While the gate is **locked**, the simulator stays in character and reasserts
the misconception. The gate **unlocks** when an LLM judge (also Flash-Lite)
counts enough pedagogical evidence in the dialogue:

- ≥4 tutor turns containing concrete counter-examples
- ≥3 tutor turns inviting reconsideration
- ≥2 student turns showing genuine doubt or self-correction
- ≥5 tutor turns total

The judge replaces a previous regex-based gate. Regex matching was vulnerable
to prompt-tuning leakage (a tutor that emitted phrases like "let's test" trip
the gate without doing real teaching). The LLM judge scores semantic intent,
which closes that loophole.

## How to use this benchmark

### Quick start

```bash
# Run a single task (smoke test)
python benchmark/run_benchmark.py --task-id fractions_01_add_num_den --label smoke

# Run all 20 tasks (one trial)
python benchmark/run_benchmark.py --workers 5 --trial-id trial1 --agent-name my_agent

# Run the non-agentic LLM baseline
python benchmark/run_llm_baseline.py --workers 5 --trial-id trial1 --agent-name baseline

# Run the stdin-driven human baseline
python benchmark/run_human_baseline.py --author human_tutor        # default 5-task sample
python benchmark/run_human_baseline.py --author tutor --all-20 --trial-id trial1
python benchmark/summarize_human_baseline.py

# Aggregate all results in benchmark/results/
python benchmark/aggregate_runs.py
```

### Plugging in your own agent

The runner expects an agent class with a `run_session(task, get_student_response, verbose)`
method that returns a `SessionResult` (see `agent.py`). Replace the import in
`benchmark/run_benchmark.py`:

```python
from agent import TutoringAgent          # default
# or
from my_agent import MyTutoringAgent     # your replacement
```

Your agent must:

- Take a task dict and produce per-turn tutor messages
- Call `get_student_response(tutor_message)` to get the simulator's reply
- Stop after at most `task["max_turns"]` turns
- Submit a final transfer-question answer via the same `get_student_response`

### Result file format

Each run writes to `benchmark/results/<label>.json`:

```json
{
  "run_at": "...",
  "trial_id": "trial1",
  "agent_name": "three_phase_agent",
  "task_file": "benchmark/benchmark.json",
  "summary": {
    "total_tasks": 20,
    "passed_tasks": 20,
    "accuracy": 1.0,
    "avg_turns_taken": 7.35,
    "diagnosed_rate": 1.0,
    "counter_example_shown_rate": 1.0,
    "confirmed_correction_rate": 0.55,
    "by_topic": { "fractions": {"total": 5, "passed": 5, "accuracy": 1.0}, ... }
  },
  "results": [
    {
      "task_id": "fractions_01_add_num_den",
      "topic": "fractions",
      "passed": true,
      "trial_id": "trial1",
      "agent_name": "three_phase_agent",
      "transfer_question": "...",
      "correct_answer": "7/12",
      "student_transfer_answer": "7/12",
      "turns_taken": 8,
      "dialogue_history": [...]
    }
  ]
}
```

The `trial_id` and `agent_name` fields exist on both the top-level payload and
each per-task result, so a single concatenation of multiple result JSONs is
enough to reconstruct any breakdown.

## Validation

Three forms of validation are reported in the benchmark paper:

1. **Agent baseline.** A non-agentic LLM-in-loop tutor (same model, same temperature,
   same turn budget) reaches 49.0% mean accuracy across 5 trials, well below the
   three-phase agent's 96.0%. The benchmark distinguishes weak from strong tutors.
2. **Ablation study.** Removing the agent's transfer-question recap drops mean
   accuracy from 96.0% to 41.0% (below baseline). The benchmark is sensitive to
   architectural changes, not just to model strength.
3. **Cross-trial consistency.** The agent's standard deviation across trials
   (5.5 pp) is much lower than the baseline's (15.2 pp). The benchmark is
   reliable enough to detect a real architectural advantage despite simulator
   stochasticity.
4. **Human reference.** The final paper reports five full-20-task human runs
   (`human_t1.json`--`human_t5.json`) against the same simulator and automatic
   transfer grader. `run_human_baseline.py` also supports a default five-task
   stratified sample for quick manual checks. `summarize_human_baseline.py`
   reports the human score for the benchmark paper.

## Reproducibility

- All 15 final-experiment result JSONs are checked in under `benchmark/results/`
  for the three automated conditions.
- `benchmark/results/archive/` contains exploratory runs (different model
  configs, the regex-gated pre-LLM-judge simulator, ablation matrix).
- `aggregate_runs.py` regenerates the headline table from the result files.
- Vertex AI is the only external dependency; no model fine-tuning is involved.

## Limitations and disclosures

- The benchmark's notion of "tutoring success" is *correct transfer answer*,
  not *durable understanding*. A student that copies a worked example without
  understanding the rule still passes.
- The simulator is itself an LLM and may produce student responses no real
  K–12 student would. We do not claim the simulator is ecologically valid; we
  claim it is consistent enough to discriminate between agents.
- The LLM-judge gate replaces regex matching but introduces its own failure
  modes (judge LLM may hallucinate or be inconsistent). The gate falls back to
  regex on judge timeout/failure to stay robust.
- 20 tasks is small. The benchmark is intended as a class project demonstration,
  not a production-scale evaluation suite.
