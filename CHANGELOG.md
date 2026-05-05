# Draft-to-Final Changelog

This file summarizes the changes made for the final CS 498 AI Agents submissions.
The same information is also summarized inside the final agent and benchmark
papers.

## Draft feedback addressed

- Replaced the earlier single-run result with five independent trials over all
  20 tasks, reporting mean accuracy and standard deviation for the structured
  agent, same-model single-prompt baseline, and recap-disabled ablation.
- Added token-level cost estimation using Vertex `count_tokens` and published
  input/output token prices, reported per tutoring session for each condition.

## Agent paper

- Completed five independent trials of all 20 benchmark tasks for the structured
  agent, the same-model single-prompt baseline, and the recap-disabled ablation.
- Added the recap-disabled ablation to isolate the effect of the transfer-recap
  step.
- Reported mean accuracy, standard deviation, average turns, tracker diagnostic
  rates, and estimated per-task cost.
- Corrected the model attribution: tutor generation and tracking use Gemini 3
  Flash Preview; the student simulator and correction-unlock judge use Gemini
  3.1 Flash-Lite Preview.
- Replaced the earlier regex correction gate with an LLM correction-unlock judge.
- Moved per-task pass rates and turns-to-correction to the appendix to keep the
  main paper within the required length.

## Benchmark paper

- Expanded validation from one agent run to five independent trials for each
  main condition: structured agent, same-model single-prompt baseline, and
  recap-disabled ablation.
- Added a same-model single-prompt baseline to show that the benchmark is
  non-trivial for an unstructured tutor using the same tutor model.
- Added per-topic and per-task result reporting, including turns-to-correction,
  to make task difficulty and TA feedback response explicit.
- Replaced substring grading with answer extraction followed by exact or numeric
  equivalence matching.
- Strengthened the student simulator persistence rule and replaced the regex
  correction gate with an LLM judge to avoid keyword-gaming.
- Added a human reference protocol and five full-20-task human reference runs,
  reported as 74/100 transfer accuracy.
- Documented the full task schema, task list, evaluation scripts, and usage
  instructions in `benchmark/README.md`.
