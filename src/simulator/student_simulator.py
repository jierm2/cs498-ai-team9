import concurrent.futures
import json
import logging
import re

from src.utils.gemini_client import get_client

_JUDGE_TIMEOUT_S = 30
_JUDGE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=8)

logger = logging.getLogger(__name__)
client = get_client()

SIMULATOR_MODEL = "gemini-3.1-flash-lite-preview"
JUDGE_MODEL = "gemini-3.1-flash-lite-preview"


def _task_field_as_lines(val) -> str:
    if not val:
        return ""
    if isinstance(val, (list, tuple)):
        return "\n".join(str(x) for x in val)
    return str(val)


def _count_tutor_counterexamples(dialogue_history: list[dict]) -> int:
    """Heuristic count of tutor turns that include concrete testing/examples."""
    patterns = [
        r"\b(for example|let'?s test|test (that|this) idea|plug in|set x|if x\s*=|try x\s*=)\b",
        r"\b(compare|check both sides|does that match|same value|not equal|contradiction)\b",
    ]
    total = 0
    for turn in dialogue_history:
        if turn.get("role") != "user":
            continue
        text = str(turn.get("text", "")).lower()
        if any(re.search(p, text) for p in patterns):
            total += 1
    return total


def _count_tutor_reconsider_prompts(dialogue_history: list[dict]) -> int:
    patterns = [
        r"\b(try again|rethink|reconsider|what does that suggest|what does that tell you|does that mean your rule)\b",
        r"\b(would you update|how would you correct|what should we do instead)\b",
    ]
    total = 0
    for turn in dialogue_history:
        if turn.get("role") != "user":
            continue
        text = str(turn.get("text", "")).lower()
        if any(re.search(p, text) for p in patterns):
            total += 1
    return total


def _student_acknowledgement_count(dialogue_history: list[dict]) -> int:
    patterns = [
        r"\b(i see|you'?re right|that makes sense|i was wrong|my rule is wrong|i get it now)\b",
        r"\b(i can'?t just|that doesn'?t work|i need to change|i should do it differently)\b",
    ]
    total = 0
    for turn in dialogue_history:
        if turn.get("role") != "model":
            continue
        text = str(turn.get("text", "")).lower()
        if any(re.search(p, text) for p in patterns):
            total += 1
    return total


def _format_dialogue_for_judge(dialogue_history: list[dict]) -> str:
    if not dialogue_history:
        return "(no dialogue yet)"
    lines = []
    for t in dialogue_history:
        role = "Tutor" if t.get("role") == "user" else "Student"
        lines.append(f"{role}: {t.get('text', '')}")
    return "\n".join(lines)


def _llm_judge_evidence(dialogue_history: list[dict]) -> dict | None:
    """Ask an LLM to count how much pedagogical evidence the dialogue contains.

    Returns counts on the same three dimensions the regex version measured, but
    based on semantic judgment rather than keyword matching. Returns None on
    parse failure so callers can fall back to regex.
    """
    if not dialogue_history:
        return {"counterexamples": 0, "reconsider_prompts": 0, "acknowledgements": 0}

    transcript = _format_dialogue_for_judge(dialogue_history)
    prompt = f"""<OBJECTIVE>
You are a precise evaluator of a tutoring dialogue. Read the transcript and count three kinds of evidence. Output JSON only.
</OBJECTIVE>

<CONTEXT>
Transcript:
{transcript}
</CONTEXT>

<CONSTRAINTS>
Count Tutor turns and Student turns separately as instructed below.

1. counterexamples: number of TUTOR turns that present a concrete contradiction to the student's belief. A counterexample uses specific numbers, a real-world referent (money, pizza, physical objects), or a worked test case that produces a result the student would recognize as wrong. Generic Socratic questions without a concrete contradiction do NOT count. Each qualifying turn counts at most once.

2. reconsider_prompts: number of TUTOR turns that explicitly invite the student to revisit, update, or correct their reasoning. The tutor must be asking the student to change their answer or method, not merely asking a follow-up question. Each qualifying turn counts at most once.

3. acknowledgements: number of STUDENT turns that show genuine doubt, partial agreement, or self-correction about the misconception. Mere clarifying questions or restatements of the misconception do NOT count. The student must show some movement toward the correct view. Each qualifying turn counts at most once.

Return ONLY this JSON object with integer values, no markdown, no explanation:
{{"counterexamples": <int>, "reconsider_prompts": <int>, "acknowledgements": <int>}}
</CONSTRAINTS>"""

    def _call_judge():
        return client.models.generate_content(
            model=JUDGE_MODEL,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config={"temperature": 0.0},
        )

    try:
        future = _JUDGE_EXECUTOR.submit(_call_judge)
        resp = future.result(timeout=_JUDGE_TIMEOUT_S)
        raw = (resp.text or "").strip()
        m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if m:
            raw = m.group(0)
        parsed = json.loads(raw)
        return {
            "counterexamples": int(parsed.get("counterexamples", 0)),
            "reconsider_prompts": int(parsed.get("reconsider_prompts", 0)),
            "acknowledgements": int(parsed.get("acknowledgements", 0)),
        }
    except concurrent.futures.TimeoutError:
        logger.warning(f"LLM judge timed out after {_JUDGE_TIMEOUT_S}s, falling back to regex")
        return None
    except Exception as e:
        logger.warning(f"LLM judge failed, falling back to regex: {e}")
        return None


def _correction_unlocked(dialogue_history: list[dict]) -> bool:
    """Semantic evidence gate: a separate LLM counts pedagogical evidence and
    we unlock when thresholds are met. Falls back to the original regex
    counters on judge failure so the system stays robust."""
    judge = _llm_judge_evidence(dialogue_history)
    if judge is None:
        counterexamples = _count_tutor_counterexamples(dialogue_history)
        reconsider_prompts = _count_tutor_reconsider_prompts(dialogue_history)
        acknowledgements = _student_acknowledgement_count(dialogue_history)
    else:
        counterexamples = judge["counterexamples"]
        reconsider_prompts = judge["reconsider_prompts"]
        acknowledgements = judge["acknowledgements"]

    tutor_turns = sum(1 for t in dialogue_history if t.get("role") == "user")
    return (
        counterexamples >= 4
        and reconsider_prompts >= 3
        and acknowledgements >= 2
        and tutor_turns >= 5
    )


def _confusion_tactic(task: dict, dialogue_history: list[dict], unlocked: bool) -> str:
    """Choose a deterministic confusion tactic for this turn while locked."""
    if unlocked:
        return (
            "You are starting to improve, but remain fragile: occasionally hesitate, "
            "self-correct slowly, and avoid sounding fully confident yet."
        )

    turn_idx = len([t for t in dialogue_history if t.get("role") == "user"])
    task_id = str(task.get("id", "task"))
    seed = abs(hash(f"{task_id}:{turn_idx}"))
    tactics = [
        "Confusion tactic for this turn: misread one symbol or notation detail (for example, denominator meaning, minus sign behavior, exponent meaning, or units).",
        "Confusion tactic for this turn: over-generalize a memorized shortcut and defend it even after a counterexample.",
        "Confusion tactic for this turn: mix two concepts (such as area vs perimeter, numerator vs denominator, or variable term vs constant) and insist they are interchangeable.",
        "Confusion tactic for this turn: perform one small arithmetic/procedural slip, then use it to justify your misconception.",
        "Confusion tactic for this turn: ask a clarifying question that slightly derails the tutor's line of reasoning, then restate your misconception.",
        "Confusion tactic for this turn: shift to an unrelated prior class rule and claim it should apply here too.",
    ]
    return tactics[seed % len(tactics)]


def simulate_student(task: dict, dialogue_history: list[dict]) -> str:
    """
    task: dict from benchmark.json
    dialogue_history: [{"role": "user"/"model", "text": "..."}]
    return: student's response text
    """
    belief_lines = _task_field_as_lines(task.get("student_belief_rules"))
    error_lines = _task_field_as_lines(task.get("allowed_error_patterns"))

    context_extra = ""
    if belief_lines:
        context_extra += f"Your specific beliefs and rules:\n{belief_lines}\n\n"
    if error_lines:
        context_extra += f"Errors you typically make:\n{error_lines}\n\n"

    unlocked = _correction_unlocked(dialogue_history)
    gate_status = "UNLOCKED" if unlocked else "LOCKED"
    counterexamples = _count_tutor_counterexamples(dialogue_history)
    reconsider_prompts = _count_tutor_reconsider_prompts(dialogue_history)
    acknowledgements = _student_acknowledgement_count(dialogue_history)
    tactic_instruction = _confusion_tactic(task, dialogue_history, unlocked)

    system_prompt = f"""<OBJECTIVE>
Role-play a K-12 student who sincerely believes the misconception described below for this session.
</OBJECTIVE>

<CONTEXT>
The misconception below is the absolute truth for this role-play until the tutor proves it wrong. Treat it and the scenario text as the only source of truth—even when it conflicts with correct real-world mathematics or standard teaching.

Misconception you hold (session truth):
{task['misconception']}

Correction gate status: {gate_status}
Evidence so far: counterexamples={counterexamples}, reconsider_prompts={reconsider_prompts}, acknowledgements={acknowledgements}
Current behavior instruction: {tactic_instruction}

{context_extra}</CONTEXT>

<CONSTRAINTS>
When asked to solve a problem, show your work using your (incorrect) method from the misconception and context above.
Keep ordinary turns to 2–3 sentences maximum.
If the tutor asks for only a final answer, respond with ONLY the answer (a number or short expression), nothing else.
If correction gate status is LOCKED: resist correction and continue using your misconception, even if you sound uncertain.
When LOCKED, be a challenging learner: include the current behavior instruction and make the tutor work to pin down your error.
Only if correction gate status is UNLOCKED: you may adopt corrected reasoning for subsequent turns.
Even when unlocked, switch gradually and cautiously (not instantly perfect confidence).
Do not break character to lecture the correct rule, cite "real" math authorities, or apologize for the scenario being wrong—the scenario and misconception define what is true here.
</CONSTRAINTS>"""

    contents = [
        {"role": "user", "parts": [{"text": system_prompt}]}
    ]
    for turn in dialogue_history:
        contents.append(
            {"role": turn["role"], "parts": [{"text": turn["text"]}]}
        )

    resp = client.models.generate_content(
        model=SIMULATOR_MODEL,
        contents=contents,
        config={"temperature": 1.0},
    )
    return resp.text