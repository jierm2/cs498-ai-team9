from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from dotenv import load_dotenv

load_dotenv()

from src.utils.gemini_client import get_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_client = get_client()
AGENT_MODEL   = "gemini-3-flash-preview"
TRACKER_MODEL = "gemini-3-flash-preview"

MAX_TRACKER_RETRIES = 2


@dataclass
class TrackerState:
    misconception_identified: bool = False
    counter_example_shown: bool = False
    student_shifting: bool = False
    confirmed_correction: bool = False
    turn_count: int = 0
    diagnosis_notes: str = ""
    last_move: str = "none"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class MisconceptionTracker:
    _BOOL_FIELDS = [
        "misconception_identified",
        "counter_example_shown",
        "student_shifting",
        "confirmed_correction",
    ]

    def __init__(self, misconception: str) -> None:
        self.misconception = misconception
        self.state = TrackerState()
        self.parse_failures = 0

    def update(self, dialogue_history: list[dict], latest_student_response: str, last_move: str) -> TrackerState:
        self.state.turn_count += 1
        self.state.last_move = last_move

        history_text = _format_history(dialogue_history)
        prompt = f"""<OBJECTIVE>
You are a precise educational state tracker. Read the dialogue and output the updated tracker state as a single JSON object.
</OBJECTIVE>

<CONTEXT>
Misconception being addressed: {self.misconception}

Conversation so far:
{history_text}

Latest student response: "{latest_student_response}"

The tutor's last move was: {last_move}

Current state:
{self.state.to_json()}
</CONTEXT>

<CONSTRAINTS>
Field semantics:
- misconception_identified: true if the tutor has clearly named or described the student's specific error.
- counter_example_shown: true if the tutor presented a concrete example or contradiction that challenges the misconception.
- student_shifting: true if the student shows doubt, asks clarifying questions, or starts correcting themselves.
- confirmed_correction: true only if the student explicitly states correct reasoning or the correct answer.
- diagnosis_notes: one short sentence describing what the student currently believes.

Do not invent dialogue that did not occur. Do not output keys other than those listed below.
Once a field is set to true, it must remain true in your output.

Return ONLY valid JSON with exactly these keys (no nulls for the booleans; use true/false):
{{"misconception_identified": bool, "counter_example_shown": bool, "student_shifting": bool, "confirmed_correction": bool, "diagnosis_notes": "..."}}

No markdown, no explanation, no code fences. ONLY the JSON object.
</CONSTRAINTS>"""

        for attempt in range(MAX_TRACKER_RETRIES + 1):
            try:
                response = _client.models.generate_content(
                    model=TRACKER_MODEL,
                    contents=[{"role": "user", "parts": [{"text": prompt}]}],
                    config={"temperature": 0.0},
                )
                raw = response.text.strip()
                json_str = _extract_json(raw)
                updates = json.loads(json_str)

                # Validate that we got the expected keys
                if not isinstance(updates, dict):
                    raise ValueError(f"Expected dict, got {type(updates)}")

                for f in self._BOOL_FIELDS:
                    if updates.get(f) is True:
                        setattr(self.state, f, True)
                if isinstance(updates.get("diagnosis_notes"), str) and updates["diagnosis_notes"]:
                    self.state.diagnosis_notes = updates["diagnosis_notes"]

                logger.debug(f"Tracker updated on attempt {attempt + 1}: {self.state.to_dict()}")
                return self.state

            except Exception as e:
                logger.warning(f"Tracker parse attempt {attempt + 1} failed: {e}")
                if attempt < MAX_TRACKER_RETRIES:
                    continue

        # All retries exhausted — apply heuristic fallback
        self.parse_failures += 1
        logger.error(f"Tracker fallback triggered (total failures: {self.parse_failures})")
        self._heuristic_update(latest_student_response, last_move)
        return self.state

    def _heuristic_update(self, student_response: str, last_move: str) -> None:
        """Simple keyword-based fallback when LLM parsing fails."""
        lower = student_response.lower()

        if last_move == MOVE_PROBING:
            # If tutor was probing, assume misconception is now visible
            self.state.misconception_identified = True

        if last_move == MOVE_CONFRONTING:
            self.state.counter_example_shown = True

        # Detect shifting signals
        shift_signals = ["i see", "oh", "wait", "that makes sense", "you're right",
                         "so i should", "i think i understand", "hmm", "actually"]
        if any(s in lower for s in shift_signals):
            self.state.student_shifting = True

    def reset(self) -> None:
        self.state = TrackerState()
        self.parse_failures = 0


MOVE_PROBING = "probing"
MOVE_CONFRONTING = "confronting"
MOVE_TEACHING = "teaching"
MOVE_CONFIRMING = "confirming"


class DialoguePlanner:
    def __init__(self, enable_teaching_move: bool = True) -> None:
        self.enable_teaching_move = enable_teaching_move

    def select_move(self, state: TrackerState, max_turns: int, turns_remaining: int) -> str:
        if turns_remaining <= 1:
            return MOVE_CONFIRMING

        if not state.misconception_identified:
            if state.turn_count >= 1:
                return MOVE_CONFRONTING
            return MOVE_PROBING

        if not state.counter_example_shown:
            return MOVE_CONFRONTING

        if state.confirmed_correction:
            return MOVE_CONFIRMING

        if not self.enable_teaching_move:
            # Ablation: original probe→confront→confirm loop, no teaching turn.
            if state.student_shifting:
                return MOVE_CONFIRMING
            if state.last_move == MOVE_CONFRONTING:
                return MOVE_CONFIRMING
            return MOVE_CONFRONTING

        # After at least one confrontation, escalate to a teaching turn that
        # names the correct rule and walks a worked example. This is what
        # gets the simulator past the "locked" gate AND gives it a procedure
        # to copy on the transfer test.
        if state.last_move in (MOVE_CONFRONTING, MOVE_PROBING):
            return MOVE_TEACHING

        if state.last_move == MOVE_TEACHING:
            return MOVE_CONFIRMING

        return MOVE_CONFRONTING

    def describe_move(self, move: str, misconception: str) -> str:
        return {
            MOVE_PROBING: (
                f"Ask the student to explain their reasoning step by step for a problem "
                f"related to: {misconception}. Listen and probe to understand their thinking."
            ),
            MOVE_CONFRONTING: (
                f"Present a concrete counter-example that directly challenges: {misconception}. "
                f"Use specific small numbers and a real-world referent (money, pizza slices, "
                f"physical objects) so the contradiction is undeniable to the student."
            ),
            MOVE_TEACHING: (
                f"The student has seen the contradiction. Now name the correct rule that fixes "
                f"the misconception '{misconception}', and walk through ONE complete worked "
                f"example with different numbers than the transfer question, showing every step. "
                f"You may state the rule and the example's answer — the goal is for the student "
                f"to have a procedure they can imitate. Do NOT solve the transfer question "
                f"itself, only a different example."
            ),
            MOVE_CONFIRMING: (
                f"Ask the student to redo a small problem related to: {misconception} using the "
                f"corrected method. Check whether they apply the rule correctly without giving "
                f"the final answer."
            ),
        }.get(move, "Continue the dialogue appropriately.")


@dataclass
class SessionResult:
    task_id: str
    turns_taken: int
    transfer_question: str
    student_transfer_answer: str
    dialogue_history: list[dict] = field(default_factory=list)
    final_tracker_state: dict = field(default_factory=dict)
    tracker_parse_failures: int = 0


TUTOR_SYSTEM_PROMPT = """<OBJECTIVE>
You are an expert math tutor. Help the student overcome a specific mathematical misconception through guided dialogue.
</OBJECTIVE>

<CONTEXT>
Guide the student to notice tensions in their own reasoning. Keep each reply concise (2–4 sentences). Be warm but precise.
</CONTEXT>

<CONSTRAINTS>
Do not repeat the same question or explanation you already gave.
Do not give away the correct answer directly; do not state the final numerical result, closed-form answer, or canonical rule that fully resolves the task for the student.
</CONSTRAINTS>"""


class TutoringAgent:
    _SYSTEM_PROMPT = TUTOR_SYSTEM_PROMPT

    def __init__(self, enable_teaching_move: bool = True, enable_transfer_recap: bool = True) -> None:
        self.enable_teaching_move = enable_teaching_move
        self.enable_transfer_recap = enable_transfer_recap
        self.planner = DialoguePlanner(enable_teaching_move=enable_teaching_move)

    def run_session(self, task: dict, get_student_response=None, verbose: bool = False) -> SessionResult:
        tracker = MisconceptionTracker(task["misconception"])
        dialogue_history: list[dict] = []
        max_turns: int = int(task.get("max_turns", 6))

        for turn_num in range(1, max_turns + 1):
            turns_remaining = max_turns - turn_num + 1
            move = self.planner.select_move(tracker.state, max_turns, turns_remaining)
            tutor_message = self._generate_tutor_response(task, dialogue_history, tracker.state, move)

            if verbose:
                print(f"\n[Turn {turn_num}/{max_turns}] Move: {move}")
                print(f"Tutor: {tutor_message}")

            student_response = get_student_response(tutor_message) if get_student_response else input("Student: ").strip()

            if verbose:
                print(f"Student: {student_response}")

            dialogue_history.append({"role": "tutor",   "content": tutor_message})
            dialogue_history.append({"role": "student", "content": student_response})
            tracker.update(dialogue_history, student_response, move)

            if verbose:
                logger.info(f"Tracker state: identified={tracker.state.misconception_identified}, "
                            f"counter_ex={tracker.state.counter_example_shown}, "
                            f"shifting={tracker.state.student_shifting}, "
                            f"confirmed={tracker.state.confirmed_correction}")

            # Early exit if correction is confirmed and we still have turns
            if tracker.state.confirmed_correction and turns_remaining > 1:
                if verbose:
                    print("[Early exit: correction confirmed]")
                break

        # Transfer prompt scaffolds the student by reminding them of the corrected
        # method we just taught. Without this, the simulator regresses to its
        # misconception on a cold-start single-shot question.
        if self.enable_transfer_recap:
            recap = self._build_transfer_recap(dialogue_history)
            transfer_prompt = (
                f"<OBJECTIVE>\n"
                f"Now answer this question, applying the corrected method we just discussed.\n"
                f"</OBJECTIVE>\n\n"
                f"<CONTEXT>\n"
                f"{recap}\n\n"
                f"Question: {task['transfer_question']}\n"
                f"</CONTEXT>\n\n"
                f"<CONSTRAINTS>\n"
                f"Give ONLY your final answer (a number, expression, or short phrase). No explanation.\n"
                f"</CONSTRAINTS>"
            )
        else:
            # Ablation: original cold-start transfer prompt, no recap.
            transfer_prompt = (
                f"<OBJECTIVE>\n"
                f"Answer the following question on your own.\n"
                f"</OBJECTIVE>\n\n"
                f"<CONTEXT>\n"
                f"{task['transfer_question']}\n"
                f"</CONTEXT>\n\n"
                f"<CONSTRAINTS>\n"
                f"Give ONLY your final answer (a number, expression, or short phrase). No explanation.\n"
                f"</CONSTRAINTS>"
            )

        if verbose:
            print(f"\n[Transfer Test] {task['transfer_question']}")

        student_transfer_answer = (
            get_student_response(transfer_prompt) if get_student_response
            else input(f"Transfer Q — {task['transfer_question']}\nStudent: ").strip()
        )

        if verbose:
            print(f"Student transfer answer: {student_transfer_answer}")

        return SessionResult(
            task_id=task["id"],
            turns_taken=tracker.state.turn_count,
            transfer_question=task["transfer_question"],
            student_transfer_answer=student_transfer_answer,
            dialogue_history=dialogue_history,
            final_tracker_state=tracker.state.to_dict(),
            tracker_parse_failures=tracker.parse_failures,
        )

    def _build_transfer_recap(self, dialogue_history: list[dict]) -> str:
        """Pull the last 2 tutor turns as a short recap so the simulator has
        the corrected method visible when answering the transfer question."""
        tutor_turns = [t["content"] for t in dialogue_history if t.get("role") == "tutor"]
        if not tutor_turns:
            return "(no prior discussion)"
        recap_turns = tutor_turns[-2:]
        joined = "\n".join(f"- Tutor said: {t}" for t in recap_turns)
        return f"Recap of what the tutor just taught:\n{joined}"

    def _generate_tutor_response(self, task: dict, dialogue_history: list[dict],
                                  state: TrackerState, move: str) -> str:
        move_desc = self.planner.describe_move(move, task["misconception"])
        history_text = _format_history(dialogue_history)

        state_summary = (
            f"- Misconception identified: {state.misconception_identified}\n"
            f"- Counter-example shown: {state.counter_example_shown}\n"
            f"- Student is shifting: {state.student_shifting}\n"
            f"- Correction confirmed: {state.confirmed_correction}\n"
            f"- Diagnosis notes: {state.diagnosis_notes or 'none yet'}"
        )

        move_guard = {
            MOVE_PROBING: "On this turn, do not correct the student's misconception; probe only.",
            MOVE_CONFRONTING: "On this turn, do not fully resolve the contradiction by giving the final answer or the complete correct rule.",
            MOVE_TEACHING: "On this turn, you MAY state the correct rule and fully solve a worked example with DIFFERENT numbers from the transfer question. Do NOT solve or reveal the answer to the transfer question itself.",
            MOVE_CONFIRMING: "On this turn, check understanding with a question or short task; do not lecture the full solution.",
        }.get(move, "")

        # During teaching, we explicitly allow giving the rule + worked example;
        # otherwise we suppress final-answer giveaway as before.
        no_giveaway_clause = (
            ""
            if move == MOVE_TEACHING
            else (
                "Do not give away the correct answer directly; do not state the final numerical "
                "result, closed-form answer, or canonical rule that fully resolves the task for "
                "the student.\n"
            )
        )

        transfer_q = task.get("transfer_question", "")
        transfer_clause = (
            f"You must NOT solve or reveal the answer to this exact transfer question: \"{transfer_q}\". "
            "Use different numbers in any worked example."
            if transfer_q
            else ""
        )

        user_prompt = f"""<OBJECTIVE>
Write ONLY your next tutor message as plain dialogue (no role labels or prefixes).
</OBJECTIVE>

<CONTEXT>
Misconception to address: {task['misconception']}

Current session state:
{state_summary}

Planned move: {move}
Move instructions: {move_desc}

Conversation so far:
{history_text if history_text != '(no turns yet)' else '(session just started)'}
</CONTEXT>

<CONSTRAINTS>
{move_guard}
{no_giveaway_clause}{transfer_clause}
Do not repeat the same question or explanation you already gave in this session.
Output only the tutor's next message—no headings, bullets, or meta-commentary about instructions.
</CONSTRAINTS>"""

        try:
            response = _client.models.generate_content(
                model=AGENT_MODEL,
                contents=[{"role": "user", "parts": [{"text": user_prompt}]}],
                config={"system_instruction": self._SYSTEM_PROMPT, "temperature": 1.0},
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Tutor generation failed: {e}")
            return "Can you walk me through your reasoning on that step by step?"


def _format_history(dialogue_history: list[dict]) -> str:
    if not dialogue_history:
        return "(no turns yet)"
    lines = []
    for turn in dialogue_history:
        role = "Tutor" if turn["role"] == "tutor" else "Student"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def _extract_json(text: str) -> str:
    """Extract JSON from LLM output, handling code fences, mixed text, etc."""
    # Try 1: code fence extraction
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # Try 2: find first { ... } block
    brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)

    # Try 3: nested braces
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i + 1]

    # Fallback: return as-is and let caller handle the parse error
    return text


if __name__ == "__main__":
    task = {
        "id": "F1",
        "topic": "fractions",
        "misconception": "Fraction addition adds numerators and denominators separately (1/2 + 1/3 = 2/5).",
        "transfer_question": "What is 1/3 + 1/4?",
        "correct_answer": "7/12",
        "max_turns": 4,
    }

    scripted = iter([
        "Easy! I add 1+1=2 on top and 2+3=5 on the bottom, so 2/5.",
        "Hmm, but 2/4 is just 1/2 which is the same as what I started with… that seems weird.",
        "Oh, I see — the pieces get smaller when the denominator is bigger.",
        "So I need to find a common denominator first?",
        "7/12",
    ])

    agent = TutoringAgent()
    result = agent.run_session(task, get_student_response=lambda _: next(scripted), verbose=True)

    print(f"\n{'='*60}\nTransfer answer: {result.student_transfer_answer}")
    print(f"Final tracker state:\n{json.dumps(result.final_tracker_state, indent=2)}")
    print(f"Tracker parse failures: {result.tracker_parse_failures}")