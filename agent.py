from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from dotenv import load_dotenv
from google import genai

load_dotenv()

_client = genai.Client()
AGENT_MODEL   = "gemini-2.5-pro-preview-03-25"
TRACKER_MODEL = "gemini-2.0-flash"


@dataclass
class TrackerState:
    misconception_identified: bool = False
    counter_example_shown: bool = False
    student_shifting: bool = False
    confirmed_correction: bool = False
    turn_count: int  = 0
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

    def update(self, dialogue_history: list[dict], latest_student_response: str, last_move: str) -> TrackerState:
        self.state.turn_count += 1
        self.state.last_move = last_move

        history_text = _format_history(dialogue_history)
        prompt = f"""You are a precise educational state tracker. Update the tracker state based on the conversation.
            Misconception being addressed: {self.misconception}
            Conversation so far:{history_text}
            Latest student response: "{latest_student_response}"

Current state:
{self.state.to_json()}

Return ONLY a JSON object, no other text."""

        try:
            response=_client.models.generate_content(
                model=TRACKER_MODEL,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config={"temperature": 0.0},
            )
            updates = json.loads(_extract_json(response.text.strip()))
            for f in self._BOOL_FIELDS:
                if updates.get(f) is True:
                    setattr(self.state, f, True)
            if isinstance(updates.get("diagnosis_notes"), str):
                self.state.diagnosis_notes = updates["diagnosis_notes"]
        except Exception:
            pass

        return self.state

    def reset(self) -> None:
        self.state = TrackerState()


MOVE_PROBING="probing"
MOVE_CONFRONTING="confronting"
MOVE_CONFIRMING ="confirming"


class DialoguePlanner:
    def select_move(self, state: TrackerState, max_turns: int, turns_remaining: int) -> str:
        if turns_remaining <= 1:
            return MOVE_CONFIRMING
        if not state.misconception_identified:
            return MOVE_PROBING
        if not state.counter_example_shown:
            return MOVE_CONFRONTING
        if not state.student_shifting:
            if state.turn_count >= 3 and state.last_move == MOVE_CONFRONTING:
                return MOVE_CONFIRMING
            return MOVE_CONFRONTING
        return MOVE_CONFIRMING

    def describe_move(self, move: str, misconception: str) -> str:
        return {
            MOVE_PROBING: (
                f"Ask the student to explain their reasoning step by step for a problem "
                f"related to: {misconception}. Do NOT correct them yet — just listen and probe."
            ),
            MOVE_CONFRONTING: (
                f"Present a concrete counter-example or logical contradiction that directly "
                f"challenges: {misconception}. Make the conflict undeniable but do not give away the answer."
            ),
            MOVE_CONFIRMING: (
                f"Test whether understanding has improved by asking a simpler variant of the "
                f"concept. Check if the correction has stuck, related to: {misconception}."
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


class TutoringAgent:
    _SYSTEM_PROMPT = (
        "You are an expert math tutor. Your goal is to help a student overcome a "
        "specific mathematical misconception through guided dialogue. "
        "You must NOT give away the correct answer directly. "
        "Instead, guide the student to discover the error in their own reasoning. "
        "Keep your responses concise (2–4 sentences). Be warm but precise."
    )

    def __init__(self) -> None:
        self.planner = DialoguePlanner()

    def run_session(self, task: dict, get_student_response=None, verbose: bool = False) -> SessionResult:
        tracker = MisconceptionTracker(task["misconception"])
        dialogue_history: list[dict] = []
        max_turns: int = task.get("max_turns", 6)

        for turn_num in range(1, max_turns + 1):
            turns_remaining = max_turns - turn_num + 1
            move = self.planner.select_move(tracker.state, max_turns, turns_remaining)
            tutor_message = self._generate_tutor_response(task, dialogue_history, tracker.state, move)

            if verbose:
                print(f"\n[Turn {turn_num}] Move: {move}")
                print(f"Tutor: {tutor_message}")

            student_response = get_student_response(tutor_message) if get_student_response else input("Student: ").strip()

            if verbose:
                print(f"Student: {student_response}")

            dialogue_history.append({"role": "tutor",   "content": tutor_message})
            dialogue_history.append({"role": "student", "content": student_response})
            tracker.update(dialogue_history, student_response, move)

        transfer_prompt = f"Now answer this question on your own: {task['transfer_question']}"

        if verbose:
            print(f"\n[Transfer Test] {task['transfer_question']}")

        student_transfer_answer = get_student_response(transfer_prompt) if get_student_response else input(f"Transfer Q — {task['transfer_question']}\nStudent: ").strip()

        if verbose:
            print(f"Student: {student_transfer_answer}")

        return SessionResult(
            task_id=task["id"],
            turns_taken=tracker.state.turn_count,
            transfer_question=task["transfer_question"],
            student_transfer_answer=student_transfer_answer,
            dialogue_history=dialogue_history,
            final_tracker_state=tracker.state.to_dict(),
        )

    def _generate_tutor_response(self, task: dict, dialogue_history: list[dict], state: TrackerState, move: str) -> str:
        move_desc    = self.planner.describe_move(move, task["misconception"])
        history_text = _format_history(dialogue_history)

        state_summary = (
            f"- Misconception identified: {state.misconception_identified}\n"
            f"- Counter-example shown: {state.counter_example_shown}\n"
            f"- Student is shifting: {state.student_shifting}\n"
            f"- Correction confirmed: {state.confirmed_correction}\n"
            f"- Diagnosis notes: {state.diagnosis_notes or 'none yet'}"
        )

        user_prompt = f"""Misconception to address: {task['misconception']}

Current session state:
{state_summary}

Planned move: {move}
Move instructions: {move_desc}

Conversation so far:
{history_text if history_text != '(no turns yet)' else '(session just started)'}

Write ONLY your next tutor message. No labels or prefixes."""

        try:
            response = _client.models.generate_content(
                model=AGENT_MODEL,
                contents=[{"role": "user", "parts": [{"text": user_prompt}]}],
                config={"system_instruction": self._SYSTEM_PROMPT, "temperature": 0.7},
            )
            return response.text.strip()
        except Exception:
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
    if "```" in text:
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        return "\n".join(lines).strip()
    return text


if __name__ == "__main__":#curr test
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
