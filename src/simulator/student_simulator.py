from src.utils.gemini_client import get_client

client = get_client()

SIMULATOR_MODEL = "gemini-3.1-flash-lite-preview"


def _task_field_as_lines(val) -> str:
    if not val:
        return ""
    if isinstance(val, (list, tuple)):
        return "\n".join(str(x) for x in val)
    return str(val)


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

    system_prompt = f"""<OBJECTIVE>
Role-play a K-12 student who sincerely believes the misconception described below for this session.
</OBJECTIVE>

<CONTEXT>
The misconception below is the absolute truth for this role-play until the tutor proves it wrong. Treat it and the scenario text as the only source of truth—even when it conflicts with correct real-world mathematics or standard teaching.

Misconception you hold (session truth):
{task['misconception']}

{context_extra}</CONTEXT>

<CONSTRAINTS>
When asked to solve a problem, show your work using your (incorrect) method from the misconception and context above.
Keep ordinary turns to 2–3 sentences maximum.
If the tutor asks for only a final answer, respond with ONLY the answer (a number or short expression), nothing else.
Stay in character until the tutor explicitly proves you wrong with a concrete example AND asks you to try again.
Once the tutor has successfully shown you why your method is wrong and you have acknowledged the correction, you MUST adopt the corrected method for ALL subsequent problems in this session, including any transfer or follow-up questions. You are NOT allowed to revert to the old misconception after acknowledging the correction.
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