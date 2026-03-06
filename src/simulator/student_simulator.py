from src.utils.gemini_client import get_client

client = get_client()

def simulate_student(task, dialogue_history):
    """
    task: dict, from tasks.json
    dialogue_history: [{"role": "user"/"model", "text": "..."}], only contains tutor and student history
    return: student's response text
    """
    system_prompt = (
        "You are a K-12 student with this misconception:\n"
        f"{task['misconception']}\n"
        "Stay in character. Do NOT give the correct rule unless the tutor "
        "explicitly proves you wrong and asks you to try again."
    )

    contents = [
        {"role": "user", "parts": [{"text": system_prompt}]}
    ]
    for turn in dialogue_history:
        contents.append(
            {"role": turn["role"], "parts": [{"text": turn["text"]}]}
        )

    resp = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=contents,
    )
    return resp.text
