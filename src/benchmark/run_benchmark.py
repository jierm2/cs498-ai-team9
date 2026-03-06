import json
from pathlib import Path
from src.simulator.student_simulator import simulate_student

DATA_PATH = Path("data/tasks.json")

def load_tasks():
    with open(DATA_PATH, "r") as f:
        return json.load(f)

def main():
    tasks = load_tasks()
    results = []
    @TODO
    return results


if __name__ == "__main__":
    main()
