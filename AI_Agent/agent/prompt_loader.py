import os


def load_prompts(prompt_dir: str) -> dict:
    """
    Load all prompt templates from prompt directory.

    Expected files:
        base_system.txt
        opponent_model_system.txt
        decision_policy_system.txt
        repair_system.txt
    """

    prompts = {}

    for file in os.listdir(prompt_dir):
        if file.endswith(".txt"):
            key = file.replace(".txt", "")
            with open(os.path.join(prompt_dir, file), "r") as f:
                prompts[key] = f.read()

    return prompts