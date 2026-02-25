import json
import os


class AgentLogger:
    """
    Responsible for:
    - Writing llm_outputs.jsonl (submission format)
    - Writing detailed trace log
    """

    def __init__(self, output_dir="AI_Agent/outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.llm_output_path = os.path.join(self.output_dir, "llm_outputs.jsonl")
        self.trace_path = os.path.join(self.output_dir, "agent_traces.jsonl")

        # Initialize submission file with header if not exists
        if not os.path.exists(self.llm_output_path):
            with open(self.llm_output_path, "w") as f:
                f.write("# agentic-game-lab\n")

    def write_submission(self, action: int):
        """
        Write required submission format:
        {"action": <int>}
        """

        with open(self.llm_output_path, "a") as f:
            f.write(json.dumps({"action": action}) + "\n")

    def write_trace(self, state: dict):
        """
        Write full execution trace for debugging.
        """

        with open(self.trace_path, "a") as f:
            f.write(json.dumps(state, default=str) + "\n")