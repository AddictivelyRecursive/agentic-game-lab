import json
import argparse
from agent.llm_agent import LLMAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    with open(args.input, "r") as f:
        turn_input = json.load(f)

    agent = LLMAgent()
    action = agent.step(turn_input)

    print("Chosen action:", action)


if __name__ == "__main__":
    main()