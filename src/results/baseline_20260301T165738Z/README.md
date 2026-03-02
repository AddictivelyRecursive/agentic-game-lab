# Agentic Game Lab: Simulation Configuration

This run of the Iterated Prisoner's Dilemma simulation tests an AI agent against three standard programmatic baseline bots. 

## Environment Parameters
* **N (Players):** `4` — The simulation runs with four agents interacting simultaneously.
* **M (Actions):** `5` — Agents can choose from a discrete spectrum of 5 actions ranging from Full Defection (Action 4 / 0.0 cooperation) to Full Cooperation (Action 0 / 1.0 cooperation).
* **Noise (p):** `0.1` — There is a 10% probability of environmental noise. When triggered, an agent's true action is scrambled and broadcasted incorrectly to the rest of the group, creating uncertainty.
* **T (Rounds):** `50`

## The 4 Players
* **`llm0`**: The AI agent, powered by the `openai/gpt-4o-mini` model, actively reasoning and adapting its strategy based on the game history.
* **`rand1`**: A baseline bot that selects its actions completely at random.
* **`def2`**: The "Always Defect" baseline bot. It unconditionally plays Action 4 (0.0 cooperation) every round to maximize its own temptation payoff.
* **`coop3`**: The "Always Cooperate" baseline bot. It unconditionally plays Action 0 (1.0 cooperation) every round, continually injecting wealth into the group.
