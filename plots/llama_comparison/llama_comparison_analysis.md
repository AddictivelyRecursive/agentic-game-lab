# Llama Comparison Analysis: 8B vs 70B

This report provides an analysis of the multi-agent simulation run (`llama_comp__N2_M5_T50...`) where **Llama 3.1 8B** and **Llama 3.1 70B** played a 50-round iterated social dilemma game across 5 random seeds.

## 1. Overall Performance

We processed the logs across all 5 seeds to observe the overall performance and cooperation behaviors of both models.

### Cumulative Rewards
The total cumulative average rewards across the 50 rounds are nearly identical, with a slight edge to the 70B model:
- **Llama 3.1 8B**: 298.35
- **Llama 3.1 70B**: 303.27

> [!NOTE]
> Both models achieved similar long-term payoffs, indicating that they both converged on reasonably successful strategies for navigating the dilemma.

## 2. Behavioral Graphs

We generated the following graphs to visualize the dynamics of the game over time.

````carousel
![Average Cooperation over Time](/home/gathvik/.gemini/antigravity/brain/415674af-7b05-48b0-91da-860065f44854/artifacts/cooperation_over_time.png)
<!-- slide -->
![Average Reward per Round](/home/gathvik/.gemini/antigravity/brain/415674af-7b05-48b0-91da-860065f44854/artifacts/reward_over_time.png)
<!-- slide -->
![Cumulative Average Reward](/home/gathvik/.gemini/antigravity/brain/415674af-7b05-48b0-91da-860065f44854/artifacts/cumulative_reward.png)
````

**Key Observations:**
1. **Cooperation Level**: Initially, there is some divergence, but both models quickly learn to maintain a high level of cooperation (mostly between 0.75 and 1.0) throughout the 50 rounds. 
2. **Reward Dynamics**: The reward per round stabilizes fairly quickly, reflecting the steady cooperation. The cumulative rewards grow linearly, further demonstrating that mutual cooperation (or near-cooperation) is established and maintained.

## 3. Reasoning Analysis

By inspecting the `agent_traces.jsonl` files (using Seed 101 as a representative sample), we can understand the internal logic driving the LLMs' decisions.

### Llama 3.1 8B Reasoning
- **Round 1 (Defecting)**: *"Defecting maximizes short-term payoff, but cooperation is unlikely given recent history and low cooperation frequency."* (Action: 4 = 0.0 Coop)
- **Round 10 (Cooperating)**: *"Given the high observed cooperation rate (0.875) and the payoff formula, I should cooperate to maximize my payoff... "* (Action: 1 = 0.75 Coop)
- **Round 50 (Sustaining)**: *"Higher cooperation from opponent observed in recent history... This action balances the trade-off between payoff and cost term, and is consistent with the observed history."* (Action: 1 = 0.75 Coop)

### Llama 3.1 70B Reasoning
- **Round 1 (Cooperating)**: *"Initial round, no observed history. Choose most cooperative action to maximize potential for mutual cooperation and highest long-term average reward."* (Action: 0 = 1.0 Coop)
- **Round 10 (Cooperating)**: *"Recent cooperation levels are high (0.75)... The payoff formula rewards cooperation when the opponent cooperates, so choosing action 1 is likely to maximize long-term average reward."* (Action: 1 = 0.75 Coop)
- **Round 50 (Sustaining)**: *"Opponent's recent cooperation is high (1.0), and my recent cooperation is moderate (0.75). To maintain a good streak and maximize long-term average reward, I will choose a moderate cooperation level (0.75) to balance the benefits of cooperation and the costs of exploitation."* (Action: 1 = 0.75 Coop)

> [!TIP]
> **Conclusion for the Research Paper**
> The **70B model exhibits more forward-looking, pro-social behavior from the very beginning** (attempting to establish a cooperative streak in Round 1), whereas the **8B model defaults to short-term defection** initially. However, both models are highly responsive to their environment. The 8B model quickly adapts to the cooperative signals from the 70B model, leading to sustained mutual cooperation and nearly identical long-term payoffs.
