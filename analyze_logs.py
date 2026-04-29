import os
import json
import matplotlib.pyplot as plt
import numpy as np

base_dir = "/home/gathvik/llm_new/agentic-game-lab/src/results/llama_comparison/llama_comp__N2_M5_T50__p0.05__lam0.25__eta0.35__seeds5__llama_family__m5__t50__20260428_085020"
seeds = ["101", "102", "103", "104", "105"]
# artifact_dir = "/home/gathvik/.gemini/antigravity/brain/415674af-7b05-48b0-91da-860065f44854/artifacts"
artifact_dir = "/home/gathvik/llm_new/agentic-game-lab/src/plots/llama_comparison/"

os.makedirs(artifact_dir, exist_ok=True)

T = 50
coop_data_8b = np.zeros((len(seeds), T))
coop_data_70b = np.zeros((len(seeds), T))
reward_data_8b = np.zeros((len(seeds), T))
reward_data_70b = np.zeros((len(seeds), T))

reasoning_examples = []

for s_idx, seed in enumerate(seeds):
    match_dir = os.path.join(base_dir, f"llama_match__s{seed}")
    ep_logs_path = os.path.join(match_dir, "episode_logs.jsonl")
    
    with open(ep_logs_path, 'r') as f:
        lines = f.readlines()
        for t, line in enumerate(lines):
            data = json.loads(line)
            # index 0 is 8b, index 1 is 70b based on context.seat_assignment
            true_coop = data["true_coop"]
            rewards = data["rewards"]
            coop_data_8b[s_idx, t] = true_coop[0]
            coop_data_70b[s_idx, t] = true_coop[1]
            reward_data_8b[s_idx, t] = rewards[0]
            reward_data_70b[s_idx, t] = rewards[1]
            
    # Sample some reasoning
    if seed == "101":
        for p_idx, player in enumerate(["p0__llama31_8b", "p1__llama31_70b"]):
            trace_path = os.path.join(match_dir, "agents", player, "agent_traces.jsonl")
            with open(trace_path, 'r') as f:
                trace_lines = f.readlines()
                for step in [0, 9, 24, 49]: # Rounds 1, 10, 25, 50
                    trace_data = json.loads(trace_lines[step])
                    decision = trace_data.get("decision", {})
                    reasoning_examples.append({
                        "player": player,
                        "round": step + 1,
                        "action": decision.get("a"),
                        "reason": decision.get("reason", "")
                    })

# Plot 1: Average Cooperation over Time
mean_coop_8b = np.mean(coop_data_8b, axis=0)
mean_coop_70b = np.mean(coop_data_70b, axis=0)
std_coop_8b = np.std(coop_data_8b, axis=0)
std_coop_70b = np.std(coop_data_70b, axis=0)

plt.figure(figsize=(10, 6))
x = np.arange(1, T+1)
plt.plot(x, mean_coop_8b, label='Llama 3.1 8B', color='blue')
plt.fill_between(x, mean_coop_8b - std_coop_8b, mean_coop_8b + std_coop_8b, color='blue', alpha=0.2)
plt.plot(x, mean_coop_70b, label='Llama 3.1 70B', color='red')
plt.fill_between(x, mean_coop_70b - std_coop_70b, mean_coop_70b + std_coop_70b, color='red', alpha=0.2)
plt.title('Average Cooperation over Time (5 seeds)')
plt.xlabel('Round (t)')
plt.ylabel('Cooperation Level (0=Defect, 1=Cooperate)')
plt.ylim(-0.1, 1.1)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(artifact_dir, "cooperation_over_time.png"))
plt.close()

# Plot 2: Average Reward over Time
mean_reward_8b = np.mean(reward_data_8b, axis=0)
mean_reward_70b = np.mean(reward_data_70b, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(x, mean_reward_8b, label='Llama 3.1 8B', color='blue')
plt.plot(x, mean_reward_70b, label='Llama 3.1 70B', color='red')
plt.title('Average Reward per Round')
plt.xlabel('Round (t)')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(artifact_dir, "reward_over_time.png"))
plt.close()

# Plot 3: Cumulative Reward
cum_reward_8b = np.cumsum(mean_reward_8b)
cum_reward_70b = np.cumsum(mean_reward_70b)

plt.figure(figsize=(10, 6))
plt.plot(x, cum_reward_8b, label='Llama 3.1 8B', color='blue')
plt.plot(x, cum_reward_70b, label='Llama 3.1 70B', color='red')
plt.title('Cumulative Average Reward')
plt.xlabel('Round (t)')
plt.ylabel('Cumulative Reward')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(artifact_dir, "cumulative_reward.png"))
plt.close()

# Output total cumulative rewards
print(f"Total Cumulative Average Reward - Llama 3.1 8B: {cum_reward_8b[-1]:.2f}")
print(f"Total Cumulative Average Reward - Llama 3.1 70B: {cum_reward_70b[-1]:.2f}")
print("Reasoning examples written to JSON.")

with open(os.path.join(artifact_dir, "reasoning_examples.json"), "w") as f:
    json.dump(reasoning_examples, f, indent=2)

