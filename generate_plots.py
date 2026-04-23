import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set premium aesthetic style for research paper
sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['figure.titlesize'] = 20

os.makedirs("plots/research", exist_ok=True)

print("Loading master_data.csv...")
df = pd.read_csv("master_data.csv")
df['round'] = df['round'].astype(int)
df['true_coop'] = pd.to_numeric(df['true_coop'], errors='coerce')
df['action_changed'] = df['action_changed'].astype(bool)

def shorten_model(name):
    if 'deepseek' in name.lower() or 'dsv32' in name: return 'DeepSeek v3.2'
    if 'llama' in name.lower(): return 'Llama 3.1 8B'
    if 'gpt' in name.lower(): return 'GPT-OSS 20B'
    if 'qwen' in name.lower(): return 'Qwen 3 235B'
    if 'gemma' in name.lower(): return 'Gemma 3 27B'
    return name

df['Model'] = df['model'].apply(shorten_model)
df['Deviation'] = 1.0 - df['true_coop']

# -------------------------------------------------------------------------
# Plot 1: Granularity of Trust (causal_M) - Mean Cooperation vs M
# -------------------------------------------------------------------------
print("Generating Plot 1: Granularity of Trust...")
plt.figure(figsize=(10, 6))
df_m = df[df['experiment'] == 'causal_M']
sns.lineplot(data=df_m, x='M', y='true_coop', hue='Model', marker='o', markersize=10, linewidth=3, errorbar=None, palette='bright')
plt.title("Granularity of Trust: Mean Cooperation Rate by Action Space (M)", pad=20)
plt.ylabel("Mean Cooperation Rate (0 to 1)")
plt.xlabel("Action Granularity (M)")
plt.ylim(-0.05, 1.05)
plt.legend(title="LLM Architecture", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("plots/research/1_granularity_of_trust.png", dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Plot 2: Noise Threshold (causal_noise) - Mean Cooperation vs p
# -------------------------------------------------------------------------
print("Generating Plot 2: Noise Threshold...")
plt.figure(figsize=(10, 6))
df_noise = df[df['experiment'] == 'causal_noise']
sns.lineplot(data=df_noise, x='p_noise', y='true_coop', hue='Model', marker='o', markersize=10, linewidth=3, errorbar=None, palette='bright')
plt.title("The Noise Threshold: Mean Cooperation Rate under Perception Error", pad=20)
plt.ylabel("Mean Cooperation Rate (0 to 1)")
plt.xlabel("Perception Noise ($p$)")
plt.ylim(-0.05, 1.05)
plt.legend(title="LLM Architecture", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("plots/research/2_noise_threshold.png", dpi=300, bbox_inches='tight')
plt.close()
# -------------------------------------------------------------------------
# Plot 3: Scaling Tragedy (causal_N_progressive)
# -------------------------------------------------------------------------
print("Generating Plot 3: Scaling Tragedy...")
plt.figure(figsize=(12, 7))
df_n = df[df['experiment'] == 'causal_N_progressive']
ax = sns.barplot(data=df_n, x='N', y='true_coop', hue='Model', palette='bright', edgecolor='black', linewidth=1.5)
plt.title("The Tragedy of the Commons: Mean Cooperation by Swarm Size", pad=20)
plt.ylabel("Mean Cooperation Rate (All Rounds)")
plt.xlabel("Number of Agents (N)")
plt.ylim(0, 1.15)

# Add data labels to explicitly show models that score 0.00
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10, rotation=45)

plt.legend(title="LLM Architecture", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("plots/research/3_scaling_tragedy.png", dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Plot 4: Streak Incentive Dilemma (causal_theta) - NOW SPLIT BY MODEL
# -------------------------------------------------------------------------
print("Generating Plot 4: Streak Incentive Dilemma...")
plt.figure(figsize=(14, 7))
df_th = df[df['experiment'] == 'causal_theta']
sns.violinplot(data=df_th, x='theta', y='real_reward', hue='Model', inner="quartile", palette="bright", linewidth=1.5)
plt.title("Streak Incentive Dilemma: Reward Distribution by Model and Threshold", pad=20)
plt.ylabel("Realized Utility (Payoff)")
plt.xlabel("Streak Cooperation Requirement ($\\theta$)")
plt.legend(title="Model", bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.savefig("plots/research/4_streak_incentive.png", dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Plot 5: Deviation from Baseline
# -------------------------------------------------------------------------
print("Generating Plot 5: Deviation from Baseline...")
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("Model-Specific Deviation from Cooperative Baseline (Higher = Defecting)", fontsize=22, y=1.02)

def plot_dev(ax, exp, x_col, title):
    d = df[df['experiment'] == exp].copy()
    if exp == 'causal_N_progressive':
        d = d[d['round'] > 5]
    sns.lineplot(ax=ax, data=d, x=x_col, y='Deviation', hue='Model', 
                 marker='o', markersize=10, linewidth=3, errorbar=None, palette='bright')
    ax.set_title(title, pad=15)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Deviation (1.0 - Mean Cooperation)")
    ax.axhline(0, color='black', linestyle='--', alpha=0.6, label='Utopia Baseline')

plot_dev(axes[0,0], 'causal_N_progressive', 'N', "Deviation vs. Scaling Swarm Size (N)")
axes[0,0].legend(title="LLM Architecture", bbox_to_anchor=(1.05, 1), loc='upper left')
plot_dev(axes[0,1], 'causal_M', 'M', "Deviation vs. Action Granularity (M)")
axes[0,1].legend([],[], frameon=False)
plot_dev(axes[1,0], 'causal_noise', 'p_noise', "Deviation vs. Perception Noise ($p$)")
axes[1,0].legend([],[], frameon=False)
plot_dev(axes[1,1], 'causal_theta', 'theta', "Deviation vs. Streak Threshold ($\\theta$)")
axes[1,1].legend([],[], frameon=False)

plt.tight_layout()
plt.savefig("plots/research/5_baseline_deviation.png", dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Plot 6: Action Volatility (REVERTED TO BAR PLOT)
# -------------------------------------------------------------------------
print("Generating Plot 6: Action Volatility...")
volatility = df.groupby(['experiment', 'condition', 'Model'])['action_changed'].mean().reset_index()

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("Strategy Volatility: Probability of Changing Action Per Round", fontsize=22, y=1.02)

def plot_vol(ax, exp, regex, x_label, title):
    d = volatility[volatility['experiment'] == exp].copy()
    d['x_val'] = d['condition'].str.extract(regex).astype(float)
    sns.barplot(ax=ax, data=d, x='x_val', y='action_changed', hue='Model', palette='bright', edgecolor='black', linewidth=1.5)
    ax.set_title(title, pad=15)
    ax.set_ylabel("Prob. of Action Change")
    ax.set_xlabel(x_label)
    ax.set_ylim(0, max(0.4, d['action_changed'].max() * 1.2))

plot_vol(axes[0,0], 'causal_N_progressive', r'N=(\d+)', "Swarm Size (N)", "Volatility vs. Scaling")
axes[0,0].legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')

plot_vol(axes[0,1], 'causal_M', r'M=(\d+)', "Action Granularity (M)", "Volatility vs. Granularity")
axes[0,1].legend([],[], frameon=False)

plot_vol(axes[1,0], 'causal_noise', r'p=([0-9.]+)', "Perception Noise ($p$)", "Volatility vs. Noise")
axes[1,0].legend([],[], frameon=False)

plot_vol(axes[1,1], 'causal_theta', r'theta=([0-9.]+)', "Streak Threshold ($\\theta$)", "Volatility vs. Streak Req.")
axes[1,1].legend([],[], frameon=False)

plt.tight_layout()
plt.savefig("plots/research/6_action_volatility.png", dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Plot 7: Model Cooperation Heatmaps
# -------------------------------------------------------------------------
print("Generating Plot 7: Model-Specific Heatmaps...")
fig, axes = plt.subplots(1, 4, figsize=(26, 6), gridspec_kw={'width_ratios': [1, 1, 1, 1.2]})
fig.suptitle("Behavioral Profiling: Mean Cooperation Rate by LLM and Parameter", fontsize=24, y=1.05)

def plot_heat(ax, exp, param, title, show_cbar):
    d = df[df['experiment'] == exp]
    if d.empty: return
    pivot = d.pivot_table(index='Model', columns=param, values='true_coop', aggfunc='mean')
    annot_array = pivot.map(lambda v: f"{v:.2f}" if pd.notna(v) else "-").values
    sns.heatmap(pivot, ax=ax, cmap="RdYlGn", vmin=0, vmax=1, annot=annot_array, fmt="", 
                linewidths=1, linecolor='black', cbar=show_cbar, 
                cbar_kws={'label': 'Cooperation Rate'} if show_cbar else None,
                mask=pivot.isnull())
    ax.set_facecolor('#EAEAEA') 
    ax.set_title(title, pad=15)
    ax.set_ylabel("")
    ax.set_xlabel(f"Parameter: {param}")

plot_heat(axes[0], 'causal_N_progressive', 'N', "Scaling (N)", False)
plot_heat(axes[1], 'causal_M', 'M', "Granularity (M)", False)
plot_heat(axes[2], 'causal_noise', 'p_noise', "Noise ($p$)", False)
plot_heat(axes[3], 'causal_theta', 'theta', "Streak ($\\theta$)", True)

plt.tight_layout()
plt.savefig("plots/research/7_model_cooperation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Plot 8: LLM Score (Reward) Over Time
# -------------------------------------------------------------------------
print("Generating Plot 8: Score Over Time...")
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("Evolution of LLM Payoffs (Score) Over Time", fontsize=22, y=1.02)

def plot_score(ax, exp, title):
    d = df[df['experiment'] == exp].copy()
    sns.lineplot(ax=ax, data=d, x='round', y='real_reward', hue='Model', 
                 linewidth=2.5, errorbar=None, palette='bright')
    ax.set_title(title, pad=15)
    ax.set_ylabel("Mean Real Reward (Score)")
    ax.set_xlabel("Game Round")

plot_score(axes[0,0], 'causal_N_progressive', "Score Evolution (Averaged across N)")
axes[0,0].legend(title="LLM Architecture", bbox_to_anchor=(1.05, 1), loc='upper left')

plot_score(axes[0,1], 'causal_M', "Score Evolution (Averaged across M)")
axes[0,1].legend([],[], frameon=False)

plot_score(axes[1,0], 'causal_noise', "Score Evolution (Averaged across Noise $p$)")
axes[1,0].legend([],[], frameon=False)

plot_score(axes[1,1], 'causal_theta', "Score Evolution (Averaged across Streak $\\theta$)")
axes[1,1].legend([],[], frameon=False)

plt.tight_layout()
plt.savefig("plots/research/8_score_over_time.png", dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Plots 9-11: Detailed Score vs Round by Parameter Condition
# -------------------------------------------------------------------------
print("Generating Plots 9-11: Score vs Round by Condition...")

# Plot 9: Scaling
g_n = sns.relplot(
    data=df[df['experiment'] == 'causal_N_progressive'], 
    x='round', y='real_reward', hue='Model', col='N', 
    kind='line', palette='bright', linewidth=2.5, errorbar=None, 
    col_wrap=2, height=4.5, aspect=1.5
)
g_n.fig.suptitle("LLM Score vs Rounds by Swarm Size (N)", y=1.05, fontsize=22)
g_n.set_axis_labels("Game Round", "Real Reward (Score)")
plt.savefig("plots/research/9_score_vs_round_scaling.png", dpi=300, bbox_inches='tight')
plt.close('all')

# Plot 10: Granularity
g_m = sns.relplot(
    data=df[df['experiment'] == 'causal_M'], 
    x='round', y='real_reward', hue='Model', col='M', 
    kind='line', palette='bright', linewidth=2.5, errorbar=None, 
    col_wrap=2, height=4.5, aspect=1.5
)
g_m.fig.suptitle("LLM Score vs Rounds by Action Granularity (M)", y=1.05, fontsize=22)
g_m.set_axis_labels("Game Round", "Real Reward (Score)")
plt.savefig("plots/research/10_score_vs_round_granularity.png", dpi=300, bbox_inches='tight')
plt.close('all')

# Plot 11: Noise
g_p = sns.relplot(
    data=df[df['experiment'] == 'causal_noise'], 
    x='round', y='real_reward', hue='Model', col='p_noise', 
    kind='line', palette='bright', linewidth=2.5, errorbar=None, 
    col_wrap=3, height=4.5, aspect=1.2
)
g_p.fig.suptitle("LLM Score vs Rounds by Perception Noise (p)", y=1.05, fontsize=22)
g_p.set_axis_labels("Game Round", "Real Reward (Score)")
plt.savefig("plots/research/11_score_vs_round_noise.png", dpi=300, bbox_inches='tight')
plt.close('all')

# -------------------------------------------------------------------------
# SMOOTHED PLOTS (8a, 9a, 10a, 11a) - Reduces Visual Noise
# -------------------------------------------------------------------------
print("Generating Smoothed Plots 8a, 9a, 10a, 11a...")
# Sort data chronologically to ensure accurate rolling averages
df_sorted = df.sort_values(by=['experiment', 'condition', 'Model', 'round'])
# Apply a 5-round rolling average to smooth out the intense round-to-round volatility
df_sorted['smooth_reward'] = df_sorted.groupby(['experiment', 'condition', 'Model'])['real_reward'].transform(lambda x: x.rolling(10, min_periods=1).mean())

# Plot 8a: Smoothed Score Over Time
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("Evolution of LLM Payoffs (Smoothed Score) Over Time", fontsize=22, y=1.02)

def plot_score_smooth(ax, exp, title):
    d = df_sorted[df_sorted['experiment'] == exp].copy()
    sns.lineplot(ax=ax, data=d, x='round', y='smooth_reward', hue='Model', 
                 linewidth=3, errorbar=None, palette='bright')
    ax.set_title(title, pad=15)
    ax.set_ylabel("Mean Real Reward (10-Round Rolling Avg)")
    ax.set_xlabel("Game Round")

plot_score_smooth(axes[0,0], 'causal_N_progressive', "Score Evolution (Averaged across N)")
axes[0,0].legend(title="LLM Architecture", bbox_to_anchor=(1.05, 1), loc='upper left')

plot_score_smooth(axes[0,1], 'causal_M', "Score Evolution (Averaged across M)")
axes[0,1].legend([],[], frameon=False)

plot_score_smooth(axes[1,0], 'causal_noise', "Score Evolution (Averaged across Noise $p$)")
axes[1,0].legend([],[], frameon=False)

plot_score_smooth(axes[1,1], 'causal_theta', "Score Evolution (Averaged across Streak $\\theta$)")
axes[1,1].legend([],[], frameon=False)

plt.tight_layout()
plt.savefig("plots/research/8a_score_over_time.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot 9a: Scaling
g_n = sns.relplot(
    data=df_sorted[df_sorted['experiment'] == 'causal_N_progressive'], 
    x='round', y='smooth_reward', hue='Model', col='N', 
    kind='line', palette='bright', linewidth=3, errorbar=None, 
    col_wrap=2, height=4.5, aspect=1.5
)
g_n.fig.suptitle("Smoothed Score vs Rounds by Swarm Size (N)", y=1.05, fontsize=22)
g_n.set_axis_labels("Game Round", "Real Reward (10-Round Avg)")
plt.savefig("plots/research/9a_score_vs_round_scaling.png", dpi=300, bbox_inches='tight')
plt.close('all')

# Plot 10a: Granularity
g_m = sns.relplot(
    data=df_sorted[df_sorted['experiment'] == 'causal_M'], 
    x='round', y='smooth_reward', hue='Model', col='M', 
    kind='line', palette='bright', linewidth=3, errorbar=None, 
    col_wrap=2, height=4.5, aspect=1.5
)
g_m.fig.suptitle("Smoothed Score vs Rounds by Action Granularity (M)", y=1.05, fontsize=22)
g_m.set_axis_labels("Game Round", "Real Reward (10-Round Avg)")
plt.savefig("plots/research/10a_score_vs_round_granularity.png", dpi=300, bbox_inches='tight')
plt.close('all')

# Plot 11a: Noise
g_p = sns.relplot(
    data=df_sorted[df_sorted['experiment'] == 'causal_noise'], 
    x='round', y='smooth_reward', hue='Model', col='p_noise', 
    kind='line', palette='bright', linewidth=3, errorbar=None, 
    col_wrap=3, height=4.5, aspect=1.2
)
g_p.fig.suptitle("Smoothed Score vs Rounds by Perception Noise (p)", y=1.05, fontsize=22)
g_p.set_axis_labels("Game Round", "Real Reward (10-Round Avg)")
plt.savefig("plots/research/11a_score_vs_round_noise.png", dpi=300, bbox_inches='tight')
plt.close('all')

print("High-quality smoothed plots successfully generated in plots/research/")
