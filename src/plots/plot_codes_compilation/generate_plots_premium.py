import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==============================================================================
# ULTRA-PREMIUM AESTHETICS ENGINE
# ==============================================================================
sns.set_theme(style="white", context="notebook")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "Helvetica", "sans-serif"],
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "axes.labelweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#CCCCCC",
    "grid.color": "#E5E5E5",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "axes.grid": True,
    "axes.grid.axis": "y",  # Horizontal gridlines only for clean look
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#DDDDDD",
    "figure.titlesize": 22,
    "figure.titleweight": "bold",
    "lines.linewidth": 3.5,
    "lines.markersize": 10
})

os.makedirs("plots/premium_research", exist_ok=True)

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

# Bespoke High-Contrast Modern Color Palette
MODEL_COLORS = {
    'DeepSeek v3.2': '#FF2A55', # Vibrant Crimson
    'Llama 3.1 8B': '#00B4D8',  # Electric Cyan
    'GPT-OSS 20B': '#06D6A0',   # Mint Green
    'Qwen 3 235B': '#FF9F1C',   # Mango Orange
    'Gemma 3 27B': '#9D4EDD'    # Neon Violet
}

# Apply sort so legends are consistent
df_sorted = df.sort_values(by=['experiment', 'condition', 'Model', 'round'])
df_sorted['smooth_reward'] = df_sorted.groupby(['experiment', 'condition', 'Model'])['real_reward'].transform(lambda x: x.rolling(10, min_periods=1).mean())

# ==============================================================================
# PLOT GENERATION
# ==============================================================================

# 1. Granularity of Trust
print("Generating Plot 1 (Premium): Granularity of Trust...")
plt.figure(figsize=(11, 6.5))
df_m = df[df['experiment'] == 'causal_M']
ax = sns.lineplot(data=df_m, x='M', y='true_coop', hue='Model', palette=MODEL_COLORS, marker='o', errorbar=None)
plt.title("Granularity of Trust: Mean Cooperation by Action Space", pad=20, color="#2B2D42")
plt.ylabel("Mean Cooperation Rate", color="#8D99AE")
plt.xlabel("Action Granularity (M)", color="#8D99AE")
plt.ylim(-0.05, 1.05)
sns.despine(trim=True, offset=5)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.savefig("plots/premium_research/1_granularity_of_trust.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Noise Threshold
print("Generating Plot 2 (Premium): Noise Threshold...")
plt.figure(figsize=(11, 6.5))
df_noise = df[df['experiment'] == 'causal_noise']
ax = sns.lineplot(data=df_noise, x='p_noise', y='true_coop', hue='Model', palette=MODEL_COLORS, marker='o', errorbar=None)
plt.title("The Noise Threshold: Cooperation Collapse Under Error", pad=20, color="#2B2D42")
plt.ylabel("Mean Cooperation Rate", color="#8D99AE")
plt.xlabel("Perception Noise ($p$)", color="#8D99AE")
plt.ylim(-0.05, 1.05)
sns.despine(trim=True, offset=5)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.savefig("plots/premium_research/2_noise_threshold.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Scaling Tragedy
print("Generating Plot 3 (Premium): Scaling Tragedy...")
plt.figure(figsize=(12, 7))
df_n = df[df['experiment'] == 'causal_N_progressive']
ax = sns.barplot(data=df_n, x='N', y='true_coop', hue='Model', palette=MODEL_COLORS, edgecolor='white', linewidth=1.5, alpha=0.9, errorbar=None)
plt.title("The Tragedy of the Commons: Mean Cooperation by Swarm Size", pad=20, color="#2B2D42")
plt.ylabel("Mean Cooperation Rate", color="#8D99AE")
plt.xlabel("Number of Agents (N)", color="#8D99AE")
plt.ylim(0, 1.15)
sns.despine(left=True) # Very clean floating bars
ax.grid(axis='y', linestyle='-', alpha=0.3)
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', padding=5, fontsize=10, fontweight='bold', color='#444444', rotation=0)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.savefig("plots/premium_research/3_scaling_tragedy.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. Streak Incentive
print("Generating Plot 4 (Premium): Streak Incentive...")
plt.figure(figsize=(14, 7))
df_th = df[df['experiment'] == 'causal_theta']
ax = sns.violinplot(data=df_th, x='theta', y='real_reward', hue='Model', inner="quartile", palette=MODEL_COLORS, linewidth=2, alpha=0.8)
plt.title("Streak Incentive Dilemma: Reward Distribution by Threshold", pad=20, color="#2B2D42")
plt.ylabel("Realized Utility (Payoff)", color="#8D99AE")
plt.xlabel("Streak Cooperation Requirement ($\\theta$)", color="#8D99AE")
sns.despine(trim=True, offset=5)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.savefig("plots/premium_research/4_streak_incentive.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. Deviation from Baseline
print("Generating Plot 5 (Premium): Baseline Deviation...")
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("Model-Specific Deviation from Cooperative Baseline", fontsize=24, fontweight='heavy', y=1.02, color="#2B2D42")

def plot_dev_premium(ax, exp, x_col, title):
    d = df[df['experiment'] == exp].copy()
    if exp == 'causal_N_progressive': d = d[d['round'] > 5]
    sns.lineplot(ax=ax, data=d, x=x_col, y='Deviation', hue='Model', palette=MODEL_COLORS, marker='o', errorbar=None)
    ax.set_title(title, pad=15, fontweight='bold', color="#4A4E69")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Deviation (1.0 - Mean Cooperation)", color="#8D99AE")
    ax.axhline(0, color='#FF5A5F', linestyle='--', alpha=0.7, linewidth=2, zorder=0)

plot_dev_premium(axes[0,0], 'causal_N_progressive', 'N', "Deviation vs. Scaling Swarm Size (N)")
axes[0,0].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plot_dev_premium(axes[0,1], 'causal_M', 'M', "Deviation vs. Action Granularity (M)")
axes[0,1].legend([],[], frameon=False)
plot_dev_premium(axes[1,0], 'causal_noise', 'p_noise', "Deviation vs. Perception Noise ($p$)")
axes[1,0].legend([],[], frameon=False)
plot_dev_premium(axes[1,1], 'causal_theta', 'theta', "Deviation vs. Streak Threshold ($\\theta$)")
axes[1,1].legend([],[], frameon=False)

sns.despine(fig=fig, trim=True, offset=5)
plt.tight_layout()
plt.savefig("plots/premium_research/5_baseline_deviation.png", dpi=300, bbox_inches='tight')
plt.close()

# 6. Action Volatility
print("Generating Plot 6 (Premium): Action Volatility...")
volatility = df.groupby(['experiment', 'condition', 'Model'])['action_changed'].mean().reset_index()
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("Strategy Volatility: Probability of Changing Action", fontsize=24, fontweight='heavy', y=1.02, color="#2B2D42")

def plot_vol_premium(ax, exp, regex, x_label, title):
    d = volatility[volatility['experiment'] == exp].copy()
    d['x_val'] = d['condition'].str.extract(regex).astype(float)
    sns.barplot(ax=ax, data=d, x='x_val', y='action_changed', hue='Model', palette=MODEL_COLORS, edgecolor='white', linewidth=1.5, alpha=0.9)
    ax.set_title(title, pad=15, fontweight='bold', color="#4A4E69")
    ax.set_ylabel("Prob. of Action Change", color="#8D99AE")
    ax.set_xlabel(x_label, color="#8D99AE")
    ax.set_ylim(0, max(0.4, d['action_changed'].max() * 1.2))

plot_vol_premium(axes[0,0], 'causal_N_progressive', r'N=(\d+)', "Swarm Size (N)", "Volatility vs. Scaling")
axes[0,0].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plot_vol_premium(axes[0,1], 'causal_M', r'M=(\d+)', "Action Granularity (M)", "Volatility vs. Granularity")
axes[0,1].legend([],[], frameon=False)
plot_vol_premium(axes[1,0], 'causal_noise', r'p=([0-9.]+)', "Perception Noise ($p$)", "Volatility vs. Noise")
axes[1,0].legend([],[], frameon=False)
plot_vol_premium(axes[1,1], 'causal_theta', r'theta=([0-9.]+)', "Streak Threshold ($\\theta$)", "Volatility vs. Streak Req.")
axes[1,1].legend([],[], frameon=False)

sns.despine(fig=fig, left=True)
plt.tight_layout()
plt.savefig("plots/premium_research/6_action_volatility.png", dpi=300, bbox_inches='tight')
plt.close()

# 8a. Smoothed Score Over Time
print("Generating Plot 8a (Premium): Smoothed Score Over Time...")
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("Evolution of LLM Payoffs (Smoothed) Over Time", fontsize=24, fontweight='heavy', y=1.02, color="#2B2D42")

def plot_score_smooth_premium(ax, exp, title):
    d = df_sorted[df_sorted['experiment'] == exp].copy()
    sns.lineplot(ax=ax, data=d, x='round', y='smooth_reward', hue='Model', palette=MODEL_COLORS, errorbar=None)
    ax.set_title(title, pad=15, fontweight='bold', color="#4A4E69")
    ax.set_ylabel("Mean Real Reward (10-Round Avg)", color="#8D99AE")
    ax.set_xlabel("Game Round", color="#8D99AE")

plot_score_smooth_premium(axes[0,0], 'causal_N_progressive', "Score Evolution (Scaling)")
axes[0,0].legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plot_score_smooth_premium(axes[0,1], 'causal_M', "Score Evolution (Granularity)")
axes[0,1].legend([],[], frameon=False)
plot_score_smooth_premium(axes[1,0], 'causal_noise', "Score Evolution (Noise)")
axes[1,0].legend([],[], frameon=False)
plot_score_smooth_premium(axes[1,1], 'causal_theta', "Score Evolution (Streak)")
axes[1,1].legend([],[], frameon=False)

sns.despine(fig=fig, trim=True, offset=5)
plt.tight_layout()
plt.savefig("plots/premium_research/8a_score_over_time.png", dpi=300, bbox_inches='tight')
plt.close()

# 7. Model Cooperation Heatmaps
print("Generating Plot 7 (Premium): Model-Specific Heatmaps...")
fig, axes = plt.subplots(1, 4, figsize=(26, 6.5), gridspec_kw={'width_ratios': [1, 1, 1, 1.25]})
fig.suptitle("Behavioral Profiling: Mean Cooperation Rate by LLM and Parameter", fontsize=24, fontweight='heavy', y=1.05, color="#2B2D42")

MODEL_ORDER = ['GPT-OSS 20B', 'Llama 3.1 8B', 'DeepSeek v3.2', 'Gemma 3 27B', 'Qwen 3 235B']

def plot_heat_premium(ax, exp, param, title, show_cbar):
    d = df[df['experiment'] == exp]
    if d.empty: return
    pivot = d.pivot_table(index='Model', columns=param, values='true_coop', aggfunc='mean')
    # Reindex to force the progressive staircase visual
    pivot = pivot.reindex(MODEL_ORDER)
    annot_array = pivot.map(lambda v: f"{v:.2f}" if pd.notna(v) else "-").values
    sns.heatmap(pivot, ax=ax, cmap="RdYlGn", vmin=0, vmax=1, annot=annot_array, fmt="", 
                linewidths=2, linecolor='white', cbar=show_cbar, 
                annot_kws={'fontweight': 'bold', 'fontsize': 12},
                cbar_kws={'label': 'Cooperation Rate'} if show_cbar else None,
                mask=pivot.isnull())
    ax.set_facecolor('#F8F9FA') # Premium very light grey for NaN
    ax.set_title(title, pad=15, fontweight='bold', color="#4A4E69")
    ax.set_ylabel("")
    ax.set_xlabel(f"Parameter: {param}", color="#8D99AE", fontweight='bold')
    ax.tick_params(axis='y', rotation=0)

plot_heat_premium(axes[0], 'causal_N_progressive', 'N', "Scaling (N)", False)
plot_heat_premium(axes[1], 'causal_M', 'M', "Granularity (M)", False)
plot_heat_premium(axes[2], 'causal_noise', 'p_noise', "Noise ($p$)", False)
plot_heat_premium(axes[3], 'causal_theta', 'theta', "Streak ($\\theta$)", True)

plt.tight_layout()
plt.savefig("plots/premium_research/7_model_cooperation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# 9a. Smoothed Score vs Round by Scaling
print("Generating Plot 9a (Premium): Score vs Scaling...")
g_n = sns.relplot(
    data=df_sorted[df_sorted['experiment'] == 'causal_N_progressive'], 
    x='round', y='smooth_reward', hue='Model', col='N', palette=MODEL_COLORS,
    kind='line', linewidth=3.5, errorbar=None, col_wrap=2, height=4.5, aspect=1.5
)
g_n.fig.suptitle("Smoothed Score Evolution by Swarm Size (N)", y=0.98, fontsize=24, fontweight='heavy', color="#2B2D42")
g_n.set_axis_labels("Game Round", "Real Reward (10-Round Avg)", color="#8D99AE")
g_n.set_titles("Swarm Size: N = {col_name}", color="#4A4E69", fontweight='bold')
sns.despine(trim=True, offset=5)
g_n.tight_layout()
g_n.fig.subplots_adjust(top=0.85)
plt.savefig("plots/premium_research/9a_score_vs_round_scaling.png", dpi=300, bbox_inches='tight')
plt.close('all')

# 10a. Smoothed Score vs Round by Granularity
print("Generating Plot 10a (Premium): Score vs Granularity...")
g_m = sns.relplot(
    data=df_sorted[df_sorted['experiment'] == 'causal_M'], 
    x='round', y='smooth_reward', hue='Model', col='M', palette=MODEL_COLORS,
    kind='line', linewidth=3.5, errorbar=None, col_wrap=2, height=4.5, aspect=1.5
)
g_m.fig.suptitle("Smoothed Score Evolution by Action Granularity (M)", y=0.98, fontsize=24, fontweight='heavy', color="#2B2D42")
g_m.set_axis_labels("Game Round", "Real Reward (10-Round Avg)", color="#8D99AE")
g_m.set_titles("Granularity: M = {col_name}", color="#4A4E69", fontweight='bold')
sns.despine(trim=True, offset=5)
g_m.tight_layout()
g_m.fig.subplots_adjust(top=0.85)
plt.savefig("plots/premium_research/10a_score_vs_round_granularity.png", dpi=300, bbox_inches='tight')
plt.close('all')

# 11a. Smoothed Score vs Round by Noise
print("Generating Plot 11a (Premium): Score vs Noise...")
g_p = sns.relplot(
    data=df_sorted[df_sorted['experiment'] == 'causal_noise'], 
    x='round', y='smooth_reward', hue='Model', col='p_noise', palette=MODEL_COLORS,
    kind='line', linewidth=3.5, errorbar=None, col_wrap=3, height=4.5, aspect=1.2
)
g_p.fig.suptitle("Smoothed Score Evolution by Perception Noise (p)", y=0.98, fontsize=24, fontweight='heavy', color="#2B2D42")
g_p.set_axis_labels("Game Round", "Real Reward (10-Round Avg)", color="#8D99AE")
g_p.set_titles("Noise: p = {col_name}", color="#4A4E69", fontweight='bold')
sns.despine(trim=True, offset=5)
g_p.tight_layout()
g_p.fig.subplots_adjust(top=0.85)
plt.savefig("plots/premium_research/11a_score_vs_round_noise.png", dpi=300, bbox_inches='tight')
plt.close('all')

print("ULTRA-PREMIUM plots successfully generated in plots/premium_research/")
