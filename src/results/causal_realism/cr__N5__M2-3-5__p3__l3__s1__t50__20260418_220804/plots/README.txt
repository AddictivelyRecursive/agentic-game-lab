Causal realism plotting outputs
================================

Run id: cr__N5__M2-3-5__p3__l3__s1__t50__20260418_220804
Rounds per match: 50
Lineup: ['always_cooperate', 'always_defect', 'graded_tft', 'gpt4o_mini', 'grok_41_fast']

Files
-----
01_agent_mean_coop_heatmaps.png      : mean true cooperation for every agent across (p, lambda), faceted by M
02_agent_mean_reward_heatmaps.png    : mean reward for every agent across (p, lambda), faceted by M
03_agent_switch_rate_heatmaps.png    : action-switching rate for every agent across (p, lambda), faceted by M
04_llm_vs_gtft_heatmaps.png          : how often each LLM matched graded_tft's action each round
05_mean_coop_vs_M_small_multiples.png: decision-granularity effect under every (p, lambda) cell
06_selected_round_dynamics.png       : representative per-round cooperation trajectories
07_group_dynamics_grid_M*.png        : 3x3 grid of group dynamics for each M
08_llm_action_index_profiles.png     : distribution of raw action indices used by each LLM

CSV outputs
-----------
round_level_summary.csv
agent_level_summary.csv
match_level_summary.csv
