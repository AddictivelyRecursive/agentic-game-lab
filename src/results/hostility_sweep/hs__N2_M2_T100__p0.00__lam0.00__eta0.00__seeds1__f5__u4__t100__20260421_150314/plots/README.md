# Hostility Sweep Results

Auto-generated report for run:

`hs__N2_M2_T100__p0.00__lam0.00__eta0.00__seeds1__f5__u4__t100__20260421_150314`

## Experiment summary

- **Setup:** 2-player hostility sweep
- **Focal LLMs:** 5
- **URND opponents:** 4
- **Seeds:** 1
- **Rounds per match:** 100
- **Action levels (M):** 2
- **Perception noise:** 0.0
- **Streak lambda:** 0.0
- **Drift eta:** 0.0

## Interpretation note

This run contains a single seed, so all figures should be interpreted as descriptive summaries rather than confidence-interval-based estimates.

## Key findings

- **Best average payoff:** Gemma-3-27B with mean reward **0.618** per round.
- **Most cooperative overall:** DeepSeek-V3.2 with mean cooperation **0.400**.
- **Most retaliatory:** Gemma-3-27B with retaliatory score **0.82**.
- **Most forgiving:** DeepSeek-V3.2 with forgiving score **0.28**.
- **Most switch-heavy policy:** DeepSeek-V3.2 with switch rate **0.13**.
- **Most responsive to opponent friendliness:** DeepSeek-V3.2 (cooperation slope **1.340** vs opponent p).
- **Least responsive / flattest reaction curve:** GPT-OSS-20B (slope **1.130**).

## Main figures

### Cooperation vs hostility
![Cooperation vs hostility](./cooperation_vs_hostility_pretty.png)

### Payoff vs hostility
![Payoff vs hostility](./payoff_vs_hostility_pretty.png)

### Behavioral profile heatmap
![Behavioral profile heatmap](./heatmap_behavior_pretty.png)

### Reward landscape heatmap
![Reward landscape heatmap](./heatmap_payoff_pretty.png)

### Cooperation landscape heatmap
![Cooperation landscape heatmap](./heatmap_cooperation_pretty.png)

### Cooperation–reward tradeoff
![Cooperation–reward tradeoff](./scatter_cooperation_vs_payoff_pretty.png)

### Round-wise cooperation trajectories
![Round-wise cooperation trajectories](./trajectories_small_multiples_pretty.png)


## Aggregate performance by model

| Model | Mean coop | Mean reward | Mean payoff gap | Nice | Retaliatory | Forgiving | Switch rate | Matches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma-3-27B | 0.318 | 0.618 | 0.450 | 1.00 | 0.82 | 0.15 | 0.09 | 4 |
| Qwen3-235B | 0.320 | 0.615 | 0.443 | 1.00 | 0.82 | 0.15 | 0.09 | 4 |
| Llama-3.1-8B | 0.328 | 0.608 | 0.420 | 1.00 | 0.81 | 0.17 | 0.09 | 4 |
| GPT-OSS-20B | 0.365 | 0.570 | 0.307 | 1.00 | 0.81 | 0.25 | 0.13 | 4 |
| DeepSeek-V3.2 | 0.400 | 0.535 | 0.202 | 1.00 | 0.79 | 0.28 | 0.13 | 4 |

## Detailed results by model × opponent hostility

| Model | Opponent p | Coop rate | Avg reward | Payoff gap | Nice | Retaliatory | Forgiving | Switch rate | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek-V3.2 | 0.2 | 0.060 | 0.340 | 0.420 | 1.00 | 0.97 | 0.11 | 0.07 | 1 |
| DeepSeek-V3.2 | 0.4 | 0.190 | 0.490 | 0.450 | 1.00 | 0.92 | 0.22 | 0.13 | 1 |
| DeepSeek-V3.2 | 0.6 | 0.500 | 0.580 | 0.120 | 1.00 | 0.76 | 0.41 | 0.25 | 1 |
| DeepSeek-V3.2 | 0.8 | 0.850 | 0.730 | -0.180 | 1.00 | 0.50 | 0.38 | 0.08 | 1 |
| Gemma-3-27B | 0.2 | 0.030 | 0.370 | 0.510 | 1.00 | 0.99 | 0.05 | 0.03 | 1 |
| Gemma-3-27B | 0.4 | 0.150 | 0.530 | 0.570 | 1.00 | 0.92 | 0.17 | 0.11 | 1 |
| Gemma-3-27B | 0.6 | 0.310 | 0.770 | 0.690 | 1.00 | 0.82 | 0.18 | 0.15 | 1 |
| Gemma-3-27B | 0.8 | 0.780 | 0.800 | 0.030 | 1.00 | 0.55 | 0.21 | 0.07 | 1 |
| GPT-OSS-20B | 0.2 | 0.090 | 0.310 | 0.330 | 1.00 | 0.97 | 0.26 | 0.11 | 1 |
| GPT-OSS-20B | 0.4 | 0.200 | 0.480 | 0.420 | 1.00 | 0.92 | 0.26 | 0.15 | 1 |
| GPT-OSS-20B | 0.6 | 0.390 | 0.690 | 0.450 | 1.00 | 0.80 | 0.26 | 0.19 | 1 |
| GPT-OSS-20B | 0.8 | 0.780 | 0.800 | 0.030 | 1.00 | 0.55 | 0.21 | 0.07 | 1 |
| Llama-3.1-8B | 0.2 | 0.030 | 0.370 | 0.510 | 1.00 | 0.99 | 0.05 | 0.03 | 1 |
| Llama-3.1-8B | 0.4 | 0.150 | 0.530 | 0.570 | 1.00 | 0.92 | 0.17 | 0.11 | 1 |
| Llama-3.1-8B | 0.6 | 0.310 | 0.770 | 0.690 | 1.00 | 0.82 | 0.18 | 0.15 | 1 |
| Llama-3.1-8B | 0.8 | 0.820 | 0.760 | -0.090 | 1.00 | 0.50 | 0.27 | 0.08 | 1 |
| Qwen3-235B | 0.2 | 0.030 | 0.370 | 0.510 | 1.00 | 0.99 | 0.05 | 0.03 | 1 |
| Qwen3-235B | 0.4 | 0.160 | 0.520 | 0.540 | 1.00 | 0.92 | 0.17 | 0.11 | 1 |
| Qwen3-235B | 0.6 | 0.310 | 0.770 | 0.690 | 1.00 | 0.82 | 0.18 | 0.15 | 1 |
| Qwen3-235B | 0.8 | 0.780 | 0.800 | 0.030 | 1.00 | 0.55 | 0.21 | 0.07 | 1 |

## Files generated

- `match_level_summary.csv`
- `grouped_summary.csv`
- `cooperation_vs_hostility_pretty.png`
- `payoff_vs_hostility_pretty.png`
- `heatmap_behavior_pretty.png`
- `heatmap_payoff_pretty.png`
- `heatmap_cooperation_pretty.png`
- `scatter_cooperation_vs_payoff_pretty.png`
- `trajectories_small_multiples_pretty.png`

---
Generated automatically by `plot_hostility_sweep_pretty.py`
