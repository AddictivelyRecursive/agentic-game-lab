# Released results

These are derived outputs from the original completed main-study dataset, not new experiments. Source SHA-256 and per-file digests are recorded in `MANIFEST.json`. No pilot episodes are included.

## Primary statistics

| File | Contents |
|---|---|
| `primary/00_episode_outcomes.csv` | 376 episode summaries: model, scaffold, N, seed, cooperation, early/late outcomes, repair and fallback rates |
| `primary/00_cells.csv` | 48 cell means, episode counts, and sample variances |
| `primary/00_effects.csv` | Six endpoint-scale linear interactions, Welch SE/df/CI, marginal and Holm-adjusted p-values |
| `primary/00_heterogeneity.csv` | Inverse-variance Q, I?, and descriptive pooled estimate |
| `primary/confirmatory_results.json` | Full primary analysis, mixed-model diagnostics, and H2 model comparison |
| `primary/descriptive_model_slopes.csv` | Descriptive model-by-scaffold slopes |
| `primary/published_model_outcomes.csv` | Rounded paper table, including per-seat per-round rewards |
| `primary/00_reproduction.csv` | Historical saved-log reproduction checks against published rounded anchors |

`python code/reproduce_results.py` recomputes primary statistics from the episode summaries. Cooperation and rate columns ending in `_percent` are percentages; slope and interaction columns are percentage points unless explicitly documented otherwise. In `00_effects.csv`, D, SE, and CI bounds are percentage points. Variance in `00_cells.csv` is squared percentage points. In `confirmatory_results.json`, cooperation coefficients use proportions; multiply by 100 for percentage points. Historical H4 labels denote the paper's H3 interaction.

## Persistence and figures

`persistence/round_curves.csv` contains group-size-balanced model/scaffold round means. `event_summary.csv` distinguishes episode streak breaks, recoveries, and terminal-zero suffixes. `keyword_summary.csv` contains literal returned-explanation keyword counts with explicit denominators. Event medians condition on occurrence. These are descriptive summaries, not causal or confirmatory temporal tests.

The three PDFs in `figures/` are the cooperation, dynamics, and group-size-specific trajectory figures used in the paper. Their underlying raw traces are not distributed.

## Provenance and limits

Saved-analysis files were copied without numerical changes and normalized to LF line endings. The published model-outcomes table preserves the paper?s displayed rounding. `MANIFEST.json` identifies their original analysis families and hashes the released bytes. The historical reproduction table is an archived result, not a substitute for running the current verification command.

Raw per-seat actions, prompts, outputs, repairs, and explanations are not included. This release permits verification of primary statistics from derived episode summaries, not independent verification of those summaries against raw decisions. Qwen's trajectory is pipeline-confounded; GPT-OSS uncertainty is weak. Larger groups contribute more decisions to lexical counts. Do not infer hidden reasoning or causal end-game effects from these summaries.
