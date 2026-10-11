# Cooperation Is Not One Number

Anonymized logs, code, and derived results for **Cooperation Is Not One Number: Scaffold, Group Size, and Persistence in Repeated Social Dilemmas Among LLM Agents**.

We evaluate 376 repeated-game episodes across six LLMs, two prompt scaffolds, and group sizes 2-5. The analysis distinguishes cooperation level, group-size response, persistence, and execution reliability.

## Repository structure

```text
code/
  src/              Agent pipeline, prompt templates, and game simulator
  experiments/      Shared study collection and validation functions
  analysis/         Saved-log reproduction and exploratory analyses
  run_main.py       Main-study collection entry point
  run_pilot.py      Pilot collection entry point
  fit_glmm_and_curve.py   Primary statistical analysis
  reproduce_results.py   Verify the included derived result tables
configs/
  design.json       Model allocation, estimands, and provenance
  main_study/       Environment, generation settings, and run templates
results/
  logs/             Compressed episode logs and saved agent inputs
  primary/          Episode summaries, cell statistics, and primary estimates
  persistence/      Round trajectories, event timing, and keyword summaries
  figures/          Three figures used in the paper
  MANIFEST.json     Source provenance and SHA-256 hashes of released results
tests/
  review/           Offline parser, simulator, inference-boundary, and statistics tests
```

## Verify the included results

Python 3.10 or later is required. From the repository root:

```bash
python -m pip install -r requirements.txt
python code/reproduce_results.py
python -m unittest discover -s tests/review
```

The reproduction script verifies all 376 episode summaries and 48 cells, cooperation means and variances, six model interactions and Welch intervals, Holm-adjusted tests, Q/I-squared heterogeneity, the pooled H1 coefficient, and H2 AIC/BIC. Numerical tolerances allow small solver differences. It does not call models or require a GPU.

## Run collection or saved-log analyses

```bash
python code/run_main.py --help
python code/fit_glmm_and_curve.py --help
python code/run_saved_log_analysis.py
```

Collection requires separately served model endpoints and completed deployment, power, and run-approval templates. No API credentials, model weights, or cluster-specific setup is included. `configs/design.json` records the six checkpoint allocations and serving settings. Historical hypothesis labels in code use H4 for the interaction labeled H3 in the paper.

Saved-log analyses use `results/logs/main.jsonl.gz` by default and validate archive and uncompressed SHA-256 digests. They write to the ignored `results/generated/` directory. The release includes all 376 episode logs (39,480 decisions), returned outputs and explanations, repair/fallback records, round states, and all 39,480 saved agent inputs. Orchestration records containing deployment paths and approval metadata are removed; scientific fields and model text are unchanged. See [results documentation](results/README.md) for units and provenance. The command reconstructs primary statistics, reliability summaries, per-round curves, streak-break/terminal-zero event timing, and literal explanation keyword counts without model calls.

## Study scope

Each episode has 30 rounds, five graded actions, zero perception noise, history-only memory, and homogeneous self-play. Structured prompts add the constant immediate-payoff ranking; free-form prompts omit it. Both disclose the horizon. Full defection is the execution fallback. Replication is unequal and some cells have zero observed variance; GPT-OSS has only two episodes per cell. Temporal and explanation summaries are descriptive.

Author notes, manuscript drafts, unrelated experiments, cluster scripts, scheduler receipts, private labeling keys are outside this shared artifact. No license grant is implied; a general public license has not yet been selected.
