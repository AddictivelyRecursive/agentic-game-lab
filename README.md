# Cooperation Is Not One Number

Code and derived results for **Cooperation Is Not One Number: Scaffold, Group Size, and Persistence in Repeated Social Dilemmas Among LLM Agents**.

We evaluate 376 repeated-game episodes across six LLMs, two prompt scaffolds, and group sizes 2?5. The analysis distinguishes cooperation level, group-size response, persistence, and execution reliability.

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

The reproduction script verifies all 376 episode summaries and 48 cells, cooperation means and variances, six model interactions and Welch intervals, Holm-adjusted tests, Q/I? heterogeneity, the pooled H1 coefficient, and H2 AIC/BIC. Numerical tolerances allow small solver differences. It does not call models or require a GPU.

## Run collection or saved-log analyses

```bash
python code/run_main.py --help
python code/fit_glmm_and_curve.py --help
python code/run_saved_log_analysis.py --input /path/to/original/main.jsonl
```

Collection requires separately served model endpoints and completed deployment, power, and run-approval templates. No API credentials, model weights, or cluster-specific setup is included. `configs/design.json` records the six checkpoint allocations and serving settings. Historical hypothesis labels in code use H4 for the interaction labeled H3 in the paper.

Saved-log analyses require the original main JSONL and validate its SHA-256 digest. They write to the ignored `results/generated/` directory. The raw logs are **not included**, so action-level reliability, event timing, and explanation counts cannot be independently reconstructed from this release alone. The included episode summaries support the primary statistical checks above; see [results documentation](results/README.md) for units and limitations.

## Study scope

Each episode has 30 rounds, five graded actions, zero perception noise, history-only memory, and homogeneous self-play. Structured prompts add the constant immediate-payoff ranking; free-form prompts omit it. Both disclose the horizon. Full defection is the execution fallback. Replication is unequal and some cells have zero observed variance; GPT-OSS has only two episodes per cell. Temporal and explanation summaries are descriptive.

Author notes, manuscript drafts, unrelated experiments, cluster scripts, scheduler receipts, private labeling keys, and raw model outputs are outside this shared artifact. No license grant is implied; a general public license has not yet been selected.
