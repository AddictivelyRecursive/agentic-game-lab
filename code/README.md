# Code guide

Run commands from the repository root using `python code/SCRIPT.py`.

- `reproduce_results.py`: verify released primary tables using the released episode summaries; no raw logs or inference.
- `fit_glmm_and_curve.py`: original primary statistical methods, taking a complete study JSONL.
- `run_saved_log_analysis.py`: original saved-log gate and exploratory reliability, heterogeneity, persistence, and narrative summaries. Uses the included anonymized compressed logs by default; writes to `results/generated/`.
- `run_main.py` / `run_pilot.py`: collection using `experiments/study_common.py`, configuration templates, and separately served model endpoints.
- `calculate_h4_power.py` / `calculate_h4_per_model_power.py`: power functions required by study validation; historical H4 is the paper's H3.
- `src/game_engine/env/`: action mapping, average-opponent payoff, streak bonus, drift, observation construction, and simultaneous simulation.
- `src/AI_Agent/prompts/`: original system, scaffold, and repair templates.
- `src/AI_Agent/agent/`: clients, JSON extraction, validation, repair, fallback, and logging.

The six-checkpoint registry is restricted to this paper. Legacy experiment suites, optional explanation coding, cluster scheduling, and deployment utilities are excluded. Reorganization changes import paths and source fingerprints; the retained scientific estimators and prompt templates are unchanged.
