# POST-HOC / EXPLORATORY saved-log analysis

No episode runner, model server, GPU or judge is used. Task 0 is a hard gate; any failed reproduction anchor exits nonzero and stops the sequence. Missing required schema fields also stop execution. See SCHEMA.md and PLAN.md for definitions and statistical limitations, RESULTS.md for real results, and results/artifact_manifest.json for hashes and actual package versions.

From the repository root:

```powershell
python -m venv analysis/.venv
analysis/.venv/Scripts/python -m pip install -r analysis/requirements.txt
analysis/.venv/Scripts/python analysis/run_all.py
```

Alternatively, with an existing environment containing the required dependencies:

```text
python analysis/run_all.py --input results/server_copies/final_main_20261010/main_complete_20261010/main.jsonl --export-dir results/analysis/main_h5_coding_20261010
```

Local execution used Python/package versions saved in package_versions.json (statsmodels 0.15.0 locally; stable 0.14.5 is pinned for Sharanga). Six-model Holm is checked against statsmodels; the gate independently verifies all published statistical anchors. Sharanga's finalizer records its own actual environment. Do not treat version differences as absent; use the source hash and reproduction gate to check results on either environment.

Individual tasks use `--input PATH`; Task 4 additionally uses `--export-dir PATH`. Run in numeric order and record task checkpoints before proceeding. `run_all.py` runs Tasks 0–4 sequentially then builds paper inserts and the final manifest; it does not commit or publish. Reruns replace generated analysis artifacts, so preserve any manually edited report and human-labeled file separately. The labeling export is generated with blank labels on every rerun. Private keys must not be given to blinded raters. `frozen_parser_v2.py` preserves the paper's exact extraction/validation functions without importing model-call code; its source digest is recorded because Sharanga's older experiment working tree lacks the newer function. No server experiment files are replaced.

On Sharanga, from `/home/jagatsesh/f20220604_ts/agentic-game-lab`:

```bash
analysis/.venv_posthoc_20261010/bin/python analysis/submit_cpu.py \
  --input results/analysis/main_complete_20261010/main.jsonl \
  --export-dir analysis/explanation_export_20261010
```

This submits four-CPU, 4 GB, ten-minute jobs for each task plus final packaging, connected by Slurm `afterok` dependencies. Four CPUs are the verified minimum for Sharanga's compute/cpulimit QOS; big_compute requires 128 and is avoided. Numerical-library threads are capped at one. A failed task prevents later tasks from executing. Slurm handles availability and dispatch; no GPU polling loop or inference submission is needed. Receipts and stdout are in analysis/logs/TIMESTAMP/. No existing jobs are cancelled or modified. The isolated analysis venv inherits available CPU packages from `mas`; pinned requirements are installed only into this venv.

Outputs: CSVs in results/, booktabs LaTeX in results/latex/, PDFs in figures/. The main summary-grid LaTeX is compact and paper-ready; large per-episode/per-decision tables use longtable and are supplementary audit records. LaTeX requires booktabs and longtable. Cooperation values are reported in percent/pp, fractions are explicitly named. CSVs retain full numeric precision. Plotting uses the headless Agg backend.

Human validation, after labeling a separate copy:

```text
python analysis/validation/score_validation.py PATH_TO_LABELED_COPY.csv
```

The script accepts INCONSISTENT / CONSISTENT / UNCLEAR, reports unlabeled and unclear cases, and does not fabricate labels. Population recall uses sampling weights; naive recall on a balanced flagged/unflagged sample is invalid. Wilson intervals are descriptive and weighted Wilson intervals approximate; consult episode bootstrap intervals in the prevalence outputs. No optional judge runs unless API access and explicit authorization are later supplied.

Bootstrap draws: 5,000 for episode analyses; 20,000 for each parametric Q null. Seed 20261010 with deterministic named streams. Nonestimable draws and cells are reported, never silently removed or regularized. Analysis timestamps use UTC in scheduler receipts; study date is 2026-10-10 in Asia/Calcutta.
