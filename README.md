# Cooperation Is Not One Number

Code accompanying *Cooperation Is Not One Number: Scaffold, Group Size, and Persistence in Repeated Social Dilemmas Among LLM Agents*.

The study evaluates 376 episodes of homogeneous self-play across six models, group sizes 2?5, and two prompt scaffolds. Each episode lasts 30 rounds. Replication per model-by-scaffold-by-group-size cell is 14 / 4 / 8 / 3 / 16 / 2 for Llama / Qwen / Gemma / Mistral / DeepSeek / GPT-OSS.

## Setup

Use Python 3.10 or later:

```bash
python -m pip install -r requirements-study.txt
python run_main.py --help
python fit_glmm_and_curve.py --help
```

Local model serving requires a separately configured OpenAI-compatible endpoint. No hosted API credentials or model weights are included. See `experiments/main_study/` for configuration and serving templates. Review run-approval and power gates before collecting data; these commands do not initiate collection.

## Contents

- `src/AI_Agent/`: clients, prompts, decision extraction, and repair pipeline.
- `src/game_engine/`: graded cooperation environment and experiment runners.
- `experiments/`: study configuration, generation settings, and collection machinery.
- Root Python scripts: collection, power planning, and primary statistical analysis.
- `analysis/`: post-hoc saved-log analyses, including persistence and reliability.
- `tests/`: simulator, extraction, study-planning, and pipeline checks.
- `docs/`: study specification, workflow, and dated internal amendments.
- `cluster/`: reusable batch lifecycle and job-planning helpers.
- `Thesis/`: anonymous manuscript sources, bibliography, tables, and figures.

Raw main/pilot logs, private labeling keys, scheduler receipts, deployment snapshots, unrelated historical plots, and author working notes are excluded. Saved-data analyses require the original data and cannot reproduce the reported results from this code-only release. No new inference is required for those analyses. The manuscript retains its existing data-availability statement until an anonymous release is actually accessible.

## Checks and review packaging

```bash
python -m unittest discover -s tests
python tools/export_review.py
```

The exporter writes `dist/acl-review.zip` from an explicit allowlist, with file hashes and without Git history. See [review release notes](docs/REVIEW_RELEASE.md). The repository's original owner URL and commit history are identifying and must not be used as the anonymous review link.
