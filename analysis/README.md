# Saved-log analysis

These analyses are post-hoc and exploratory. They do not call models or launch GPU jobs.

```bash
python -m pip install -r analysis/requirements.txt
python analysis/run_all.py --input /path/to/main.jsonl --export-dir results/explanation_export
python analysis/narrative_support_20261011.py --help
```

The original main dataset is not included. The reproduction gate verifies its exact SHA-256 digest and published statistical anchors before later tasks run. An arbitrary replacement dataset will fail this gate. See `SCHEMA.md` for the input schema. `common.py` records model allocations and the expected source digest.

Tasks 00?04 cover reproduction, fallback sensitivity, heterogeneity, persistence, and exploratory explanation mismatch. CSV/LaTeX outputs go to `analysis/results/`, figures to `analysis/figures/`, and validation samples to `analysis/validation/`. These generated files are ignored. Reruns can replace outputs; preserve manually labeled copies separately. Private validation keys must remain separate from blinded labeling samples.

`narrative_support_20261011.py` separately describes active-streak breaks, terminal-zero onsets, disclosed horizons, and literal explanation keywords. These summaries do not identify causal mechanisms or validate returned reasoning.
