# POST-HOC / EXPLORATORY — Sharanga execution receipt

Server project: `/home/jagatsesh/f20220604_ts/agentic-game-lab`.
Saved input: `results/analysis/main_complete_20261010/main.jsonl`, SHA256 matches the verified local source. Explanation export copied into `analysis/explanation_export_20261010`; original export identity/text hashes pass. Experiment code, raw logs and existing jobs were not modified.

Successful jobs, in strict dependent order:

| Job | Script | State | Exit | Wall time |
|---|---|---|---|---|
| 388781 | 00_reproduction.py | COMPLETED | 0:0 | 18 seconds |
| 388782 | 01_robustness.py | COMPLETED | 0:0 | 32 seconds |
| 388783 | 02_heterogeneity.py | COMPLETED | 0:0 | 9 seconds |
| 388784 | 03_persistence.py | COMPLETED | 0:0 | 9 seconds |
| 388785 | 04_mismatch.py | COMPLETED | 0:0 | 17 seconds |
| 388786 | finalize.py | COMPLETED | 0:0 | 4 seconds |

Each requested four CPUs, 4 GB, ten minutes, zero GPUs, compute partition/cpulimit QOS. Four CPUs are the measured QOS minimum; big_compute requires 128 CPUs. Numerical threads were capped at one. Slurm afterok dependencies and kill-on-invalid-dep prevented later tasks from running after failure. No polling daemon, cron, inference job or judge was installed.

Receipt/stdout: `analysis/logs/20261010T172837Z/`. The isolated venv is `analysis/.venv_posthoc_20261010`; it inherits the mas environment and contains its own pinned CPU analysis packages. Neither mas nor slaybench was altered. Actual package versions, job receipts and the 29-table numerical comparison are preserved in server_verification.json.

Deployment issues, preserved rather than hidden:

- Initial submissions were rejected before allocation because of missing QOS and then the four-CPU QOS minimum. Requests were corrected after inspecting account/QOS metadata.
- Job 388763 failed before numerical reproduction because the older server experiment tree lacks extract_decision_json. Jobs 388764–388768 were automatically cancelled by dependency handling.
- Job 388772 failed because the analysis-only parser snapshot had not yet arrived during the transfer retry; jobs 388773–388777 were automatically cancelled. No published-number comparison failed in either attempt.
- The exact v2 extractor and validator were frozen from the paper's local code into analysis/frozen_parser_v2.py, with a source digest. Its existence/import was checked before the successful chain. This avoids changing server experiment code or importing inference entrypoints.

All 97 numerical anchors pass on Sharanga. All 29 CSV result tables match local values with rtol=atol=1e-10. No model calls occurred. Human validation labels remain pending. Local generated artifacts were visually checked, and the summary-grid LaTeX compiled successfully after replacing its Unicode minus in a heading with a portable ASCII hyphen.

The downloaded verification archive is preserved under local analysis/qa/; it was extracted into an isolated directory, never over the local working tree. Platform-specific PDF metadata and local/server source-path strings are expected to differ; scientific CSV results were compared directly.
