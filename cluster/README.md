# Batch execution helpers

`main_study_jobs.py` plans study jobs. `submit.sh`, `lib.sh`, and the batch templates implement a shared model-server lifecycle. Configure paths, environment, partition, GPU requests, and endpoints for your own installation before use. Inspect each command's help and template before submitting; execution consumes compute.

The paper used local vLLM endpoints at temperature 0.7, top-p 1.0, a 16,384-token context limit, tensor parallelism one, and output ceilings of 8,192 tokens for DeepSeek and 2,048 for the other models. Hardware assignments and reliability are documented in the manuscript appendix. Historical launch scripts, personal account paths, scheduler receipts, and server snapshots are excluded from this release.
