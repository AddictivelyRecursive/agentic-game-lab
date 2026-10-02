# Step 0 deployment status

Checked 2026-10-02 on Sharanga. **Step 0 complete: no.**
The authoritative study specification is [study_specification.md](study_specification.md).

## Cancellation update

At the user's request, jobs **376881, 376882, 376883, 376884, 376885** were
cancelled on 2026-10-02. `sacct` reported CANCELLED for each; `squeue` listed
none of these jobs afterward. Only these five exact IDs were passed to scancel.
GPT-OSS submission had already been rejected and had no job to cancel.
No other user's or pre-existing jobs were cancelled. No further server
connections or submissions were made after verifying cancellation.

The tables below retain the earlier deployment attempt history, not live queue
status. All six experiment scripts have since been implemented **locally**;
see [experiment_workflow.md](experiment_workflow.md). The two earlier remote
source-file copies remain in that checkout; cancellation did not revert them.

| Alias | Confirmed reachable | Trivial prompt response | Step 0 job/status |
|---|---|---|---|
| llama-3.1-8b-instruct | No, pending verification | Not produced | 376881, A100, pending Resources |
| qwen3-8b | No, pending verification | Not produced | 376882, A100, pending Priority |
| gemma-3-12b-it | No, pending verification | Not produced | 376883, A100, pending Priority |
| mistral-small-3.1-24b-instruct | No, pending verification | Not produced | 376884, A100, pending Priority |
| deepseek-r1-distill-qwen-14b | No, pending verification | Not produced | 376885, A100, pending Priority |
| gpt-oss-120b | No, submission blocked | Not produced | H200 submission rejected: QOSMaxSubmitJobPerUserLimit |

No means not confirmed; it is not evidence that the model cannot serve.
All submitted checks use the existing LocalOpenAIClient and N=2, M=5,
p=0, theta=0.50, T=5, seed=101, structured reasoning. Every seat uses
the selected model with separate agent state. These are deployment checks,
not pilot data. Each job requests one GPU and at most 90 minutes.

## GPT-OSS requirements

- Requested a separate one-GPU H200 allocation with 128 GB host RAM.
- Actual VRAM usage, loaded quantization, coexistence capacity, trivial response,
  and five-round latency: **unmeasured**, because Slurm rejected submission.
- No shared serving allocation with the other five models has been validated.
- No empirical full-grid time or monetary cost estimate is available yet.
  No model has been dropped. Do not substitute smaller-model latency estimates.
- No retry is scheduled or authorized by the local implementation work.

## Repair and evidence

Prior Qwen smoke 376611 failed before serving with
`PermissionError: [Errno 13] Permission denied: 'git'` during metadata capture.
The launcher now captures required Git provenance before conda activation,
preventing conda PATH changes from shadowing the system Git executable.
The two modified source files were copied to the existing remote checkout;
no commit or push was made.

Smoke execution now saves `episodes/deployment_probe.json` with the literal
response to `Reply with the single word READY.` and elapsed time. Each completed
episode saves `round_latency.json`: the sum of sequential agent decision times
per round, including repairs and excluding simulator bookkeeping. Existing raw
episode and agent outputs remain available for inspection. The job manifest and
vLLM log must also be inspected for actual serving configuration and memory use.

Validation: all 32 existing tests passed on Sharanga; Bash syntax passed.
Locally 25 tests passed and seven POSIX-only tests were skipped.
Mocked tests do not establish real model reachability.

## Next gate

Inspect `results/sharanga/smoke_<jobid>/` after queued checks finish, resolve
failures, and complete the isolated GPT-OSS smoke before any pilot dry-run.
The six requested study scripts are now implemented locally; this deployment
gate still has not passed.
No pilot, main run, power calculation, hypothesis testing, or coding was run.
Pilot and main execution still require explicit conversational approval of
their respective dry-run plans under section 4 of the supplied specification.

DeepSeek-R1-Distill-Qwen-14B is a DeepSeek-trained distillation on a Qwen base;
the design treats its independent training organization/process explicitly.
