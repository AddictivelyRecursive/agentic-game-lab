# Sharanga local inference

## Architecture and scope

```text
sbatch -> one GPU node -> conda -> cached HF model -> vLLM
       -> loopback OpenAI chat endpoint -> existing agents/simulator -> results
       -> terminate only this job's experiment and vLLM process groups
```

Previously, `LLMWrapperAgent` selected Ollama, OpenRouter, or a dummy client.
OpenRouter required `OPENROUTER_API_KEY`. The agent pipeline builds a system
message plus a user JSON payload containing the current observation, intended
history, optional model memory, and optional EU ranking; repair calls use that
same evidence interface. It does not maintain a separate chat-message transcript.

`backend="local"` now adds `LocalOpenAIClient.generate(system_prompt, user_prompt)`.
It sends those messages unchanged to `/v1/chat/completions`, accepts only loopback
HTTP URLs, disables proxies and redirects, and requires no API key. The existing
validation/repair budget handles failures; the transport adds no automatic retries.
Legacy clients retain their existing behavior. Direct `LLMAgent` users can inject
the local client. Wrapper users pass `backend="local"`, `model_name`, `base_url`,
and `generation_config`; omitted local model/URL use `LLM_MODEL`/`LLM_BASE_URL`.

`run_local_episode` is a configurable single-layout runner using the unchanged
simulator, prompt builder, logger, and episode writer. Every seat uses the selected
model with separate state. It does **not** relabel the legacy heterogeneous lineup
as one model. Legacy sweep scripts are unchanged and still use their configured
providers. This is infrastructure, not the Phase 2 N × mode sweep or paper data.
The legacy study validator requires T=50 and is unchanged; the new infrastructure
entrypoint validates its own inputs to allow the explicitly requested short smoke.

## What was inspected on Sharanga

Read-only inspection on 2026-10-02 covered `/home/jagatsesh/slay-bench/cluster/`
`lib.sh`, `sharanga_smoke.sbatch`, and `sharanga_run_level.sbatch`. That repository
was not modified. Conda package metadata reported:

| Environment | vLLM | torch | transformers |
|---|---|---|---|
| `slaybench` (default here) | 0.25.1 | 2.11.0 | 5.14.1 |
| `slaybench08` | 0.8.5.post1 | 2.6.0 | 4.51.3 |
| `mas` | absent | absent | absent |

All 25 requested model cache directories were present. Directory presence does
not prove complete weights, chat-template support, or successful GPU loading.
`sinfo` reported A100 80GB and H100 80GB partitions, and H200 NVL. Time limits
were A100 five days, H100 three days, **H200 one day** (shorter than the reference
script's older comments). Recheck site quotas before large jobs.

The reference's offline cache, activation order, compatibility flags, and TP
workarounds are preserved. Its broad `pkill`/port-holder cleanup is not used.

## One-time prerequisites

1. Commit/push these local changes, then clone/pull this repository on Sharanga.
   Nothing is automatically committed, pushed, or submitted by implementation.
2. Use the existing `slaybench` environment, or set `CONDA_ENV` to a separately
   prepared compatible environment. No job installs or upgrades packages.
   `requests` was already present there. If missing in your chosen environment,
   install `requirements-local.txt` once, with permission to modify that environment.
   This file covers the client/simulator, not plotting or the GPU serving stack.
3. Ensure cached weights remain under `/scratch/$USER/hf/hub`. The cache may be
   purged by cluster policy. Missing/incomplete weights require a separate manual
   prefetch; gated models may require a token for that prefetch only. Batch jobs
   never download models or require an external provider token.
4. `bash`, `setsid`, `curl`, `git`, `nvidia-smi`, Slurm, and the conda activation
   script must be available. The submission helper needs only Python 3 and sbatch
   on the login node. It does not import vLLM or load a model there.

## Commands

Run from the repository root after `ssh sharanga` and `git pull origin main`.
These are commands for you to submit later; no real model smoke has been run yet.

```bash
# Cheap smoke: one GPU, qwen3-8b, N=2, M=5, five rounds, one seed, structured.
sbatch cluster/sharanga_smoke.sbatch
# Equivalent helper; selects the smoke script and its A100 default.
KIND=smoke bash cluster/submit.sh qwen3-8b

# One ordinary example: N=2, 20 rounds, one seed, free_form; not a paper sweep.
bash cluster/submit.sh qwen3-8b
# Direct submission also works (default H200, one GPU).
sbatch --export=ALL,MODEL=qwen3-8b cluster/sharanga_run.sbatch

# Another model / selectable partition and resource overrides.
PARTITION=gpu_h100_4 bash cluster/submit.sh qwen3-32b --time=03:00:00

# ~70B BF16 example: registry supplies TP=2 and helper requests two GPUs.
PARTITION=gpu_h200_8 bash cluster/submit.sh qwen2.5-72b-instruct --mem=200G
# Direct sbatch must explicitly request GPUs matching the resolved TP size.
sbatch --partition=gpu_h200_8 --gres=gpu:2 --cpus-per-task=8 --mem=200G \
  --export=ALL,MODEL=qwen2.5-72b-instruct,TP_SIZE=2,HEALTH_WAIT_MIN=90 \
  cluster/sharanga_run.sbatch

# Use a reviewed single-layout configuration, including explicit seeds/generation.
EXPERIMENT_CONFIG="$PWD/experiments/local_episode.json" bash cluster/submit.sh qwen3-8b

squeue -u "$USER"
tail -f agl_smoke_<JOBID>.out
tail -f results/sharanga/smoke_<JOBID>/vllm.log
tail -f results/sharanga/run_<JOBID>/experiment.log
scancel <JOBID>
```

The helper only resolves the alias/TP and submits a batch job. It uses four CPUs
per GPU by default. `PARTITION`, `TP_SIZE`, `CPUS_PER_TASK`, and trailing sbatch
options override its resource defaults. Slurm command-line resource flags override
the script directives. A mismatched allocation/TP is rejected before server startup.
Do not pass a trailing script name or experiment command to the helper.

## Configuration

| Setting | Default / meaning |
|---|---|
| `MODEL` | `qwen3-8b`; central alias in `src/AI_Agent/model_registry.py` |
| `MODEL_REPO` | Registry canonical HF ID; explicit override is logged |
| `SERVED_NAME` | Alias; becomes `LLM_MODEL` |
| `CONDA_ENV` | `slaybench`; configurable for architecture compatibility |
| `CONDA_ACTIVATE` | `$HOME/miniconda3/bin/activate` |
| `HF_HOME` | `/scratch/$USER/hf`; `HF_HUB_CACHE` is set to its `hub/` |
| `PORT` | `18000 + SLURM_JOB_ID % 20000`; collision checked, never kills listener |
| `TP_SIZE` | Registry default, otherwise 1; must equal allocated GPUs |
| `MAX_MODEL_LEN` | 16384 |
| `GPU_MEM_UTIL` | 0.90 |
| `HEALTH_WAIT_MIN` | 60; override for longer loads (including very large models) |
| `EXTRA_SERVE_ARGS` | Empty; shell-quoted arguments parsed with shlex, never eval |
| `LOCAL_TIMEOUT` | 600 seconds per HTTP request, no transport retry |
| `EXPERIMENT_CONFIG` | `experiments/local_episode.json` for ordinary jobs |
| `SMOKE_CONFIG` | `experiments/local_smoke.json` for smoke jobs |
| `RUN_DIR` | `results/sharanga/{smoke,run}_<JOBID>`; must not already exist |
| `REPO_ROOT` | `SLURM_SUBMIT_DIR`; helper resolves it from its own location |

Offline and cluster compatibility flags are set unconditionally. TP>1 adds
`--disable-custom-all-reduce` and
`--compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}'`.
The registry supports optional per-model `tp_size`, `health_wait_min`, and
`extra_serve_args`; environment overrides win. Only the 70B/72B entries have TP=2
recommendations. Other large models require a deliberately selected allocation;
the generic TP=1 default is **not** a fit or compatibility claim.

Managed serving options cannot be replaced through `EXTRA_SERVE_ARGS`; use their
named settings. For additional options with JSON, export in the shell and use
`--export=ALL` so sbatch does not split JSON commas:

```bash
export EXTRA_SERVE_ARGS='--enforce-eager'
bash cluster/submit.sh qwen3-8b
```

No model-specific parser or chat template is silently selected. Base models such
as `qwen2.5-7b` may need an explicitly supplied local `--chat-template` file.
Newer architectures, multimodal models, and quantized models may need a different
environment or documented serving arguments. Missing text `content` fails through
the existing repair/validation path; reasoning-only fields are not silently
substituted. An alias in the registry is not a claim that a model is validated.

The new example configs explicitly set temperature=0.2, top_p=1, max_tokens=2048,
generation seed=null, and episode seed=101. The previous remote client only sent
temperature=0.2, leaving other defaults to the provider. These local values are
documented infrastructure defaults, not a claim of identical remote generation.
Choose and freeze them before scientific collection. A null generation seed means
no request seed; it is distinct from simulator seeds. No family-specific changes
are made to these values. Context length includes prompt and output; overflow
fails rather than truncating history. `--generation-config vllm` prevents model
repository generation defaults from silently changing sampling:
[vLLM serving reference](https://docs.vllm.ai/en/latest/cli/serve/).

## Outputs and failure behavior

Slurm stdout is `agl_{smoke,run}_<JOBID>.out` in the submission directory. Under
`RUN_DIR` are `job.json`, `gpus.csv`, the copied `experiment_config.json`,
`vllm.log`, `experiment.log`, and `episodes/` with existing episode JSON/JSONL
and per-agent traces. Job metadata includes job/partition/host, GPU inventory and
count, model identity, actual vLLM argv, package versions, generation settings,
seeds, git commit and tracked dirty status. The server and client use only loopback.

Health checks require both `/health` and the exact served model in `/v1/models`,
and verify that the launched process remains alive. Job-derived ports can still
collide after modulo wraparound; collisions fail with instructions to override
`PORT`. Failures print diagnostic tails and return nonzero. Completed episodes
and written agent traces survive; an interrupted episode may lack a complete
episode summary. Fallback-containing episodes are saved, flagged invalid, and
cause the run to fail. No alternate model/provider or automatic download is used.

EXIT/TERM/INT cleanup signals only the process groups started by this job.
Uncatchable SIGKILL/node failure relies on Slurm's job cleanup. No account-wide
process termination, cache deletion, shared-environment mutation, or automatic
resubmission is performed.

## Validation without model inference

```bash
python -m unittest discover -s tests -v
for f in cluster/*.sh cluster/*.sbatch; do bash -n "$f" || exit; done
PYTHONPATH=src python -m game_engine.experiments.run_local_episode \
  --config experiments/local_smoke.json --validate-only
PYTHONPATH=src python -m AI_Agent.model_registry
```

Tests mock the HTTP boundary, including repair/history preservation, saved results,
API aborts, and fallback visibility. POSIX-only shell tests use fake curl/kill/sbatch
and do not start vLLM or submit jobs. An actual allocated-GPU smoke remains the
deployment gate; no full sweep or main data collection is authorized by this setup.

Implementation validation: 32 tests passed in an isolated temporary copy on
Sharanga, using its existing Python environment and fake lifecycle commands.
No GPU allocation, real inference, or Slurm submission was performed. Bash syntax
checks and both example configuration checks also passed.
