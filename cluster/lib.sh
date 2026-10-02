# Shared lifecycle; source from a batch script. No inference on the login node.

die() { echo "ERROR: $*" >&2; exit 1; }

log_tail() { tail -n 80 "$VLLM_LOG" >&2 || true; }

stop_group() {
    local pid="${1:-}" i
    [[ -n "$pid" ]] || return 0
    # Negative PID targets only the session/process group this job launched.
    kill -TERM -- "-$pid" 2>/dev/null || true
    for i in {1..10}; do
        kill -0 -- "-$pid" 2>/dev/null || break
        sleep 1
    done
    kill -KILL -- "-$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

cleanup() {
    local status=$?
    trap - EXIT TERM INT
    stop_group "${EXPERIMENT_PID:-}"
    stop_group "${VLLM_PID:-}"
    echo "Job exit status: $status"
    exit "$status"
}

wait_for_vllm() {
    local deadline=$((SECONDS + HEALTH_WAIT_MIN * 60))
    while (( SECONDS < deadline )); do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            log_tail; die "vLLM died during startup"
        fi
        if grep -qi 'address already in use' "$VLLM_LOG"; then
            log_tail; die "Port $PORT is occupied; choose PORT explicitly"
        fi
        if curl --noproxy '*' --max-time 5 -fsS -o /dev/null "http://127.0.0.1:$PORT/health" 2>/dev/null &&
           curl --noproxy '*' --max-time 5 -fsS "${LLM_BASE_URL}/models" 2>/dev/null |
               python -c 'import json,sys; data=json.load(sys.stdin); sys.exit(0 if any(m.get("id")==sys.argv[1] for m in data.get("data",[])) else 1)' "$SERVED_NAME"; then
            kill -0 "$VLLM_PID" 2>/dev/null || { log_tail; die "vLLM exited during readiness"; }
            echo "vLLM ready: $LLM_BASE_URL ($SERVED_NAME)"
            return 0
        fi
        sleep 5
    done
    log_tail
    die "vLLM readiness timed out after $HEALTH_WAIT_MIN minutes"
}

run_job() {
    local kind="$1" model_values allocated_gpus extra_file
    [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_GPUS:-${SLURM_GPUS_ON_NODE:-}}" ]] ||
        die "Run through sbatch with a GPU allocation; no login-node inference"
    [[ "${SLURM_JOB_NUM_NODES:-1}" == 1 ]] || die "This launcher supports one node only"
    cd "$REPO_ROOT"
    # Capture provenance before conda can shadow the system git executable.
    export AGL_GIT_COMMIT AGL_GIT_STATUS
    AGL_GIT_COMMIT=$(git rev-parse HEAD) || die "Cannot capture repository commit"
    AGL_GIT_STATUS=$(git status --porcelain --untracked-files=no) || die "Cannot capture repository status"
    echo "Git commit: $AGL_GIT_COMMIT"
    echo "Host: $(hostname); job: $SLURM_JOB_ID; partition: ${SLURM_JOB_PARTITION:-unknown}"

    # Conda activation hooks reference unset variables: enable nounset afterward.
    export CONDA_ENV="${CONDA_ENV:-slaybench}"
    source "${CONDA_ACTIVATE:-$HOME/miniconda3/bin/activate}" "$CONDA_ENV"
    set -u
    export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
    export HF_HOME="${HF_HOME:-/scratch/$USER/hf}"
    export HF_HUB_CACHE="$HF_HOME/hub"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
    export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
    unset VLLM_API_KEY  # This job's loopback endpoint is deliberately key-free.
    export PYTHONUNBUFFERED=1
    export VLLM_USE_FLASHINFER_SAMPLER=0
    export VLLM_ALLREDUCE_USE_FLASHINFER=0
    export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0
    export VLLM_USE_DEEP_GEMM=0
    export VLLM_MOE_USE_DEEP_GEMM=0
    export LIBRARY_PATH="/usr/lib64:${LIBRARY_PATH:-}"

    export MODEL="${MODEL:-qwen3-8b}"
    model_values=$(python -m AI_Agent.model_registry "$MODEL" --field repo --field tp_size --field health_wait_min --field extra_serve_args)
    local -a defaults
    mapfile -t defaults <<< "$model_values"
    export MODEL_REPO="${MODEL_REPO:-${defaults[0]}}"
    export SERVED_NAME="${SERVED_NAME:-$MODEL}"
    export TP_SIZE="${TP_SIZE:-${defaults[1]}}"
    export PORT="${PORT:-$((18000 + SLURM_JOB_ID % 20000))}"
    export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
    export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
    export HEALTH_WAIT_MIN="${HEALTH_WAIT_MIN:-${defaults[2]}}"
    export EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS-${defaults[3]:-}}"
    export LOCAL_TIMEOUT="${LOCAL_TIMEOUT:-600}"
    [[ "$TP_SIZE" =~ ^[1-9][0-9]*$ ]] || die "TP_SIZE must be a positive integer"
    [[ "$HEALTH_WAIT_MIN" =~ ^[1-9][0-9]*$ ]] || die "HEALTH_WAIT_MIN must be a positive integer"
    [[ "$PORT" =~ ^[1-9][0-9]*$ ]] && (( PORT <= 65535 )) || die "PORT must be in 1..65535"
    [[ "$MAX_MODEL_LEN" =~ ^[1-9][0-9]*$ ]] || die "MAX_MODEL_LEN must be a positive integer"
    allocated_gpus="${SLURM_GPUS_ON_NODE:-}"
    if [[ ! "$allocated_gpus" =~ ^[1-9][0-9]*$ ]]; then
        local -a gpu_ids
        IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES:?Cannot determine allocated GPUs}"
        allocated_gpus="${#gpu_ids[@]}"
    fi
    [[ "$allocated_gpus" == "$TP_SIZE" ]] || die "Allocated GPUs=$allocated_gpus but TP_SIZE=$TP_SIZE; match --gres=gpu:$TP_SIZE"
    export ALLOCATED_GPUS="$allocated_gpus"
    local hub_dir="${HF_HOME}/hub/models--${MODEL_REPO//\//--}"
    [[ -d "$hub_dir" ]] || die "Selected model is not present in: $hub_dir. Prefetch it manually before submitting."
    command -v vllm >/dev/null || die "vllm missing from conda environment $CONDA_ENV"
    command -v setsid >/dev/null || die "setsid is required for job-scoped cleanup"
    command -v curl >/dev/null || die "curl is required for health checks"

    export LLM_BASE_URL="http://127.0.0.1:$PORT/v1" LLM_MODEL="$SERVED_NAME"
    export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-$REPO_ROOT/experiments/local_episode.json}"
    if [[ "$kind" == smoke && -z "${SMOKE_CONFIG:-}" ]]; then
        export EXPERIMENT_CONFIG="$REPO_ROOT/experiments/local_smoke.json"
    elif [[ "$kind" == smoke ]]; then
        export EXPERIMENT_CONFIG="$SMOKE_CONFIG"
    fi
    export RUN_DIR="${RUN_DIR:-$REPO_ROOT/results/sharanga/${kind}_${SLURM_JOB_ID}}"
    [[ ! -e "$RUN_DIR" ]] || die "Output already exists: $RUN_DIR; choose a new RUN_DIR"
    mkdir -p "$RUN_DIR"
    export VLLM_LOG="$RUN_DIR/vllm.log" JOB_MANIFEST="$RUN_DIR/job.json"
    python -m game_engine.experiments.run_local_episode --config "$EXPERIMENT_CONFIG" --validate-only > "$RUN_DIR/experiment_config.json"
    cat "$RUN_DIR/experiment_config.json"
    nvidia-smi --query-gpu=name,uuid,memory.total --format=csv > "$RUN_DIR/gpus.csv"
    cat "$RUN_DIR/gpus.csv"

    # Fail on occupied ports; never terminate an existing listener.
    python - "$PORT" <<'PY'
import socket, sys
with socket.socket() as sock:
    try: sock.bind(('127.0.0.1', int(sys.argv[1])))
    except OSError as exc: sys.exit(f'Port {sys.argv[1]} unavailable: {exc}; override PORT')
PY
    local -a serve_args extra_args
    serve_args=(serve "$MODEL_REPO" --served-model-name "$SERVED_NAME"
        --host 127.0.0.1 --port "$PORT" --tensor-parallel-size "$TP_SIZE"
        --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL"
        --generation-config vllm)
    if (( TP_SIZE > 1 )); then
        serve_args+=(--disable-custom-all-reduce --compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}')
    fi
    # shlex supports quoted JSON without eval or shell command execution.
    extra_file="$RUN_DIR/extra_args.bin"
    python - "$extra_file" <<'PY'
import os, pathlib, shlex, sys
args = shlex.split(os.environ['EXTRA_SERVE_ARGS'])
# These options belong to the job contract, not the extension escape hatch.
reserved = {'--host','--port','--model','--served-model-name','--tensor-parallel-size',
            '--max-model-len','--gpu-memory-utilization','--generation-config',
            '--override-generation-config','--config','--api-key','--disable-custom-all-reduce',
            '--compilation-config','--download-dir'}
if any(arg.split('=',1)[0] in reserved for arg in args):
    sys.exit('EXTRA_SERVE_ARGS overrides a managed option; use the documented environment setting')
pathlib.Path(sys.argv[1]).write_bytes(b''.join(a.encode()+b'\0' for a in args))
PY
    mapfile -d '' -t extra_args < "$extra_file"
    serve_args+=("${extra_args[@]}")
    python - "${serve_args[@]}" <<'PY'
import importlib.metadata as md, json, os, pathlib, socket, subprocess, sys
keys = ('SLURM_JOB_ID','SLURM_JOB_PARTITION','ALLOCATED_GPUS','CUDA_VISIBLE_DEVICES',
        'MODEL','MODEL_REPO','SERVED_NAME','TP_SIZE','PORT','MAX_MODEL_LEN','GPU_MEM_UTIL',
        'HEALTH_WAIT_MIN','EXTRA_SERVE_ARGS','CONDA_ENV','HF_HOME','LLM_BASE_URL','LOCAL_TIMEOUT')
packages = {}
for name in ('vllm','torch','transformers','requests'):
    try: packages[name] = md.version(name)
    except md.PackageNotFoundError: packages[name] = None
root = pathlib.Path(os.environ['RUN_DIR'])
manifest = dict(environment={k: os.environ.get(k) for k in keys}, hostname=socket.gethostname(),
    git_commit=os.environ['AGL_GIT_COMMIT'],
    git_status=os.environ['AGL_GIT_STATUS'],
    packages=packages, vllm_command=['vllm',*sys.argv[1:]],
    gpu_inventory=(root/'gpus.csv').read_text(),
    experiment=json.loads((root/'experiment_config.json').read_text()))
(root/'job.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
PY
    VLLM_PID="" EXPERIMENT_PID=""
    trap cleanup EXIT
    trap 'exit 143' TERM
    trap 'exit 130' INT
    setsid vllm "${serve_args[@]}" > "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    wait_for_vllm
    local -a probe_args=()
    [[ "$kind" != smoke ]] || probe_args+=(--deployment-check)
    setsid python -m game_engine.experiments.run_local_episode \
        --config "$RUN_DIR/experiment_config.json" --output "$RUN_DIR/episodes" \
        "${probe_args[@]}" \
        > "$RUN_DIR/experiment.log" 2>&1 &
    EXPERIMENT_PID=$!
    local experiment_status=0
    wait "$EXPERIMENT_PID" || experiment_status=$?
    if (( experiment_status != 0 )); then
        tail -n 80 "$RUN_DIR/experiment.log" >&2
        log_tail
        die "Experiment failed (status $experiment_status); partial results retained in $RUN_DIR"
    fi
    echo "Completed: $RUN_DIR (experiment.log, vllm.log, job.json, episodes/)"
}
