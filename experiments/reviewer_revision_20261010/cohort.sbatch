#!/bin/bash
# Prospective technical calibration; one GPU; no episode collection.
set -eo pipefail
[[ -n "${SLURM_JOB_ID:-}" && -n "${BUNDLE:-}" && -n "${MODEL:-}" && -n "${MODEL_PATH:-}" && -n "${CHUNK:-}" ]] || exit 2
source "$HOME/miniconda3/bin/activate" slaybench
export HF_HOME="/scratch/$USER/hf" HF_HUB_CACHE="/scratch/$USER/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0 VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export LIBRARY_PATH="/usr/lib64:${LIBRARY_PATH:-}"
unset VLLM_API_KEY
RUN_DIR="$BUNDLE/job_logs/${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
PORT=$((28000 + SLURM_JOB_ID % 10000))
VLLM_PID="" REQUEST_PID=""
stop_group() { local pid="$1"; [[ -n "$pid" ]] || return 0; kill -TERM -- "-$pid" 2>/dev/null || true; for i in {1..10}; do kill -0 "$pid" 2>/dev/null || break; sleep 1; done; kill -KILL -- "-$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; }
cleanup() { local status=$?; trap - EXIT TERM INT; stop_group "$REQUEST_PID"; stop_group "$VLLM_PID"; exit "$status"; }
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
python -c 'import socket,sys; s=socket.socket(); s.bind(("127.0.0.1",int(sys.argv[1]))); s.close()' "$PORT"
EXTRA=()
[[ "$MODEL" != mistral-small-3.1-24b-instruct ]] || EXTRA=(--tokenizer-mode mistral)
setsid vllm serve "$MODEL_PATH" --served-model-name "$MODEL" --host 127.0.0.1 --port "$PORT" --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 16384 --max-num-seqs 1 --gpu-memory-utilization 0.90 --generation-config vllm --enforce-eager "${EXTRA[@]}" > "$RUN_DIR/vllm.log" 2>&1 &
VLLM_PID=$!
ready=0
STARTUP_DEADLINE=$((SECONDS + 1200))
while (( SECONDS < STARTUP_DEADLINE )); do
  kill -0 "$VLLM_PID" 2>/dev/null || { tail -n 40 "$RUN_DIR/vllm.log"; exit 4; }
  if curl --noproxy '*' --max-time 5 -fsS -o /dev/null "http://127.0.0.1:$PORT/health"; then ready=1; break; fi
  sleep 5
done
[[ "$ready" == 1 ]] || exit 5
setsid python "$BUNDLE/experiments/reviewer_revision_20261010/run_chunk.py" --chunk "$CHUNK" --base-url "http://127.0.0.1:$PORT/v1" > "$RUN_DIR/collection.log" 2>&1 &
REQUEST_PID=$!
wait "$REQUEST_PID"
