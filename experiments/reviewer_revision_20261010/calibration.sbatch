#!/bin/bash
# Prospective technical calibration; one GPU; no episode collection.
set -eo pipefail
[[ -n "${SLURM_JOB_ID:-}" && -n "${BUNDLE:-}" && -n "${MODEL:-}" && -n "${MODEL_REPO:-}" ]] || exit 2
source "$HOME/miniconda3/bin/activate" slaybench
export HF_HOME="/scratch/$USER/hf" HF_HUB_CACHE="/scratch/$USER/hf/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0 VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export LIBRARY_PATH="/usr/lib64:${LIBRARY_PATH:-}"
unset VLLM_API_KEY
RUN_DIR="$BUNDLE/runs/${MODEL}_${SLURM_JOB_ID}"
mkdir "$RUN_DIR"
MODEL_PATH=$(python "$BUNDLE/cache_snapshot.py" --repo "$MODEL_REPO" --hub "$HF_HUB_CACHE" --output "$RUN_DIR/cache.json")
PORT=$((28000 + SLURM_JOB_ID % 10000))
python -c 'import socket,sys; s=socket.socket(); s.bind(("127.0.0.1",int(sys.argv[1]))); s.close()' "$PORT"
VLLM_PID="" REQUEST_PID=""
stop_group() { local pid="$1"; [[ -n "$pid" ]] || return 0; kill -TERM -- "-$pid" 2>/dev/null || true; for i in {1..10}; do kill -0 "$pid" 2>/dev/null || break; sleep 1; done; kill -KILL -- "-$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; }
cleanup() { local status=$?; trap - EXIT TERM INT; stop_group "$REQUEST_PID"; stop_group "$VLLM_PID"; exit "$status"; }
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv > "$RUN_DIR/gpus.csv"
python "$BUNDLE/runtime_manifest.py" --output "$RUN_DIR/job.json"
setsid vllm serve "$MODEL_PATH" --served-model-name "$MODEL" --host 127.0.0.1 --port "$PORT" --tensor-parallel-size 1 --max-model-len 32768 --max-num-seqs 1 --gpu-memory-utilization 0.90 --generation-config vllm --enforce-eager > "$RUN_DIR/vllm.log" 2>&1 &
VLLM_PID=$!
ready=0
STARTUP_DEADLINE=$((SECONDS + 1200))
while (( SECONDS < STARTUP_DEADLINE )); do
  kill -0 "$VLLM_PID" 2>/dev/null || { tail -n 40 "$RUN_DIR/vllm.log"; exit 4; }
  if curl --noproxy '*' --max-time 5 -fsS -o /dev/null "http://127.0.0.1:$PORT/health"; then ready=1; break; fi
  sleep 5
done
[[ "$ready" == 1 ]] || exit 5
setsid python "$BUNDLE/calibrate.py" --model "$MODEL" --bank "$BUNDLE/request_bank.jsonl" --model-path "$MODEL_PATH" --base-url "http://127.0.0.1:$PORT/v1" --output "$RUN_DIR/diagnostic" > "$RUN_DIR/requests.log" 2>&1 &
REQUEST_PID=$!
wait "$REQUEST_PID"
