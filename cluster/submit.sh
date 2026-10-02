#!/bin/bash
# Submission only. No environment activation, model loading, or server startup.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
MODEL="${1:-qwen3-8b}"
export MODEL
[[ $# -eq 0 ]] || shift
KIND="${KIND:-run}"
[[ "$KIND" == run || "$KIND" == smoke ]] || { echo "KIND must be run or smoke" >&2; exit 1; }
registry_tp=$("${PYTHON:-python3}" -m AI_Agent.model_registry "$MODEL" --field tp_size)
export TP_SIZE="${TP_SIZE:-$registry_tp}"
[[ "$TP_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid TP_SIZE" >&2; exit 1; }
default_partition=gpu_h200_8
[[ "$KIND" != smoke ]] || default_partition=gpu_a100_8
cd "$REPO_ROOT"
exec sbatch --export=ALL --partition="${PARTITION:-$default_partition}" \
    --gres="gpu:$TP_SIZE" --cpus-per-task="${CPUS_PER_TASK:-$((4 * TP_SIZE))}" \
    "$@" "$REPO_ROOT/cluster/sharanga_${KIND}.sbatch"
