#!/bin/bash
#SBATCH --job-name=agl_smoke
#SBATCH --partition=gpu_a100_8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=agl_smoke_%j.out

set -eo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
source "$REPO_ROOT/cluster/lib.sh"
run_job smoke
