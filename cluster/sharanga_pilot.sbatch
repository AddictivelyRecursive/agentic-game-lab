#!/bin/bash
#SBATCH --job-name=agl_pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=agl_pilot_%j.out
set -eo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}"
source "$REPO_ROOT/cluster/lib.sh"
run_job pilot
