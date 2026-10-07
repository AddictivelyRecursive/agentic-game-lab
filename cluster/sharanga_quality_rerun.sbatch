#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=agl_quality_%j.out
set -eo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?}}"
# Separate lifecycle copy keeps current jobs and their seed plan unchanged.
source "$REPO_ROOT/cluster/quality_rerun_lib.sh"
run_job pilot
