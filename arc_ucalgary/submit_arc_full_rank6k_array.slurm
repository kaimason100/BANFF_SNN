#!/bin/bash
#SBATCH --job-name=snn_full_rank6k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=0-24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100,gpu-a100,gpu-l40,gpu-v100
#SBATCH --array=1-3
#SBATCH --output=arc_full_rank6k-%A_%a.out
#SBATCH --error=arc_full_rank6k-%A_%a.err

set -euo pipefail

find_repo_root() {
    local start="$1"
    local candidate
    for candidate in \
        "${start}" \
        "${start}"/* \
        "${start}"/*/* \
        "${start}"/*/*/* \
        "${start}"/*/*/*/*; do
        if [ -f "${candidate}/arc_ucalgary/run_arc_full_rank6k_array.m" ] && \
           [ -f "${candidate}/shared/matlab/snn_primary_api.m" ]; then
            printf '%s\n' "${candidate}/arc_ucalgary"
            return 0
        fi
        if [ -f "${candidate}/run_arc_full_rank6k_array.m" ] && \
           [ -f "${candidate}/shared/matlab/snn_primary_api.m" ]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    candidate="$(find "${start}" -maxdepth 6 -type f -path '*/shared/matlab/snn_primary_api.m' -print -quit 2>/dev/null | sed 's#/shared/matlab/snn_primary_api.m##')"
    if [ -n "${candidate}" ] && [ -f "${candidate}/arc_ucalgary/run_arc_full_rank6k_array.m" ]; then
        printf '%s\n' "${candidate}/arc_ucalgary"
        return 0
    fi
    if [ -n "${candidate}" ] && [ -f "${candidate}/run_arc_full_rank6k_array.m" ]; then
        printf '%s\n' "${candidate}"
        return 0
    fi
    return 1
}

ROOT="$(find_repo_root "${SLURM_SUBMIT_DIR}" || true)"
if [ -z "${ROOT}" ]; then
    echo "ERROR: Could not locate repository root from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}"
    exit 4
fi

cd "${ROOT}"
mkdir -p outputs/arc_slurm outputs/arc_logs outputs/models outputs/checkpoints

echo "Full-rank 6k proof job ${SLURM_JOB_ID:-no_job_id} task ${SLURM_ARRAY_TASK_ID:-single} starting on $(hostname) at $(date)"
echo "Repository root: ${PWD}"

module purge

load_first_available() {
    for mod in "$@"; do
        if module load "${mod}" >/dev/null 2>&1; then
            echo "Loaded module: ${mod}"
            return 0
        fi
    done
    return 1
}

if [ -n "${MATLAB_MODULE:-}" ]; then
    module load "${MATLAB_MODULE}"
else
    load_first_available \
        matlab/r2024b matlab/R2024b \
        matlab/r2024a matlab/R2024a \
        matlab/r2023b matlab/R2023b \
        matlab/r2023a matlab/R2023a \
        matlab/r2022b matlab/R2022b \
        matlab/r2022a matlab/R2022a \
        matlab/r2021b matlab/R2021b \
        matlab/r2021a matlab/R2021a \
        matlab || {
            echo "ERROR: Could not load a MATLAB module. Run: module avail matlab"
            exit 2
        }
fi

if [ -n "${CUDA_MODULE:-}" ]; then
    module load "${CUDA_MODULE}"
else
    load_first_available cuda/12.2 cuda/12.1 cuda/12.0 cuda/11.8 cuda/11.7 cuda || true
fi

module list
nvidia-smi || true

MATLAB_CMD="addpath(pwd, '-begin'); run_arc_full_rank6k_array;"

matlab -nodisplay -nosplash -nodesktop -r "try, ${MATLAB_CMD} catch ME, disp(getReport(ME,'extended')); exit(1); end; exit(0);"

echo "Full-rank 6k proof job ${SLURM_JOB_ID:-no_job_id} task ${SLURM_ARRAY_TASK_ID:-single} finished at $(date)"
