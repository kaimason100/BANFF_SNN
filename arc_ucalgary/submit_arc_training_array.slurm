#!/bin/bash
#SBATCH --job-name=snn_arc_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=0-24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
#SBATCH --array=1-27
#SBATCH --output=arc_slurm-%A_%a.out
#SBATCH --error=arc_slurm-%A_%a.err

# Default target is gpu-h100. Submit to lower-priority ARC GPU nodes with:
#   sbatch --partition=gpu-a100 arc_ucalgary/submit_arc_training_array.slurm
#   sbatch --partition=gpu-l40  arc_ucalgary/submit_arc_training_array.slurm
#   sbatch --partition=gpu-v100 arc_ucalgary/submit_arc_training_array.slurm

set -euo pipefail

echo "arc_ucalgary standalone bundle version: 20260601_210000"

find_repo_root() {
    local start="$1"
    local candidate

    # Always use the standalone ARC bundle when this project contains one.
    for candidate in \
        "${start}" \
        "${start}"/* \
        "${start}"/*/* \
        "${start}"/*/*/* \
        "${start}"/*/*/*/*; do
        if [ -f "${candidate}/arc_ucalgary/run_arc_training_array.m" ] && \
           [ -f "${candidate}/shared/matlab/snn_primary_api.m" ] && \
           [ -d "${candidate}/arc_ucalgary/train" ]; then
            printf '%s\n' "${candidate}/arc_ucalgary"
            return 0
        fi
        if [ -f "${candidate}/run_arc_training_array.m" ] && \
           [ -f "${candidate}/shared/matlab/snn_primary_api.m" ] && \
           [ -d "${candidate}/train" ] && \
           [ -d "${candidate}/common" ]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    candidate="$(find "${start}" -maxdepth 6 -type f -path '*/shared/matlab/snn_primary_api.m' -print -quit 2>/dev/null | sed 's#/shared/matlab/snn_primary_api.m##')"
    if [ -n "${candidate}" ] && [ -f "${candidate}/arc_ucalgary/run_arc_training_array.m" ]; then
        printf '%s\n' "${candidate}/arc_ucalgary"
        return 0
    fi
    if [ -n "${candidate}" ] && [ -f "${candidate}/run_arc_training_array.m" ]; then
        printf '%s\n' "${candidate}"
        return 0
    fi
    return 1
}

ROOT="$(find_repo_root "${SLURM_SUBMIT_DIR}" || true)"
if [ -z "${ROOT}" ]; then
    echo "ERROR: Could not locate a standalone ARC bundle from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}"
    echo "A valid bundle contains shared/matlab/snn_primary_api.m, train/, and common/."
    exit 4
fi

cd "${ROOT}"
mkdir -p outputs/arc_slurm outputs/arc_logs outputs/models outputs/checkpoints

echo "Job ${SLURM_JOB_ID:-no_job_id} task ${SLURM_ARRAY_TASK_ID:-single} starting on $(hostname) at $(date)"
echo "Repository root: ${PWD}"

# Publication ARC defaults. Users can override these with explicit SNN_*
# exports when intentionally running architecture ablations.
: "${SNN_N_HIDDEN:=32000}"
: "${SNN_RECURRENT_MODE:=low_rank}"
: "${SNN_DECODER_MODE:=signed}"
: "${SNN_SIGNED_DECODER_DISTRIBUTION:=uniform}"
: "${SNN_MAX_SPARSE_FULL_RANK_NNZ:=20000000}"
: "${SNN_MAX_FULL_RANK_RECURRENT_BYTES:=2684354560}"
export SNN_N_HIDDEN SNN_RECURRENT_MODE SNN_DECODER_MODE SNN_SIGNED_DECODER_DISTRIBUTION SNN_MAX_SPARSE_FULL_RANK_NNZ SNN_MAX_FULL_RANK_RECURRENT_BYTES
echo "ARC architecture defaults: N_hidden=${SNN_N_HIDDEN}, recurrent=${SNN_RECURRENT_MODE}, decoder=${SNN_DECODER_MODE}, signed_distribution=${SNN_SIGNED_DECODER_DISTRIBUTION}, max_sparse_nnz=${SNN_MAX_SPARSE_FULL_RANK_NNZ}, max_recurrent_bytes=${SNN_MAX_FULL_RANK_RECURRENT_BYTES}"

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

if [ -d "/scratch/${SLURM_JOB_ID:-}" ]; then
    export TMPDIR="/scratch/${SLURM_JOB_ID}"
    export MATLAB_PREFDIR="${TMPDIR}/matlab_prefdir_${SLURM_ARRAY_TASK_ID:-0}"
    mkdir -p "${MATLAB_PREFDIR}"
fi

matlab -nodisplay -nosplash -nodesktop -r "try, addpath(pwd, '-begin'); run_arc_training_array; catch ME, disp(getReport(ME,'extended')); exit(1); end; exit(0);"

echo "Job ${SLURM_JOB_ID:-no_job_id} task ${SLURM_ARRAY_TASK_ID:-single} finished at $(date)"
