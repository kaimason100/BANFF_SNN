#!/bin/bash
# One array element requests one GPU and runs one complete BANFF task/seed.
# The MATLAB process owns scientific checkpoint creation; this wrapper handles
# scheduler signals and same-partition continuation without altering model state.
#SBATCH --job-name=banff_snn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
# Restricted tasks are present in the default mixed array, so direct submission
# must use only the universally compatible partitions. submit_all.sh explicitly
# adds V100 for the eligible task subsets.
#SBATCH --partition=gpu-h100,gpu-a100,gpu-l40
#SBATCH --array=1-27
#SBATCH --requeue
#SBATCH --output=arc_ucalgary/arc-%A_%a.out
#SBATCH --error=arc_ucalgary/arc-%A_%a.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p outputs

module purge || exit 1
module load matlab/r2023a || exit 1

echo "===== JOB ENVIRONMENT ====="
pwd
hostname
which matlab

echo "===== CODE HASHES ====="
sha256sum banff.m
sha256sum banff_train.m
sha256sum banff_test.m
sha256sum banff_eval.m
sha256sum banff_model.m
sha256sum banff_data.m
sha256sum banff_metrics.m
sha256sum banff_provenance.m
sha256sum banff_publication.m
sha256sum run_experiment.m
sha256sum arc_ucalgary/run_arc.m

echo "===== GPU ====="
nvidia-smi || exit 1

matlab -batch "run('arc_ucalgary/run_arc.m')"
status=$?
if [ "${status}" -eq 75 ]; then
    # Initial jobs may be offered to several compatible partitions. Once one
    # partition runs this task, submit continuation as a new one-item array
    # restricted to that exact partition.
    allocated_partition="${SLURM_JOB_PARTITION:-}"
    if [ -z "${allocated_partition}" ]; then
        echo "Checkpoint saved, but SLURM_JOB_PARTITION is empty; refusing an unpinned resubmission." >&2
        exit 1
    fi
    case "${allocated_partition}" in
        *,*)
            echo "Checkpoint saved, but SLURM_JOB_PARTITION contains multiple partitions; refusing an unpinned resubmission." >&2
            exit 1
            ;;
    esac
    if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
        echo "Checkpoint saved, but SLURM_ARRAY_TASK_ID is empty; cannot resubmit this array item." >&2
        exit 1
    fi

    submit_script="${SLURM_SUBMIT_DIR}/arc_ucalgary/submit_arc.slurm"
    if [ ! -f "${submit_script}" ]; then
        echo "Checkpoint saved, but the resubmission script was not found: ${submit_script}" >&2
        exit 1
    fi
    echo "Checkpoint saved on partition ${allocated_partition}."
    echo "Resubmitting array item ${SLURM_ARRAY_TASK_ID} to the same partition only."
    sbatch --partition="${allocated_partition}" \
        --array="${SLURM_ARRAY_TASK_ID}" \
        --export=ALL \
        "${submit_script}"
    resubmit_status=$?
    if [ "${resubmit_status}" -ne 0 ]; then
        echo "Checkpoint is intact, but same-partition resubmission failed." >&2
        exit "${resubmit_status}"
    fi
    exit 0
fi
exit "${status}"
