#!/bin/bash
#SBATCH --job-name=banff_snn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
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
sha256sum banff_publication.m
sha256sum run_experiment.m
sha256sum arc_ucalgary/run_arc.m

echo "===== GPU ====="
nvidia-smi || exit 1

matlab -batch "run('arc_ucalgary/run_arc.m')"
status=$?
if [ "${status}" -eq 75 ]; then
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    exit 0
fi
exit "${status}"
