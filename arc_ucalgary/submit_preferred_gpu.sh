#!/bin/bash
# Submit the ARC training array to GPU partitions in the requested preference
# order: h100, a100, l40, then v100. Command-line sbatch failures try the next
# partition; once a job is accepted by SLURM, this script exits.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_SCRIPT="${SCRIPT_DIR}/submit_arc_training_array.slurm"
PARTITIONS=(gpu-h100 gpu-a100 gpu-l40 gpu-v100)

for part in "${PARTITIONS[@]}"; do
    echo "Trying ARC partition: ${part}"
    if sbatch --partition="${part}" "${SUBMIT_SCRIPT}"; then
        echo "Submitted to ${part}"
        exit 0
    fi
done

echo "ERROR: sbatch submission failed for all preferred GPU partitions: ${PARTITIONS[*]}" >&2
exit 1
