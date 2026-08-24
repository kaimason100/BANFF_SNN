#!/bin/bash
# Submit every required BANFF training array from the repository root.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

fast_partitions="gpu-h100,gpu-a100,gpu-l40"
static_partitions="${fast_partitions},gpu-v100"
script="arc_ucalgary/submit_arc.slurm"

# V100-eligible: Breast Cancer, Abalone, Toyota and Yacht.
# Restricted: MNIST, Afro-MNIST and all dynamical-system tasks.
sbatch --partition="${static_partitions}" --array=1,4-6,10,13-15,19,22-24 "${script}"
sbatch --partition="${fast_partitions}" --array=2-3,7-9,11-12,16-18,20-21,25-27 "${script}"

# Full-rank and SPSA: Breast Cancer/Yacht may use V100; Van der Pol may not.
sbatch --export=ALL,BANFF_ARCHITECTURE=full_rank --partition="${static_partitions}" --array=1-2 "${script}"
sbatch --export=ALL,BANFF_ARCHITECTURE=full_rank --partition="${fast_partitions}" --array=3 "${script}"
sbatch --export=ALL,BANFF_ARCHITECTURE=spsa --partition="${static_partitions}" --array=1-2 "${script}"
sbatch --export=ALL,BANFF_ARCHITECTURE=spsa --partition="${fast_partitions}" --array=3 "${script}"

# Neuron sweep: Breast Cancer/Yacht (1-10) may use V100; Lorenz (11-15) may not.
sbatch --export=ALL,BANFF_ARCHITECTURE=neuron_sweep --partition="${static_partitions}" --array=1-10 "${script}"
sbatch --export=ALL,BANFF_ARCHITECTURE=neuron_sweep --partition="${fast_partitions}" --array=11-15 "${script}"
