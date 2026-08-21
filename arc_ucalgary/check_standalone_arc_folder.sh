#!/bin/bash
# Verify that this transferred arc_ucalgary folder is self-contained before
# submitting jobs to ARC.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
missing=0

check_path() {
    local path="$1"
    if [ ! -e "${ROOT}/${path}" ]; then
        echo "MISSING: ${path}"
        missing=1
    else
        echo "OK: ${path}"
    fi
}

echo "Checking standalone ARC folder: ${ROOT}"
check_path "run_arc_training_array.m"
check_path "check_arc_path_resolution.m"
check_path "add_project_paths.m"
check_path "submit_arc_training_array.slurm"
check_path "submit_arc_static_neuron_sweeps_array.slurm"
check_path "run_arc_static_neuron_sweeps_array.m"
check_path "train/run_arc_static_neuron_sweep_gpu.m"
check_path "submit_preferred_gpu.sh"
check_path "common/arc_compile_required_mex.m"
check_path "common/arc_apply_network_seed.m"
check_path "common/arc_configure_checkpoint.m"
check_path "common/arc_resubmit_if_needed.m"
check_path "common/arc_clear_checkpoint_after_finish.m"
check_path "Classification/training/helpers/load_snn_classification_task.m"
check_path "train/run_arc_dynamics_vanderpol_gpu.m"
check_path "shared/matlab/snn_primary_api.m"
check_path "shared/matlab/snn_primary_api_functions/closed_loop_truth_from_network_state.m"
check_path "Classification/src/cuda/snn_classify_time_loop_gpu_mex.cu"
check_path "Regression/src/cuda/snn_regress_time_loop_gpu_mex.cu"
check_path "dynamical_systems/src/cuda/snn_time_loop_gpu_mex.cu"
check_path "dynamical_systems/utilities/make_dynamical_system.m"
check_path "dynamical_systems/utilities/simulate_dynamical_system.m"
check_path "spsa_gpu/arc/submit_spsa_gpu_array.slurm"
check_path "spsa_gpu/arc/run_spsa_gpu_array.m"
check_path "spsa_gpu/build/compile_spsa_gpu_mex.m"
check_path "spsa_gpu/matlab/spsa_gpu_add_paths.m"
check_path "spsa_gpu/matlab/spsa_gpu_run_task.m"
check_path "spsa_gpu/src/cuda/spsa_classify_time_loop_gpu_mex.cu"
check_path "spsa_gpu/src/cuda/spsa_regress_time_loop_gpu_mex.cu"
check_path "spsa_gpu/src/cuda/spsa_dynamics_time_loop_gpu_mex.cu"
check_path "data/raw/abalone_dataset.mat"
check_path "data/raw/breast_cancer_dataset.mat"
check_path "data/raw/mnist.mat"
check_path "data/raw/afro_mnist_vai.mat"
check_path "data/raw/toyota_dataset.mat"
check_path "data/raw/yacht_dataset.mat"

if [ "${missing}" -ne 0 ]; then
    echo
    echo "This ARC bundle is missing required code or dataset files."
    echo "For MNIST/Afro-MNIST jobs, copy mnist.mat and afro_mnist_vai.mat into data/raw/."
    exit 1
fi

echo
echo "Standalone ARC folder looks complete."
echo "Submit from this folder with: bash submit_preferred_gpu.sh"
