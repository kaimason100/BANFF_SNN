# Separate GPU SPSA Module

This folder is an additive GPU SPSA implementation. It does not edit or replace
the active e-prop backend. The CUDA files in `src/cuda/` are copies of the
current release MEX sources and are compiled under separate names:

- `spsa_classify_time_loop_gpu_mex`
- `spsa_regress_time_loop_gpu_mex`
- `spsa_dynamics_time_loop_gpu_mex`

The SPSA update remains MATLAB-side. The expensive black-box loss probes run
through the copied resident GPU MEX files by setting `B+c*delta` and
`B-c*delta` with `update_bias`, then running forward-only loss commands.

## Checks

Before training, `spsa_gpu_run_task` runs equivalence checks by default:

- Builds the same model with `make_primary_model`.
- Verifies that fixed model fields do not change during SPSA training.
- Compares copied SPSA GPU MEX losses and outputs against the CPU reference on
  small deterministic subsets.
- If the existing release GPU MEX is already compiled, also compares copied
  SPSA GPU MEX outputs against the release GPU MEX.

The only intended algorithmic difference from the release scripts is the
hidden-bias update rule: SPSA loss probing plus the existing Adam bias update,
instead of e-prop gradients.

## Local Use

```matlab
repo_root = spsa_gpu_add_paths;
compile_spsa_gpu_mex;
result = spsa_gpu_run_task("yacht");
```

Supported proof tasks are `bc`, `yacht`, and `vanderpol`.

Fresh-run defaults are 5,000 epochs for BC, 50,000 for Yacht, and 200,000 for
Van der Pol. The BC result used by the current saved publication analysis is a
5,000-epoch base model followed by a 45,000-epoch fine-tuning phase (50,000
cumulative epochs).

### Continue a saved model

Use the explicit continuation helper to fine-tune a completed saved result:

```matlab
result = spsa_gpu_continue_task("bc", ...
    "outputs/models/classification_BC_lowrank_SPSA_GPU_primary_seed001.mat", 5000);
```

The helper writes a new result file by default. Continuation starts from the
validation-selected bias vector, preserves the previous history, resets Adam's
moments, and holds the original terminal learning rate and SPSA perturbation
size for the added epochs. It is therefore a clearly delineated fine-tuning
phase rather than an exact continuation of the original Adam trajectory.

After copying ARC results back locally, the dedicated SPSA test scripts follow
recorded `source_file` links and test the terminal continuation model for each
requested seed. The selected result contains one cumulative history; training
plots mark every continuation boundary with a labelled dashed line and the
resolver prints total epochs and phase boundaries. When several branches
descend from one source model, the resolver selects the terminal branch with
the longest cumulative history and warns about the alternatives. A single
linear continuation chain per seed remains the clearest and recommended
workflow.

## ARC Use

From the repository root on ARC:

```bash
sbatch --array=1-3 --partition=gpu-h100,gpu-a100,gpu-l40,gpu-v100 spsa_gpu/arc/submit_spsa_gpu_array.slurm
```

Array mapping:

1. breast-cancer classification
2. Yacht Hydrodynamics regression
3. Van der Pol dynamical system

The SPSA GPU module has its own checkpoint/resubmit path under
`outputs/checkpoints/`, independent of the release backend checkpoint files.
Checkpoint-triggered jobs are resubmitted to the same partition selected for
the running job, but are not pinned to the same physical node.

Fresh ARC runs can override the task default with `SPSA_EPOCHS`, for example
`SPSA_EPOCHS=50000` for the Yacht proof. The override is preserved across
checkpoint-triggered resubmissions.

For an ARC continuation, set both continuation variables and a distinct output
file. Checkpoint-triggered resubmissions preserve these variables:

```bash
sbatch --array=1 --partition=gpu-l40 \
  --export=ALL,SPSA_CONTINUE_FROM=outputs/models/classification_BC_lowrank_SPSA_GPU_primary_seed001.mat,SPSA_ADDITIONAL_EPOCHS=5000,SPSA_MODEL_FILE=outputs/models/classification_BC_lowrank_SPSA_GPU_primary_seed001_continued_5000epochs.mat \
  spsa_gpu/arc/submit_spsa_gpu_array.slurm
```
