# ARC GPU SPSA Jobs

This directory contains the ARC entry point for the standalone GPU SPSA module.
It does not modify or replace the existing e-prop training back-end.

Array tasks:

- `1`: breast cancer classification
- `2`: Yacht Hydrodynamics regression
- `3`: Van der Pol dynamical system

Submit all three tasks with the preferred GPU classes:

```bash
sbatch --partition=gpu-h100,gpu-a100,gpu-l40,gpu-v100 spsa_gpu/arc/submit_spsa_gpu_array.slurm
```

Submit one task by overriding the array range, for example Yacht only:

```bash
sbatch --array=2 --partition=gpu-h100,gpu-a100,gpu-l40,gpu-v100 spsa_gpu/arc/submit_spsa_gpu_array.slurm
```

Checkpointing is enabled by default in the ARC runner. If a job reaches the
checkpoint wall-time guard, only the current array element is resubmitted and
startup checks are skipped for the checkpoint-resumed job. Resubmitted jobs
reuse the current SLURM partition, such as `gpu-v100`, but do not pin to the
original physical node. The default guard is 23 hours before the 24-hour SLURM
limit. Override it with `SPSA_CHECKPOINT_HOURS` if a different cluster time
limit is used.

Set `SPSA_EPOCHS` for a fresh-run epoch override. This value is explicitly
preserved when a checkpointed job resubmits itself.

To continue breast-cancer SPSA fine-tuning for 5,000 more epochs, submit task
1 with the completed source model, added epochs, and a new destination file:

```bash
sbatch --array=1 --partition=gpu-l40 \
  --export=ALL,SPSA_CONTINUE_FROM=outputs/models/classification_BC_lowrank_SPSA_GPU_primary_seed001.mat,SPSA_ADDITIONAL_EPOCHS=5000,SPSA_MODEL_FILE=outputs/models/classification_BC_lowrank_SPSA_GPU_primary_seed001_continued_5000epochs.mat \
  spsa_gpu/arc/submit_spsa_gpu_array.slurm
```

Continuation deliberately resets Adam moments because the saved result stores
the validation-selected model, not an optimizer state corresponding to that
selected model. It retains the prior history and holds the terminal learning
rate and SPSA perturbation size during the added epochs.
