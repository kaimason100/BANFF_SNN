# ARC GPU Training Scripts

This folder is self-contained for ARC transfer. It contains ARC cluster wrappers
and a bundled copy of the project code needed by those wrappers:
`shared/`, `Classification/`, `Regression/`, `dynamical_systems/`, and `data/`.
The wrappers call the same `snn_primary_api` paths and use the same task
options/model filenames as the local Live Editor GPU scripts. ARC changes are
limited to disabling live plots during execution, enabling text progress,
printing job metadata, and compiling the required active GPU MEX when missing or
stale. ARC is used for training, in-training validation, checkpointing, and
model saving only; run saved-model testing and publication analysis locally.

Current ARC publication defaults are 32,000 spiking neurons with the low-rank
signed-uniform architecture:

```text
SNN_N_HIDDEN=32000
SNN_RECURRENT_MODE=low_rank
SNN_DECODER_MODE=signed
SNN_SIGNED_DECODER_DISTRIBUTION=uniform
SNN_MAX_SPARSE_FULL_RANK_NNZ=20000000
SNN_MAX_FULL_RANK_RECURRENT_BYTES=2684354560
```

The SLURM script exports those defaults and the wrappers also apply them through
`arc_apply_architecture_env`. Set the same `SNN_*` variables explicitly only when
you are intentionally running an architecture ablation. The ARC log prints the
resolved `N_hidden`, recurrent mode, decoder mode, signed-decoder
distribution, full-rank sparsity/storage controls, sparse memory limits, and
final model recurrent storage/edge density.

You can transfer only this `arc_ucalgary` folder to ARC. On ARC, submit from
inside the transferred folder:

```bash
cd /path/to/arc_ucalgary
bash submit_preferred_gpu.sh
```

Submit all GPU training jobs as an array, using the requested preference order:

```bash
bash arc_ucalgary/submit_preferred_gpu.sh
```

If you are already inside the transferred standalone `arc_ucalgary` folder, use:

```bash
bash submit_preferred_gpu.sh
```

Or submit to a specific ARC GPU partition:

```bash
sbatch --partition=gpu-h100 arc_ucalgary/submit_arc_training_array.slurm
sbatch --partition=gpu-a100 arc_ucalgary/submit_arc_training_array.slurm
sbatch --partition=gpu-l40  arc_ucalgary/submit_arc_training_array.slurm
sbatch --partition=gpu-v100 arc_ucalgary/submit_arc_training_array.slurm
```

From inside the standalone transferred folder, omit the `arc_ucalgary/` prefix.

Run one network seed for one task by overriding the array index. The array is
task-major with three seeds per task: array items 1-3 are task 1 seeds 1-3,
items 4-6 are task 2 seeds 1-3, and so on. For example, MNIST seed 1 is item 4:

```bash
sbatch --partition=gpu-h100 --array=4 arc_ucalgary/submit_arc_training_array.slurm
```

Task order:

1. classification BC
2. classification MNIST
3. classification Afro-MNIST (Vai)
4. regression abalone
5. regression car price (Toyota dataset)
6. regression yacht
7. NAR Lorenz
8. NAR Sprott-S
9. NAR Van der Pol

The dedicated static neuron sweep is independent of the general task array,
so its numbering is unchanged by older ARC bundles that still contain removed
tasks. It trains seed 1 at 1k, 2k, 4k, 8k, and 16k neurons; the ordinary task
models supply the 32k points. Array items 1-5 are Yacht Hydrodynamics and
items 6-10 are breast-cancer classification. Yacht defaults to 100,000 epochs
and BC defaults to 5,000 epochs:

```bash
sbatch --array=1-10 --partition=gpu-h100,gpu-a100,gpu-l40,gpu-v100 \
  --export=ALL,SNN_YACHT_SWEEP_EPOCHS=100000,SNN_BC_SWEEP_EPOCHS=5000 \
  arc_ucalgary/submit_arc_static_neuron_sweeps_array.slurm
```

From inside the standalone transferred folder, omit the `arc_ucalgary/`
prefix. Checkpoint resubmissions preserve the selected array item, selected
partition, and both task-specific epoch overrides.

Saved-model provenance note: the Abalone models and the pre-correction Toyota
models contain 25,000 epochs for each of seeds 1--3. The corresponding ARC
wrappers default to 25,000 epochs. Toyota must be rerun from epoch 1 under the
duplicate-grouped split before its results are used. `SNN_EPOCHS` remains
available for intentional alternative runs and is retained by checkpoint
resubmissions.

Regression splitting is fitted from the training partition and keeps every
exact feature-plus-target duplicate group in one partition. This changes the
Toyota split; legacy Toyota models and timeout checkpoints must not be reused.
The shared training code rejects a legacy checkpoint when duplicate groups are
present. Regression datasets without exact duplicate groups retain the earlier
seeded row-wise partition. Dynamical-system validation uses seed 1001, whereas
all final-test initial conditions use seed 123; the local test path rejects any
exact overlap before evaluation.

Long jobs checkpoint automatically at epoch boundaries after 23 hours of
runtime. The checkpoint includes the trained bias vector, best model so far,
training histories, MATLAB RNG state, and the GPU Adam/AMSGrad state held by
the MEX file. The job then submits the same SLURM array item back to the same
partition without pinning it to the original node; the resubmitted checkpoint
job loads the checkpoint and resumes from the next epoch. Final successful
completion removes the checkpoint file and saves the usual seed-specific model
in `outputs/models`.
