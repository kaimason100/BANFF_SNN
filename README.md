# BANFF SNN

MATLAB code for **bias-only learning in a fixed-synapse recurrent spiking
network**. The principal model uses adaptive leaky integrate-and-fire (ALIF)
neurons, a fixed factorised recurrent scaffold and a fixed linear decoder.
**Only the per-neuron bias vector is trained.**

The repository is deliberately kept small. Training and testing share one
forward simulator (`banff_model.m`); there are no separate CPU/GPU scientific
implementations, CUDA files, MEX files, or task-specific training scripts.

## Start here

The scientific core is eight readable MATLAB files:

- `banff.m` — public entry point and all experiment configuration.
- `banff_train.m` — training loops, checkpointing and progress reporting.
- `banff_test.m` — held-out model evaluation.
- `banff_eval.m` — losses and validation/test protocols shared by train/test.
- `banff_model.m` — **the single network simulator**: fixed network creation,
  ALIF/LSTI neuron dynamics, synaptic filtering, eligibility and conventional
  bias-corrected AMSGrad.
- `banff_data.m` — datasets, deterministic splits, preprocessing and target
  dynamical systems.
- `banff_metrics.m` — classification, regression and phase-portrait metrics.
- `banff_publication.m` — compact figure-facing export after testing.

`run_experiment.m` is a short convenience launcher. `banff_plot.m` is only a
figure/replay adapter; it calls the readable reference timestep inside
`banff_model.m` and does not define another neuron model.

The code-side mathematical definition is in
[`docs/METHODS_ELIGIBILITY.md`](docs/METHODS_ELIGIBILITY.md).

## Requirements

- MATLAB R2023a or newer.
- Parallel Computing Toolbox and a supported NVIDIA GPU for training/testing.
- The principal 32,000-neuron experiments require substantial GPU memory.
- Some publication figures additionally use Signal Processing Toolbox and
  Statistics and Machine Learning Toolbox.

No CUDA, MEX or C++ source is required.

## Principal learning rule

The publication default is

```matlab
cfg.eligibility_mode = "hard_spike";
cfg.hard_event_gain = single(1);  % 1/mV
```

For the local bias-sensitivity state

\[
\epsilon^u=\partial u/\partial B,\qquad
\epsilon^w=\partial w/\partial B,
\]

the complete coupled membrane sensitivity is propagated as

\[
\epsilon^u_{new}
=a\epsilon^u_{old}+(1-a)(1-\epsilon^w_{old}).
\]

At a real LSTI spike, the sensitivity is first propagated to the inferred
event time \(\rho\). The raw event eligibility is

\[
e^{raw}=s\,\phi\,\epsilon^{u,-},\qquad
\phi=1\;\mathrm{mV}^{-1}.
\]

The reset branch and \(\rho\) are treated as stop-gradient. The same event is
inserted at \(\rho\) into the two-stage rise-decay eligibility filter matched
to the decoder's filtered-spike state. A non-spiking timestep still propagates
the local membrane/adaptation sensitivities and decays the eligibility states,
but injects no new hard event.

This is an **event-gated e-prop-style rule for intrinsic bias plasticity**. It
should not be described as the canonical Bellec synaptic ALIF eligibility
formula.

### Optional continuous-surrogate ablation

For comparison only:

```matlab
options.eligibility_mode = "surrogate";
options.surrogate_peak = single(0.7);       % 1/mV
options.surrogate_half_width = single(10);  % mV
```

These values are inactive in hard-spike mode and therefore do not change the
hard-spike scientific-configuration fingerprint.

## Running experiments

Principal 32k low-rank experiment, three seeds:

```matlab
run_experiment("train", "lorenz", "main", 1:3);
results = run_experiment("test", "lorenz", "main", 1:3);
analysisFile = banff_publication(results);
```

Other profiles:

```matlab
run_experiment("train", "yacht", "full_rank", 1);
run_experiment("train", "breast_cancer", "spsa", 1);
run_experiment("train", "lorenz", "neuron_sweep", 1, ...
    struct('N_hidden', 8000));
```

Available tasks:

`breast_cancer`, `mnist`, `afro_mnist_vai`, `abalone`, `toyota`, `yacht`,
`lorenz`, `sprott_s`, `vanderpol`.

Inspect the complete resolved configuration without requiring a GPU:

```matlab
cfg = banff("config", "lorenz", struct());
```

## Main epoch counts

| Task | Main e-prop epochs |
|---|---:|
| Breast Cancer | 5,000 |
| MNIST | 1,000 |
| Afro-MNIST (Vai) | 1,000 |
| Abalone | 25,000 |
| Toyota | 25,000 |
| Yacht | 25,000 |
| Lorenz | 100,000 |
| Sprott S | 100,000 |
| Van der Pol | 100,000 |

Supplementary full-rank, SPSA and neuron-sweep overrides are centralised in
`banff.m` rather than hidden in separate scripts.

The breast-cancer SPSA control is one continuous 50,000-epoch optimisation.
Its learning-rate and SPSA-perturbation schedules both span all 50,000 epochs;
the optimiser state is not reset during training.

## Static-task training semantics

`batch_size=256` is a **GPU memory batch**, not an optimiser minibatch. Gradients
are accumulated over the complete training set and one conventional AMSGrad
update is performed per epoch. The running maximum is taken over the raw
second-moment accumulator before applying the current Adam bias correction.
Static training and validation arrays remain GPU-resident;
the batch size therefore controls simulator-state memory rather than repeated
host-to-device transfers.

The static simulator averages the filtered hidden state over the readout window
before applying the linear decoder. The dynamics simulator precomputes encoder
currents for contiguous teacher-forced blocks, fuses current components inside
the element-wise neuron kernel, and omits trajectory storage when callers request
only loss and gradient. These reorderings retain the same model equations but
may produce normal single-precision reduction-order differences.

Feature normalisation and regression-target normalisation are fitted only on
the training partition. Exact duplicate feature-target rows are kept in one
split. Saved split indices and normalisation statistics are replayed at test
time, and raw dataset contents are protected by SHA-256 checks.

## Dynamical-system protocol

Each epoch samples one contiguous normalised target-system window. Inputs use a
30 ms teacher-forced / 55 ms closed-loop repeating schedule, while supervision
always remains the next reference state. Validation is fully closed loop and
selects the bias vector with the lowest phase-portrait sliced-Wasserstein
score. Final testing uses a 5 s network-only warmup, then starts the matched
true trajectory from the terminal network output.

For dynamical-system evaluation, `initial_condition_jitter` is the maximum
absolute per-coordinate perturbation: random initial conditions are sampled
uniformly from `[-initial_condition_jitter,+initial_condition_jitter]` around
the reference state. Burn-in is endpoint-inclusive and retains the sample at
exactly `burn_in_time`.

## Reproducibility

A model filename contains a 12-character SHA-256 fingerprint of its scientific
configuration. Saved results also contain the full resolved configuration,
seed-independent scientific fingerprint, checkpoint fingerprint, MATLAB/GPU
information, dataset/split metadata and SHA-256 hashes of the scientific core.

Checkpoint continuation is refused if either the checkpoint configuration or
core source hashes differ. Testing likewise refuses to evaluate a model when
the mathematical simulator/data/evaluation source differs from the source used
for training.

## Tests

There is deliberately **one test file**, not a separate test framework:

```matlab
addpath('tests');
run_tests("quick");
```

Before final manuscript runs on ARC:

```matlab
run_tests("full");
```

The quick checks cover data hashes/splits, the factorised recurrent operator,
the corrected local-state finite-difference derivative, LSTI event-time
sensitivity, phase metric and target-system integration. The full checks add
CPU/GPU execution agreement for the shared scalar timestep, checkpoint/restart,
all nine train/test smoke paths, and tiny surrogate/full-rank/SPSA checks.

## Publication figures

The plotting **layouts, labels, axes and styling were restored from the
original reference repository supplied during the code audit**. They are plain `.m`
files in `figures/scripts/` so the plotting code is readable and diffable.
Only data-loading/replay plumbing was adapted to the compact publication result
format.

Figure loaders accept schema-5 exports from this release and verify the saved
scientific fingerprint across seeds, so older analyses cannot be selected in
place of updated-model results.

The one intentional scientific exception is `plot_ALIF_dynamics.m`: its
original stimulus and visual layout are retained, but the traces are generated
by the corrected `banff_model` equations rather than reproducing the obsolete
historical eligibility recurrence.

Copies of the original generated figure images are included unchanged in
`figures/reference_original/` for visual comparison.

Workflow:

```matlab
results = run_experiment("test", "lorenz", "main", 1:3);
banff_publication(results);
% Then run the required script in figures/scripts/
```

## Repository layout

```text
BANFF_SNN/
├── README.md
├── banff.m
├── banff_train.m
├── banff_test.m
├── banff_eval.m
├── banff_model.m
├── banff_data.m
├── banff_metrics.m
├── banff_publication.m
├── banff_plot.m
├── run_experiment.m
├── tests/run_tests.m
├── examples/simulate_random_network_activity.m
├── data/
├── arc_ucalgary/
├── figures/
└── docs/
```

Task-specific held-out evaluation Live Scripts are in `evaluation/`. They use
the current publication API and architecture while restoring task-specific
tables, plots and spiking diagnostics.

## ARC / University of Calgary

From the repository root:

```bash
bash arc_ucalgary/submit_all.sh
```

The Slurm job loads MATLAB R2023a, prints source hashes/GPU information and
resubmits a checkpointed item only to the partition that ran its first segment,
while retaining the multi-partition hierarchy for initial submissions. See
`arc_ucalgary/README_SIMPLIFIED.md`.

## Public release

Dataset provenance is documented in `docs/DATA_SOURCES.md`; third-party notices
are in `docs/THIRD_PARTY_NOTICES.md`. This package does not choose a software
licence or author list on your behalf; add the final `LICENSE`, citation/DOI and
manuscript-specific metadata before public deposition.
