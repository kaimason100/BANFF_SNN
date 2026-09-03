# Task-specific held-out evaluation

For read-only monitoring of unfinished time-limit checkpoints, use the
task-specific Live Scripts in `checkpoints/`. Those scripts use validation data
by default and compare the current checkpoint state with its validation-selected
best state.

Run the `.mlx` file for the task you want to evaluate. Each Live Script calls
the current publication architecture (`run_experiment("test",...)`, then
`banff`, `banff_test`, `banff_eval`, and `banff_model`) and does not import or
execute the older repository.

Every script reports per-seed values and across-seed mean/sample-SD, audits
the saved seed/model identities and current fixed-network generator, and shows:

- test metric and loss summaries;
- training and validation histories;
- learned-bias swarm plots;
- task-specific held-out plots;
- firing-rate, active/silent-neuron, raster, inverse-ISI instantaneous-rate,
  voltage and trained-versus-untrained per-neuron contribution distributions
  (static; the primary panel uses a common RMS-over-sample-and-time definition
  for net encoder current, net recurrent current, adaptation and bias deviation;
  gross absolute afferent RMS is shown separately to diagnose cancellation, and
  decoder RMS contributions are shown separately in normalized-output units;
  each plotted datum is one hidden neuron, box summaries use all neurons, and
  learned bias is shown after removal of its configured initial baseline),
  and event-time sensitivity (dynamics) diagnostics.

All seed-specific diagnostic figures are generated for every requested seed;
the metric, training-history and learned-bias figures remain across-seed
summaries. Exported seed-specific figures include the seed in their filename so
that one seed cannot overwrite another.

Every task also performs an inference-time recurrent ablation. It retains the
trained bias and all other fixed operators, sets only recurrent drive to zero
in an evaluation copy, and reports paired full-versus-ablated held-out metrics.
For dynamical-system tasks it additionally plots phase portraits for the full
and zero-recurrence networks against their corresponding reference trajectories.
Set `display_options.run_recurrent_ablation = false` to skip this extra pass.

Dynamical-system tasks use the same trained-versus-untrained current-magnitude
definition as static tasks, aggregated over every selected initial condition
and scored closed-loop timestep. Warmup establishes the recurrent state but is
excluded from these magnitudes.

Classification scripts add confusion, class-count, confidence, and (for the
image datasets) example-image plots. Regression scripts add prediction/truth,
residual, RMSE, correlation and signed-error diagnostics. Dynamical-system
scripts add distance by initial condition, true/network time series, every
pairwise phase portrait, an event raster, and the event-time rho distribution.

The editable first section of each Live Script controls `seeds`, `profile`,
scientific `overrides`, and display/export options. Figures are shown inline by
default. Set `display_options.save_figures = true` to export them beneath
`outputs/evaluation/<task>/`.

For completed Lorenz models whose filenames or scientific settings differ from
the current defaults, call `evaluate_saved_lorenz_models` from a Live Script.
With no file argument it opens a multi-file selector; explicit paths may instead
be supplied as a string array. The helper uses every selected model's saved
configuration, performs the complete standard held-out analysis, groups only
models with identical scientific and training-source identities, and prints a
cross-model comparison table. Checkpoint files remain the responsibility of the
separate checkpoint evaluator.

`diagnose_edge_of_chaos.m` is a separate evaluation-only recurrent-regime
diagnostic. It applies no external input, initializes voltages randomly around
threshold, resolves the initial recurrent cascade over a broad gain sweep, and
measures extinction, recurrent E/I cancellation, adaptation, persistence and
finite-time separation of nearby trajectories. A longer 2-by-2-by-2 factorial
crosses recurrence, adaptation and bias; selected-gain and initial-volley-density
controls test whether the apparent transition depends on gain or initialization.
These quantities help locate a useful operating boundary but are not a formal
proof of chaos.

`calibrate_matrix_scaling.m` uses only training data and the production GPU
timestep to recommend encoder, recurrent and decoder gains. It retains the
analytic `1/sqrt(D)`, `1/sqrt(N)`, and low-rank/full-rank fan-in factors, then
matches declared task-driven encoder-current RMS, net recurrent-to-encoder RMS,
and normalized decoder-output SD targets. It supports both static and dynamical
tasks and prints conservative/central/strong sensitivity settings. Calibration
targets are modelling choices and must be frozen before held-out testing.

`banff_search_breast_cancer_recurrent_regime.m` performs a validation-gated,
multi-fidelity search for a breast-cancer operating regime in which encoder and
signed-net recurrent RMS currents are comparable and zeroing recurrence causes
a material accuracy loss. Every candidate enforces `decoder_gain ==
recurrent_gain`. A smaller-network grid screen is followed by configurable
32k-neuron, multi-seed confirmation. The held-out test set is not evaluated
until one target-size candidate satisfies the predeclared accuracy, current
balance, ablation-effect and seed-replication criteria. The winning regime is
then passed to the standard complete evaluator.
Open `search_breast_cancer_recurrent_regime.mlx` to configure and run this
analysis interactively. It leaves all progress, per-run and aggregate tables,
figures, held-out evaluation outputs and validation-derived recommendations in
the Live Editor output panel; the `banff_search_...` function remains the single
implementation used by the Live Script and the thin same-named `.m` batch
launcher.
When the repository itself is inside OneDrive on Windows, this search defaults
to `%LOCALAPPDATA%/BANFF_SNN_pub/regime_search/breast_cancer` so HDF5-backed
`-v7.3` model files are not exposed as partially synchronised cloud
placeholders. ARC/Linux retains repository-relative output. A small v7.3
save/load round-trip is performed before any expensive candidate training.

## Delayed cue-response recurrence task

`delayed_cue` is an experimental temporal classification task on the
`codex/network-initialisation-debugging` branch. A signed binary cue is shown
for 300 ms, followed by a silent 150-ms delay and a 300-ms, class-independent
response signal. The cue channel is exactly zero throughout the delay and
response window. The configurable distractor channel is retained but has zero
amplitude by default. It uses the same default network size, low-rank structure, matrix gains,
neuron parameters, and random initialization rules as the other `main` tasks;
there is no task-specific connectivity mask.
Opposite-cue trials are generated in matched pairs. If distractors are enabled
in a later experiment, each pair receives an identical distractor sequence, so
the cue-removed control remains exactly unbiased.

Train and evaluate it with:

```matlab
run_experiment("train","delayed_cue","main",1:3);
report = evaluate_delayed_cue_models(1:3);
```

The evaluator compares the intact network with zero recurrent current and a
cue-removed control. It declares recurrent dependence only when each seed has
at least 80% intact accuracy, loses at least 20 percentage points after
recurrent ablation, and remains at or below 60% when the cue is removed. These
are empirical criteria: because adaptive neurons possess intrinsic state, task
structure alone cannot prove that synaptic recurrence was used. The predeclared
zero-recurrence control supplies that operational test. The default 150-ms
delay is three synaptic decay time constants.
