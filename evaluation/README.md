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
