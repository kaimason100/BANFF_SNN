# Task-specific held-out evaluation

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
- firing-rate, active/silent-neuron, raster, ISI, voltage/current (static),
  and event-time sensitivity (dynamics) diagnostics.

Classification scripts add confusion, class-count, confidence, and (for the
image datasets) example-image plots. Regression scripts add prediction/truth,
residual, RMSE, correlation and signed-error diagnostics. Dynamical-system
scripts add distance by initial condition, true/network time series, every
pairwise phase portrait, an event raster, and the event-time rho distribution.

The editable first section of each Live Script controls `seeds`, `profile`,
scientific `overrides`, and display/export options. Figures are shown inline by
default. Set `display_options.save_figures = true` to export them beneath
`outputs/evaluation/<task>/`.
