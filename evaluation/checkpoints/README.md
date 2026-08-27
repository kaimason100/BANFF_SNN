# Saved-checkpoint evaluation

This folder contains read-only Live Scripts for inspecting networks saved by the
time-limit checkpoint mechanism before training has finished. The scripts never
resume training, overwrite a checkpoint, or replace a completed model result.

By default, each script:

- locates seeds 1--3 using the current task configuration and experiment profile;
- skips seeds whose checkpoint is not present;
- reconstructs both the state at the checkpoint epoch and the checkpoint's
  validation-selected best state;
- recomputes validation performance independently;
- runs the same statistics and plot pipeline as the completed-model Live
  Scripts, including the matched untrained network, inference-time recurrent
  ablation, hidden-bias swarm, firing-rate/raster/inverse-ISI diagnostics,
  task-specific plots and, for static tasks, full-split current-magnitude
  distributions. Dynamical-system tasks receive the corresponding full-split
  closed-loop current distributions and recurrent-ablation phase portraits; and
- leaves held-out test evaluation disabled to avoid using the test set for
  decisions about unfinished training.

Run the Live Script for the relevant task from MATLAB. Set `profile` to
`"main"`, `"full_rank"`, `"spsa"`, or `"neuron_sweep"` and reproduce any
scientific training overrides. For a neuron sweep, specify `N_hidden` in
`overrides`.

If a checkpoint has a non-current filename hash, enter its full path in
`display_options.checkpoint_files`. Explicit paths bypass filename derivation,
but do not bypass source-provenance or structural validation. Multiple files may
be supplied as a string vector.

Set `display_options.assessment_split` to `"validation"` or `"test"`.
Validation is the default. Test results must not be used for model selection,
hyperparameter tuning, early stopping, or deciding whether to continue a seed.
Set `display_options.checkpoint_state` to `"current"` to inspect the state at
the checkpoint epoch or `"best"` to inspect its validation-selected state. The
progress table always reports both states independently.

All figures are created without a `Name` property. When these scripts are run
in the MATLAB Live Editor, their figures are displayed in the Live Editor output
area according to MATLAB's figure-output setting.

The shared implementation is `banff_evaluate_checkpoints.m`. Task scripts are
provided for breast cancer, MNIST, Afro-MNIST (Vai), Abalone, Toyota, Yacht,
Lorenz, Sprott-S, and Van der Pol.
