# Publication figures

The scripts in `scripts/` preserve the plotting layout, labels, axis choices and
styling from the original reference publication figure sources supplied during the code audit. They were converted from Live Scripts to plain `.m` so the
source is visible and version-control friendly.

Only infrastructure was changed where necessary: paths now point to this small
package, tested results are read from `outputs/publication_analysis/`, and large
deterministic fixed matrices are regenerated from the saved seed/configuration
rather than duplicated in every analysis file.

The loaders accept current schema-5 publication exports only and verify that
every saved seed has the same scientific-configuration fingerprint. Regenerate
the publication exports after training with this release; obsolete exports are
skipped rather than mixed with updated model results.

`plot_ALIF_dynamics.m` is the one deliberate scientific update: it retains the
original stimulus and plotting design but obtains its forward and eligibility
traces from the corrected `banff_model` reference timestep. It therefore does
not reproduce the historical incorrect eligibility recurrence.

Exact original generated images are copied unchanged into
`reference_original/` so the regenerated figures can be visually checked.

Typical workflow:

1. Train and test the required seeds with `run_experiment.m`.
2. Export the compact result with `banff_publication(results)`.
3. Run the corresponding script in `scripts/`.

Small result loaders are in `matlab/`. The retained third-party colormap helper
and its licence are in `third_party/`. New generated figures are written beneath
`outputs/figures/`.
