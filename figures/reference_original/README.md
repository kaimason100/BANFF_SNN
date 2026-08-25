# Original figure references

These image/PDF-composition reference files are copied unchanged from the
original reference repository supplied during the code audit. They are included only so
that the publication figure scripts can be visually checked against the exact
original layouts.

The scripts in `../scripts/` preserve the original plotting/layout code as far
as possible, while their data loading and replay calls are adapted to the
publication-ready BANFF result format. `plot_ALIF_dynamics.m` deliberately uses
the corrected `banff_model` neuron/eligibility equations rather than reproducing
the obsolete eligibility calculation from the historical script; its stimulus
and plotting/layout code remain the original design.
