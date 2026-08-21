# Data Layout

## External Data

Put external datasets in `data/raw/`.

Expected dataset filenames used by the current scripts:

- `breast_cancer_dataset.mat`
- `abalone_dataset.mat`
- `toyota_dataset.mat`
- `yacht_dataset.mat`
- `mnist.mat`
- `afro_mnist_vai.mat`
- `idx_neuron_FC_plot.mat`

MNIST and Afro-MNIST (Vai) are loaded from local `.mat` files with predefined `training` and `test` structs.

For tabular regression, normalisation is fitted on the training partition.
Exact duplicate feature-plus-target rows are assigned as one group, so they
cannot cross training, validation, and test partitions. The realised split
indices, split policy, duplicate-group count, and source-file SHA-256 are
retained in saved metadata; testing reuses those indices and checks the hash.

## Derived Data

- Use `data/raw/` for manually imported intermediate data.
- Use `data/processed/` for cleaned datasets if you later add preprocessing code.
- Use `data/derived/` for derived artifacts that are not final paper figures or result bundles.

## Outputs

Active training, testing, ARC checkpointing, and publication analysis write beneath the project `outputs/` directory. The package also provides:

- `outputs/repro/`
- `outputs/fc/`
- `outputs/batch/`

These folders are intended for auxiliary reproducibility, functional-connectivity, and batch artifacts. Use `prepare_run_environment('repro')`, `prepare_run_environment('fc')`, or `prepare_run_environment('batch')` only for scripts that explicitly use those output locations.
