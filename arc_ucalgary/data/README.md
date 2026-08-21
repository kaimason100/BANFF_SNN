# Data Layout

## External Data

Put external datasets in `data/raw/`.

Expected dataset filenames used by the ARC training wrappers:

- `breast_cancer_dataset.mat`
- `abalone_dataset.mat`
- `toyota_dataset.mat`
- `yacht_dataset.mat`
- `mnist.mat`
- `afro_mnist_vai.mat`

MNIST and Afro-MNIST (Vai) are loaded from local `.mat` files with predefined `training` and `test` structs.

For tabular regression, normalisation is fitted on the training partition.
Exact duplicate feature-plus-target rows are assigned as one group, so they
cannot cross training, validation, and test partitions. The realised split
indices, split policy, duplicate-group count, and source-file SHA-256 are
retained in saved metadata; local testing reuses those indices and checks the
hash.

## Derived Data

- Use `data/raw/` for manually imported intermediate data.
- Use `data/processed/` for cleaned datasets if you later add preprocessing code.
- Use `data/derived/` for derived artifacts that are not final paper figures or result bundles.

## Outputs

ARC jobs create and use these output folders from the repository or standalone
ARC bundle root:

- `outputs/models/` for completed seed-specific training results.
- `outputs/checkpoints/` for wall-time checkpoint/resume state.
- `outputs/arc_logs/` for MATLAB diary logs.
- `outputs/arc_slurm/` for SLURM stdout/stderr when the submit script writes
  logs into that folder.
