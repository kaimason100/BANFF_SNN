# Data layout

The active datasets are stored in `data/raw/` with the exact filenames used by
`banff_data.m`:

- `breast_cancer_dataset.mat`
- `abalone_dataset.mat`
- `toyota_dataset.mat`
- `yacht_dataset.mat`
- `mnist.mat`
- `afro_mnist_vai.mat`

`banff_data.m` verifies the raw-file SHA-256 whenever a saved trained model is
reloaded. The publication test suite also checks the distributed dataset hashes.

## Splits

- MNIST and Afro-MNIST keep their official test partitions. The official
  training set is deterministically split 80/20 into train/validation.
- Breast Cancer uses a stratified 60/20/20 split while keeping exact
  feature+label duplicate groups together.
- Regression datasets use 60/20/20 partitions while keeping exact
  feature+target duplicate groups together.

The realized indices and split-policy string are saved with every trained
model. Testing reuses those exact indices.

## Preprocessing

Feature mean/standard deviation and regression target mean/standard deviation
are fitted on the training split only. The exact saved statistics are reused
when testing or rebuilding publication data; the code also checks that they
remain consistent with the verified raw dataset and saved split.

See `../docs/DATA_SOURCES.md` for dataset provenance and citations.
