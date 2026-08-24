MNIST and Afro-MNIST (Vai) datasets
=================================

The publication package already contains:

- `data/raw/mnist.mat`
- `data/raw/afro_mnist_vai.mat`

Each file contains `training` and `test` structs:

- `training.images`: 28 x 28 x 1 x N, 28 x 28 x N, 784 x N, or N x 784
- `training.labels`: N labels, either 0:9 or numeric/categorical class labels
- `test.images`: same image layout
- `test.labels`: held-out test labels

`banff_data.m` keeps the supplied test partition untouched and deterministically
splits only the supplied training partition 80/20 into training/validation.
Feature normalisation is fitted on the training subset only. The ARC Slurm jobs
run from the repository root and use these same `data/raw/` files; no duplicate
ARC data directory is required.
