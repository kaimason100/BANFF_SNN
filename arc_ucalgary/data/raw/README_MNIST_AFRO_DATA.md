MNIST and Afro-MNIST (Vai) datasets
=================================

Copy the full external image datasets into this folder before submitting ARC
jobs:

- data/raw/mnist.mat
- data/raw/afro_mnist_vai.mat

Each .mat file must contain `training` and `test` structs:

- `training.images`: 28 x 28 x 1 x N, 28 x 28 x N, 784 x N, or N x 784
- `training.labels`: N labels, either 0:9 or categorical/numeric class labels
- `test.images`: same image layout
- `test.labels`: held-out test labels

Normalisation statistics are fitted on the training subset only. The official
`test` struct is used only for final testing. The training struct is split
80/20 into train/validation, matching the attached non-spiking reference code.
