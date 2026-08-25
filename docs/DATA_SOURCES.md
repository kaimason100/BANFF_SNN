# Dataset sources and integrity

The exact MATLAB files consumed by the release scripts are stored in
`data/raw/`. The ARC standalone bundle contains byte-identical copies.

| File | Source used in manuscript | Licence/status | SHA-256 |
| --- | --- | --- | --- |
| `breast_cancer_dataset.mat` | Breast Cancer Wisconsin (Diagnostic), UCI, DOI `10.24432/C5DW2B` | CC BY 4.0 | `4bf45f311ee0a7efc16b9548bfc68f7841483ce47b27dd09ecfe2609abd06d62` |
| `mnist.mat` | MNIST Database of Handwritten Digits, DOI `10.24432/C53K8Q` | Follow the original MNIST acknowledgement and usage terms | `5eee8657f3c5a853f923077fab5e01215a342cae8f989a67a2d703cf9adc65fa` |
| `afro_mnist_vai.mat` | Afro-MNIST v1, Vai subset, Zenodo DOI `10.5281/zenodo.4050071` | See the Zenodo record's Rights field | `9e77781ca362d3a144a71685cc8e1d7776ddf27353f1e4c6e4cbc2cf5b362f9a` |
| `abalone_dataset.mat` | Abalone, UCI, DOI `10.24432/C55C7W` | CC BY 4.0 | `1f962b467f659af39bfd6f46b192acf5846c44c7ee3f62993e8223a2ec1acb7b` |
| `toyota_dataset.mat` | Toyota subset of the 100,000 UK Used Car Data Set, Kaggle | CC0 Public Domain | `09a0b8c5f600d06756c7eee172ee277d17c654e3a03253fcb00ad9e6a93b1528` |
| `yacht_dataset.mat` | Yacht Hydrodynamics, UCI, DOI `10.24432/C5XG7R` | CC BY 4.0 | `df92835406c002b2c1550b52d095085c86e6a8bbdbf3dc790b4d780dd9057d03` |

The code fits preprocessing statistics on training data only. Regression data
are split deterministically using seed 42; exact feature-plus-target duplicate
rows are grouped so that a duplicate group cannot cross partitions. MNIST and
Afro-MNIST retain their supplied test sets and split only the supplied training
pool into training and validation sets. Seed-specific model files persist the
realised indices and preprocessing settings used during evaluation.

Attribution requirements apply independently of any licence chosen later for
the authors' source code.
