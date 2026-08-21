# BANFF SNN

Reproducibility code and data for **Versatile Learning without Synaptic
Plasticity in a Spiking Neural Network**.

This release contains the files required to retrain, evaluate and regenerate
the analyses and figures reported in the manuscript. Generated models,
checkpoints, timestamped analyses, compiled MEX binaries, figures, caches,
archives and unrelated side projects are deliberately not included.

## Requirements

- MATLAB with the Parallel Computing Toolbox
- A supported CUDA toolkit and C++ compiler for GPU training
- Sufficient GPU memory for the 32,000-neuron principal experiments
- Signal Processing Toolbox for `findpeaks` in the return-map analysis

The source was statically reviewed in the release workspace. MATLAB and CUDA
were not executed as part of preparing this GitHub upload. Exact bitwise
agreement across MATLAB, CUDA, driver and GPU versions is not expected.

## Repository contents

- `Classification/`: Breast Cancer, MNIST and Afro-MNIST (Vai) training and testing
- `Regression/`: Abalone, Toyota car-price and Yacht training and testing
- `dynamical_systems/`: Lorenz, Sprott S and Van der Pol training and testing
- `shared/matlab/`: common model, learning, evaluation and analysis implementation
- `shared/plotting/publication_figures/`: manuscript figure notebooks
- `spsa_gpu/`: SPSA proof-of-principle experiments
- `arc_ucalgary/`: standalone ARC/SLURM reproduction bundle
- `data/raw/`: exact benchmark MAT files used by the scripts
- `tests/`: architecture sanity checks
- `DATA_SOURCES.md`: dataset provenance, licences and hashes
- `SHA256SUMS`: release-file integrity manifest

## Principal experiments

The manuscript reports three initialisation seeds for each principal task.
The authoritative batch route is the ARC array:

```bash
cd arc_ucalgary
bash submit_preferred_gpu.sh
```

The array order is:

1. Breast Cancer classification
2. MNIST classification
3. Afro-MNIST Vai classification
4. Abalone regression
5. Toyota car-price regression
6. Yacht regression
7. Lorenz dynamics
8. Sprott S dynamics
9. Van der Pol dynamics

Each task runs seeds 1, 2 and 3. The ARC README documents array indices,
checkpoint/resume behaviour and deliberate environment overrides.

For a local run, open MATLAB in the repository root and run:

```matlab
setup_project_paths
```

Compile the current CUDA sources using the appropriate Live Editor build
notebook:

- `Classification/build/compile_classification_gpu_mex.mlx`
- `Regression/build/compile_regression_gpu_mex.mlx`
- `dynamical_systems/build/compile_dynamical_systems_gpu_mex.mlx`

Then run the matching `GPU_implementation_*.mlx` training notebook. Run it
once for each required `opts.seed` value. Training writes seed-specific files
under `outputs/models/`; that directory is intentionally generated locally.

## Evaluation and analysis

After all required models have been generated, run the corresponding
`TEST_GPU_implementation_*.mlx` notebook. Principal-task test notebooks load
seeds 1, 2 and 3, restore the validation-selected bias vector, reuse the saved
partition/preprocessing metadata and save publication-analysis MAT files under
`outputs/publication_analysis/`.

Before accepting GPU results, run:

- `Classification/test/CHECK_GPU_vs_CPU_classification_small.mlx`
- `Regression/test/CHECK_GPU_vs_CPU_regression_small.mlx`
- `dynamical_systems/test/CHECK_GPU_vs_CPU_dynamical_systems_small.mlx`
- `tests/run_architecture_sanity_tests.m`

These checks are tolerance-based, not bitwise-equivalence claims.

## Supplementary experiments

- Full-rank 6,000-neuron proofs: `arc_ucalgary/submit_arc_full_rank6k_array.slurm`
- Lorenz neuron-count sweep: `arc_ucalgary/submit_arc_lorenz_neuron_sweep_array.slurm`
- Breast Cancer/Yacht neuron-count sweeps: `arc_ucalgary/submit_arc_static_neuron_sweeps_array.slurm`
- SPSA proofs: `spsa_gpu/arc/submit_spsa_gpu_array.slurm`

The ordinary 32,000-neuron principal models provide the 32k points in the
neuron-count panels.

## Figure regeneration

Run the full test notebooks first so that current analysis files exist. Then
use the notebooks in `shared/plotting/publication_figures/`:

- Figure 1: `plot_flowchart_publication_figure.mlx`
- Figure 2 activity panels: `plot_neural_activity_publication_panels.mlx`
- Figure 3 timing redistribution: `analyse_jitter.mlx`, then `plot_jitter.mlx`
- Supplementary Figures 1-3: `plot_phase_portraits_return_maps_poincare_sections.mlx`
- Supplementary Figure 4: `plot_full_rank_publication_figure.mlx`
- Supplementary Figure 5: `plot_spsa_publication_figure.mlx`
- Supplementary Figure 6: `plot_lorenz_neuron_count_publication.mlx`
- Supplementary Figure 7: `plot_dynamics_ic_variation_publication.mlx`

Publication loaders currently select the newest matching complete analysis
file. For an archival release, record and verify the selected file hashes in
addition to the repository commit.

## Reproducibility boundaries

- Only the hidden-neuron bias vector is trained in the main experiments.
- The main learning rule is an approximate e-prop-style local
  surrogate-gradient rule, not exact BPTT.
- Toyota models produced before duplicate-grouped splitting must not be reused.
- Dynamical-system validation and final-test initial conditions use separate
  seeds and are checked for exact overlap.
- Compiled binaries are not distributed; build them from the included CUDA
  sources so the executable corresponds to this release.

Third-party attribution is recorded in `THIRD_PARTY_NOTICES.md` and
`DATA_SOURCES.md`. No licence for the authors' original source code is granted
by this release unless a separate `LICENSE` file is added by the copyright
holders.
