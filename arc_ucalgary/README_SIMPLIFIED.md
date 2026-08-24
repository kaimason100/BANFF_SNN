# University of Calgary ARC

The ARC jobs run the same MATLAB `gpuArray`/`arrayfun` implementation as the
local package. No CUDA compiler or MEX build is required. The principal e-prop
jobs use the default `hard_spike` event-gated eligibility rule defined in
`docs/METHODS_ELIGIBILITY.md`.

Before expensive manuscript runs, run the full regression suite once on the
intended MATLAB/GPU environment:

```bash
matlab -batch "addpath('tests'); run_tests('full')"
```

From the repository root, submit every manuscript array (48 jobs total):

```bash
bash arc_ucalgary/submit_all.sh
```

The submitter allows Breast Cancer and the regression tasks onto V100 as well
as H100/A100/L40. MNIST, Afro-MNIST and the dynamical-system tasks are kept off
V100.

Individual supplementary arrays can be submitted with:

```bash
BANFF_ARCHITECTURE=full_rank sbatch --array=1-3 arc_ucalgary/submit_arc.slurm
BANFF_ARCHITECTURE=spsa sbatch --array=1-3 arc_ucalgary/submit_arc.slurm
BANFF_ARCHITECTURE=neuron_sweep sbatch --array=1-15 arc_ucalgary/submit_arc.slurm
```

Each job checkpoints after 23 hours. A checkpoint exits with code 75 and the
Slurm script requeues the same array item. `submit_arc.slurm` explicitly loads
`matlab/r2023a` and prints core source hashes plus `nvidia-smi` information.
Saved result files additionally contain MATLAB/GPU provenance and the
scientific-configuration fingerprint.
