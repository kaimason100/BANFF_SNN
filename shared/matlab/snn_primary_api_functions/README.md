# `snn_primary_api` Private Helpers

This folder contains the internal implementation of `snn_primary_api.m`, split
into one MATLAB function per file.

`shared/matlab/snn_primary_api.m` adds this helper folder to the MATLAB path
at runtime. Scripts should still call the public API rather than these helpers,
which keeps the release interface small:

```matlab
result = snn_primary_api('train_static', 'classification', 'gpu', opts);
result = snn_primary_api('test_dynamics', 'dynamical_systems', 'cpu', opts);
```

The split is intended to make the implementation auditable:

- training entry points: `train_static.m`, `train_dynamics.m`
- testing entry points: `test_static.m`, `test_dynamics.m`
- CPU time loops: `static_epoch_cpu.m`, `dynamics_epoch_cpu.m`
- GPU/MEX wrappers: `*_gpu.m`, `static_mex_name.m`, `mex_source_file.m`
- ARC checkpoint/resume: `arc_*checkpoint*.m`
- shared numerical helpers: `supervised_loss_grad.m`, `pearson_r_p.m`,
  `phase_portrait_wasserstein_distance.m`

Regression helpers keep exact duplicate feature-plus-target rows in one split
and persist realised indices plus dataset provenance. Dynamical-system helpers
keep validation and final-test initial conditions separate and check for exact
overlap before testing. Shared training/evaluation helpers are mirrored in the
ARC bundle; final saved-model tests and publication exports remain local.
