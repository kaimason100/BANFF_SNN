% default_dynamics_options.m
function opts = default_dynamics_options(backend, mode)
opts = common_options(mode);
opts.system_name = 'vanderpol';
opts.T_sim = single(5);
opts.long_sim_time = single(2000);
opts.burn_in_time = single(10);
opts.train_blocks = 1;
opts.closed_loop_validate_every = 100;
opts.closed_loop_validation_time = single(50);
opts.closed_loop_validation_ics = 5;
opts.closed_loop_validation_plot = struct( ...
    'enable', false, ...
    'every', 1, ...
    'max_initial_conditions', 5, ...
    'max_points', 2000);
opts.closed_loop_test_time = single(50);
opts.closed_loop_test_warmup_time = single(5);
opts.closed_loop_test_ics = 5;
opts.closed_loop_ic_jitter = single(0.01);
opts.closed_loop_ic_seed = 1001;
opts.closed_loop_test_ic_seed = 123;
opts.closed_loop_ic_include_reference = true;
opts.closed_loop_test_include_reference = false;
opts.recompute_dynamics_norm = false;
opts.allow_legacy_diagnostic_gpu_eval = false;
opts.wd = struct('NumProjections', 128, 'TrimFraction', 0.10, ...
    'Subsample', 5, 'TransientFraction', 0.10, 'MaxPoints', 1250);
opts.dyn_sys_rate = 8;
opts.use_multistep = true;
opts.W_warmup = round(0.030 / opts.dt);
opts.H_free = round(0.055/opts.dt);
if mode == "train" && backend == "gpu"
    opts.N_hidden = 32000;
    opts.epochs = 2e5;
else
    opts.N_hidden = 128;
    opts.epochs = 3;
end
if mode == "check" || mode == "bench"
    opts.N_hidden = 32;
    opts.N_rec = 4;
    opts.T_sim = single(0.030);
    opts.long_sim_time = single(0.100);
    opts.burn_in_time = single(0);
    opts.train_blocks = 1;
    opts.closed_loop_validate_every = 1;
    opts.closed_loop_validation_time = single(0.030);
    opts.closed_loop_test_time = single(0.030);
    opts.closed_loop_test_warmup_time = single(0);
    opts.epochs = 1;
end
end
