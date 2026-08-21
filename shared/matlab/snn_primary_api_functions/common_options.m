% common_options.m
function opts = common_options(mode)
opts = struct();
opts.seed = 42;
opts.init_seed = opts.seed;
opts.split_seed = 42;
opts.dt = single(1e-3);
opts.N_rec = 10;
opts.SCALE = struct('enc', single(2), 'rec', single(0.05), 'dec', single(0.1));
opts.arch = default_arch_options();
opts.NET = struct('p_rec', 1, 'variance_correction', true, ...
    'dale', struct('enable', true, 'p_exc', 0.5, 'sign', []));
opts.neuron = struct('tau_u', single(50e-3), 'tau_w', single(500e-3), ...
    'tau_s_rise', single(2e-3), 'tau_s_decay', single(50e-3), ...
    'E_L', single(-70), 'V_th', single(-50), 'V_reset', single(-65), ...
    'a_param', single(0), 'b_param', single(0.5), ...
    'phi_u', single(1), 'delta_u', single(0.8));
opts.SCHED = struct('type', 'exponential', 'lr_start', single(5e-2), 'lr_end', single(1e-3));
opts.adam = struct('b1', single(0.9), 'b2', single(0.999), 'eps', single(1e-8));
opts.check_tolerance = struct('abs', 5e-4, 'rel', 5e-4);
opts.bench_repeats = 3;
opts.live_plot = struct('enable', false, 'every', 10);
if mode == "bench"
    opts.check_tolerance = struct('abs', inf, 'rel', inf);
end
end
