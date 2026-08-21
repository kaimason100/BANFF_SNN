% dynamics_training_metadata.m
function meta = dynamics_training_metadata(x, opts)
%DYNAMICS_TRAINING_METADATA Save normalization and trajectory conventions.
meta = struct();
meta.system_name = char(get_opt(opts, 'system_name', 'unknown'));
try
    sys = make_dynamics_system_for_api(meta.system_name);
    meta.system_params = sys.params;
    meta.system_dim = sys.dim;
    meta.x0_default = single(sys.x0_default(:));
catch
    meta.system_params = struct();
    meta.system_dim = NaN;
    meta.x0_default = single([]);
end
if isstruct(x) && isfield(x, 'mu') && isfield(x, 'sigma')
    meta.mu = single(x.mu);
    meta.sigma = single(x.sigma);
elseif isfield(opts, 'dynamics_mu') && isfield(opts, 'dynamics_sigma')
    meta.mu = single(opts.dynamics_mu);
    meta.sigma = single(opts.dynamics_sigma);
else
    meta.mu = single([]);
    meta.sigma = single([]);
end
meta.dt = single(get_opt(opts, 'dt', NaN));
meta.T_sim = single(get_opt(opts, 'T_sim', NaN));
meta.long_sim_time = single(get_opt(opts, 'long_sim_time', NaN));
meta.burn_in_time = single(get_opt(opts, 'burn_in_time', 0));
meta.integrator = char(get_opt(opts, 'integrator', 'euler'));
meta.dyn_sys_rate = single(get_opt(opts, 'dyn_sys_rate', NaN));
meta.split_seed = double(get_opt(opts, 'split_seed', get_opt(opts, 'seed', NaN)));
meta.init_seed = double(get_opt(opts, 'init_seed', get_opt(opts, 'seed', NaN)));
meta.sample_convention = 'endpoint_inclusive';
meta.num_samples = round(double(meta.T_sim) / double(meta.dt)) + 1;
meta.num_transitions = meta.num_samples - 1;
if isstruct(x) && isfield(x, 'train_blocks')
    meta.train_blocks = x.train_blocks;
end
end
