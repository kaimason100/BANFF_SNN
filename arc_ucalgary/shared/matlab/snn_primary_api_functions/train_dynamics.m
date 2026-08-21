% train_dynamics.m
function result = train_dynamics(backend, opts)
opts = merge_options_with_seed(default_dynamics_options(backend, "train"), opts);
if isfield(opts, 'seed_list') && numel(opts.seed_list) > 1
    result = train_seed_list(@(one_opts) train_dynamics(backend, rmfield_if_present(one_opts, 'seed_list')), opts);
    return;
end
opts.dynamics_split = 'train';
[x, P] = make_dynamics_problem(opts);
lambda = make_lambda_sequence_for_data(x, opts);
validate_dynamics_data(x, lambda, 'train_dynamics');
if backend == "gpu"
    result = train_dynamics_gpu(x, lambda, P, opts);
else
    result = train_dynamics_cpu(x, lambda, P, opts);
end
end

