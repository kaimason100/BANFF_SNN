% bench_dynamics.m
function result = bench_dynamics(opts)
opts = merge_options_with_seed(default_dynamics_options("cpu", "bench"), opts);
[x, P] = make_dynamics_problem(opts);
if isstruct(x) && isfield(x, 'pool')
    x = x.pool(:,1:x.steps);
end
lambda = make_lambda_sequence(size(x,2), opts);
validate_dynamics_data(x, lambda, 'bench_dynamics');

cpu_times = zeros(opts.bench_repeats,1);
for i = 1:opts.bench_repeats
    t0 = tic;
    dynamics_epoch_cpu(x, lambda, P, opts, true);
    cpu_times(i) = toc(t0);
end

gpu_times = zeros(opts.bench_repeats,1);
Pg = init_dynamics_gpu(P, opts, size(x,2));
for i = 1:opts.bench_repeats
    t0 = tic;
    dynamics_train_epoch_gpu(x, lambda, Pg, opts, 1);
    gpu_times(i) = toc(t0);
end
clear_dynamics_gpu();

result = summarize_benchmark(cpu_times, gpu_times);
end

