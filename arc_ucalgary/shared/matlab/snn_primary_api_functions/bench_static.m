% bench_static.m
function result = bench_static(domain, opts)
opts = merge_options_with_seed(default_static_options(domain, "cpu", "bench"), opts);
opts.synthetic = true;
data = load_static_data(domain, opts);
validate_static_data(data, domain, 'bench_static');
P = make_primary_model(size(data.X_train,1), size(data.Y_train,1), opts);
order = int32(1:size(data.X_train,2));

cpu_times = zeros(opts.bench_repeats,1);
for i = 1:opts.bench_repeats
    Pc = P;
    t0 = tic;
    g = static_epoch_cpu(domain, data.X_train, data.Y_train, Pc, opts, order, true);
    if isstruct(g), error('Unexpected gradient output.'); end
    cpu_times(i) = toc(t0);
end

gpu_times = zeros(opts.bench_repeats,1);
Pg = init_static_gpu(domain, data, P, opts);
for i = 1:opts.bench_repeats
    t0 = tic;
    static_train_epoch_gpu(domain, data, 'train', order, Pg, opts, 1);
    gpu_times(i) = toc(t0);
end
clear_static_gpu(domain);

result = summarize_benchmark(cpu_times, gpu_times);
end

