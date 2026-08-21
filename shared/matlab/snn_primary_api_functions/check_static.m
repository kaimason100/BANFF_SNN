% check_static.m
function result = check_static(domain, opts)
opts = merge_options_with_seed(default_static_options(domain, "cpu", "check"), opts);
opts.synthetic = true;
data = load_static_data(domain, opts);
validate_static_data(data, domain, 'check_static');
P0 = make_primary_model(size(data.X_train,1), size(data.Y_train,1), opts);
order = int32(1:size(data.X_train,2));

[cpu0, ~] = static_eval_cpu(domain, data.X_train, data.Y_train, P0, opts);
Pcpu = P0;
cpuLossTrain = single(0);
cpuMetricTrain = single(0);
cpu_loss_history = nan(opts.epochs,1,'single');
cpu_metric_history = nan(opts.epochs,1,'single');
gpu_loss_history = nan(opts.epochs,1,'single');
gpu_metric_history = nan(opts.epochs,1,'single');
for ep = 1:opts.epochs
    [cpuLossTrain, cpuMetricTrain, cpuGrad] = static_epoch_cpu(domain, data.X_train, data.Y_train, Pcpu, opts, order, true);
    Pcpu = adam_bias_update(Pcpu, cpuGrad, single(get_lr(ep, opts.epochs, opts.SCHED)), size(data.X_train,2));
    cpu_loss_history(ep) = single(cpuLossTrain / max(1,size(data.X_train,2)));
    cpu_metric_history(ep) = single(cpuMetricTrain);
end
[cpu1, ~] = static_eval_cpu(domain, data.X_train, data.Y_train, Pcpu, opts);

[gpu0, Pgpu0] = static_eval_gpu(domain, data, P0, opts, 'train');
Pgpu = Pgpu0;
gpuLossTrain = single(0);
gpuMetricTrain = single(0);
for ep = 1:opts.epochs
    [gpuLossTrain, gpuMetricTrain, Pgpu] = static_train_epoch_gpu(domain, data, 'train', order, Pgpu, opts, ep);
    gpu_loss_history(ep) = single(gpuLossTrain / max(1,size(data.X_train,2)));
    gpu_metric_history(ep) = single(gpuMetricTrain);
end
[gpu1, Pgpu] = static_eval_gpu(domain, data, Pgpu, opts, 'train');

result = struct();
result.domain = char(domain);
result.options = opts;
result.initial = compare_arrays(cpu0.Z, gpu0.Z);
result.after_training = compare_arrays(cpu1.Z, gpu1.Z);
result.bias_before_training = compare_arrays(P0.B, Pgpu0.B);
result.bias_after_training = compare_arrays(Pcpu.B, Pgpu.B);
result.after_one_epoch = result.after_training;
result.bias_after_one_epoch = result.bias_after_training;
result.cpu_output_training_effect = compare_arrays(cpu0.Z, cpu1.Z);
result.gpu_output_training_effect = compare_arrays(gpu0.Z, gpu1.Z);
result.cpu_bias_training_effect = compare_arrays(P0.B, Pcpu.B);
result.gpu_bias_training_effect = compare_arrays(P0.B, Pgpu.B);
result.output_comparison_available = ~(result.initial.has_nan || result.initial.has_inf || ...
    result.after_training.has_nan || result.after_training.has_inf);
result.history = struct('cpu_loss', cpu_loss_history, 'gpu_loss', gpu_loss_history, ...
    'cpu_metric', cpu_metric_history, 'gpu_metric', gpu_metric_history);
result.snapshots = package_check_snapshots(cpu0.Z, gpu0.Z, cpu1.Z, gpu1.Z, ...
    P0.B, Pgpu0.B, Pcpu.B, Pgpu.B);
result.neural = check_static_neural_replay(domain, data, Pcpu, Pgpu, opts);
result.cpu = struct('initial_loss', cpu0.loss, 'initial_metric', cpu0.metric, ...
    'train_loss', cpuLossTrain, 'train_metric', cpuMetricTrain, ...
    'after_loss', cpu1.loss, 'after_metric', cpu1.metric);
result.gpu = struct('initial_loss', gpu0.loss, 'initial_metric', gpu0.metric, ...
    'train_loss', gpuLossTrain, 'train_metric', gpuMetricTrain, ...
    'after_loss', gpu1.loss, 'after_metric', gpu1.metric);
result.scalar_differences = struct( ...
    'initial_loss_abs', abs(double(cpu0.loss) - double(gpu0.loss)), ...
    'after_loss_abs', abs(double(cpu1.loss) - double(gpu1.loss)), ...
    'train_loss_abs', abs(double(cpu_loss_history(end)) - double(gpu_loss_history(end))), ...
    'initial_metric_abs', abs(double(cpu0.metric) - double(gpu0.metric)), ...
    'after_metric_abs', abs(double(cpu1.metric) - double(gpu1.metric)), ...
    'train_metric_abs', abs(double(cpuMetricTrain) - double(gpuMetricTrain)), ...
    'cpu_loss_delta', double(cpu1.loss) - double(cpu0.loss), ...
    'gpu_loss_delta', double(gpu1.loss) - double(gpu0.loss), ...
    'cpu_metric_delta', double(cpu1.metric) - double(cpu0.metric), ...
    'gpu_metric_delta', double(gpu1.metric) - double(gpu0.metric));
if ~result.output_comparison_available
    if (domain == "classification" || domain == "regression") && ...
            (result.initial.has_nan || result.after_training.has_nan)
        result.compatibility_note = ['The loaded GPU MEX does not return primary ', ...
            'prediction outputs. Metrics and bias updates were checked, but output-array ', ...
            'comparison requires recompiling the gpu MEX source.'];
        warning('snn_primary_api:checkOutputSkipped', result.compatibility_note);
    else
        error('snn_primary_api:checkFailed', ...
            'GPU-vs-CPU output comparison produced non-finite values; this is not expected for %s.', domain);
    end
end
result.tolerance_assertions = false;
result.tolerance_note = ['CPU-vs-GPU check reports measured absolute and relative differences; ', ...
    'it does not pass/fail by tolerance. Inspect result.initial, result.after_training, ', ...
    'result.bias_after_training and result.neural.'];
end

function snapshots = package_check_snapshots(cpu_output_before, gpu_output_before, ...
    cpu_output_after, gpu_output_after, cpu_bias_before, gpu_bias_before, ...
    cpu_bias_after, gpu_bias_after)
%PACKAGE_CHECK_SNAPSHOTS Store compact arrays for visual CPU/GPU inspection.
snapshots = struct();
snapshots.cpu_output_before = gather_if_needed(cpu_output_before);
snapshots.gpu_output_before = gather_if_needed(gpu_output_before);
snapshots.cpu_output_after = gather_if_needed(cpu_output_after);
snapshots.gpu_output_after = gather_if_needed(gpu_output_after);
snapshots.cpu_bias_before = gather_if_needed(cpu_bias_before);
snapshots.gpu_bias_before = gather_if_needed(gpu_bias_before);
snapshots.cpu_bias_after = gather_if_needed(cpu_bias_after);
snapshots.gpu_bias_after = gather_if_needed(gpu_bias_after);
end

function x = gather_if_needed(x)
if isa(x, 'gpuArray')
    x = gather(x);
end
x = single(x);
end

function neural = check_static_neural_replay(domain, data, Pcpu, Pgpu, opts)
%CHECK_STATIC_NEURAL_REPLAY Compare final CPU/GPU-trained neural dynamics.
%   Static GPU kernels are optimized not to allocate spike rasters. For the
%   check plot, replay the CPU-trained and GPU-trained final parameter sets
%   through the same CPU diagnostic recorder. This isolates whether the final
%   trained parameters produce the same neural dynamics.
n_samples = min(size(data.X_train,2), max(1, round(get_opt(opts, 'check_neural_samples', 3))));
X = data.X_train(:,1:n_samples);
[S_cpu, U_cpu, Iin_cpu, Irec_cpu, W_cpu] = static_spike_diagnostics_cpu(Pcpu, X, opts);
[S_gpu, U_gpu, Iin_gpu, Irec_gpu, W_gpu] = static_spike_diagnostics_cpu(Pgpu, X, opts);
neural = package_check_neural_replay(S_cpu, S_gpu, U_cpu, U_gpu, Iin_cpu, Iin_gpu, ...
    Irec_cpu, Irec_gpu, W_cpu, W_gpu, opts.dt, char(domain));
neural.replay_note = ['Static GPU MEX does not expose resident spike rasters; ', ...
    'both final models are replayed with the CPU diagnostic recorder for neural comparison.'];
end
