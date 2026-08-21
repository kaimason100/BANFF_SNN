% check_dynamics.m
function result = check_dynamics(opts)
opts = merge_options_with_seed(default_dynamics_options("cpu", "check"), opts);
[x, P0] = make_dynamics_problem(opts);
if isstruct(x) && isfield(x, 'pool')
    x = x.pool(:,1:x.steps);
end
lambda = make_lambda_sequence(size(x,2), opts);
validate_dynamics_data(x, lambda, 'check_dynamics');
[cpu0, ~] = dynamics_eval_cpu(x, lambda, P0, opts);
Pcpu = P0;
cpuLossTrain = single(0);
cpu_loss_history = nan(opts.epochs,1,'single');
gpu_loss_history = nan(opts.epochs,1,'single');
for ep = 1:opts.epochs
    [cpuLossTrain, cpuGrad] = dynamics_epoch_cpu(x, lambda, Pcpu, opts, true);
    Pcpu = adam_bias_update(Pcpu, cpuGrad, single(get_lr(ep, opts.epochs, opts.SCHED)), size(x,2)-1);
    cpu_loss_history(ep) = single(cpuLossTrain / max(1,size(x,2)-1));
end
[cpu1, ~] = dynamics_eval_cpu(x, lambda, Pcpu, opts);

[gpu0, Pgpu0] = dynamics_eval_gpu(x, lambda, P0, opts);
Pgpu = Pgpu0;
gpuLossTrain = single(0);
for ep = 1:opts.epochs
    [gpuLossTrain, Pgpu] = dynamics_train_epoch_gpu(x, lambda, Pgpu, opts, ep);
    gpu_loss_history(ep) = single(gpuLossTrain / max(1,size(x,2)-1));
end
[gpu1, Pgpu] = dynamics_eval_gpu(x, lambda, Pgpu, opts);

result = struct();
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
result.history = struct('cpu_loss', cpu_loss_history, 'gpu_loss', gpu_loss_history);
result.snapshots = package_check_snapshots(cpu0.Z, gpu0.Z, cpu1.Z, gpu1.Z, ...
    P0.B, Pgpu0.B, Pcpu.B, Pgpu.B);
result.neural = check_dynamics_neural_replay(x, lambda, Pcpu, Pgpu, opts);
result.cpu = struct('initial_loss', cpu0.loss, 'train_loss', cpuLossTrain, 'after_loss', cpu1.loss);
result.gpu = struct('initial_loss', gpu0.loss, 'train_loss', gpuLossTrain, 'after_loss', gpu1.loss);
result.scalar_differences = struct( ...
    'initial_loss_abs', abs(double(cpu0.loss) - double(gpu0.loss)), ...
    'after_loss_abs', abs(double(cpu1.loss) - double(gpu1.loss)), ...
    'train_loss_abs', abs(double(cpu_loss_history(end)) - double(gpu_loss_history(end))), ...
    'cpu_loss_delta', double(cpu1.loss) - double(cpu0.loss), ...
    'gpu_loss_delta', double(gpu1.loss) - double(gpu0.loss));
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

function neural = check_dynamics_neural_replay(x, lambda, Pcpu, Pgpu, opts)
%CHECK_DYNAMICS_NEURAL_REPLAY Compare final CPU/GPU-trained neural dynamics.
%   The replay uses identical input/feedback sequence and records spikes,
%   voltage, input current, recurrent current and adaptation for both final
%   models. Any difference therefore reflects CPU-vs-GPU trained parameters.
[S_cpu, U_cpu, Iin_cpu, Irec_cpu, W_cpu] = dynamics_spike_diagnostics_cpu(Pcpu, x, lambda, opts);
[S_gpu, U_gpu, Iin_gpu, Irec_gpu, W_gpu] = dynamics_spike_diagnostics_cpu(Pgpu, x, lambda, opts);
neural = package_check_neural_replay(S_cpu, S_gpu, U_cpu, U_gpu, Iin_cpu, Iin_gpu, ...
    Irec_cpu, Irec_gpu, W_cpu, W_gpu, opts.dt, 'dynamical_systems');
neural.replay_note = ['Final CPU-trained and GPU-trained models are replayed ', ...
    'with the CPU diagnostic recorder on the same trajectory.'];
end
