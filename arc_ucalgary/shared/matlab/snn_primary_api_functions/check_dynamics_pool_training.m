% check_dynamics_pool_training.m
function result = check_dynamics_pool_training(opts)
%CHECK_DYNAMICS_POOL_TRAINING Exercise the random contiguous-block path.
%   This is the path used by ARC dynamics training when x is a long trajectory
%   pool plus train_blocks, rather than one fixed matrix trajectory.
opts = merge_options_with_seed(default_dynamics_options("gpu", "check"), opts);
opts.N_hidden = min(double(get_opt(opts, 'N_hidden', 32)), 64);
opts.N_rec = min(double(get_opt(opts, 'N_rec', 4)), opts.N_hidden);
opts.T_sim = single(get_opt(opts, 'T_sim', 0.030));
opts.long_sim_time = single(get_opt(opts, 'long_sim_time', 0.120));
opts.burn_in_time = single(0);
opts.train_blocks = max(2, round(get_opt(opts, 'train_blocks', 3)));
opts.dynamics_split = 'train';
if exist('snn_time_loop_gpu_mex', 'file') ~= 3
    result = struct('status', 'skipped', 'reason', 'snn_time_loop_gpu_mex is not compiled or not on the MATLAB path.');
    return;
end
[x, P0] = make_dynamics_problem(opts);
lambda = make_lambda_sequence_for_data(x, opts);
validate_dynamics_data(x, lambda, 'check_dynamics_pool_training');
rng(1234, 'twister');
Pcpu = P0;
cpu_loss = single(0);
lr = single(get_lr(1, opts.epochs, opts.SCHED));
block_steps = max(1, x.steps - 1);
for bb = 1:x.train_blocks
    start_idx = randi(x.max_start_idx, 1, 'uint32');
    xb = x.pool(:, double(start_idx):double(start_idx)+x.steps-1);
    [block_loss, gB] = dynamics_epoch_cpu(xb, lambda, Pcpu, opts, true);
    Pcpu = adam_bias_update(Pcpu, gB, lr, block_steps);
    cpu_loss = cpu_loss + block_loss;
end
rng(1234, 'twister');
Pg = init_dynamics_gpu(P0, opts, max_sequence_steps(x));
[gpu_loss, Pg] = dynamics_train_epoch_gpu(x, lambda, Pg, opts, 1);
cleanup = onCleanup(@() clear_dynamics_gpu()); %#ok<NASGU>
result = struct();
result.status = 'checked';
result.loss = compare_arrays(single(cpu_loss), single(gpu_loss));
result.bias_after_one_epoch = compare_arrays(Pcpu.B, Pg.B);
result.tolerance_assertions = false;
result.tolerance_note = ['Pool/block CPU-vs-GPU check reports measured bias ', ...
    'differences and does not pass/fail by tolerance.'];
end
