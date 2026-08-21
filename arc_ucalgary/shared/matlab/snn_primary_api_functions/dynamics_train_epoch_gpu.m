% dynamics_train_epoch_gpu.m
function [loss_sum, Pg] = dynamics_train_epoch_gpu(x, lambda, Pg, opts, ep)
if isstruct(x) && isfield(x, 'pool')
    loss_sum = single(0);
    lr = single(get_lr(ep, opts.epochs, opts.SCHED));
    for bb = 1:x.train_blocks
        start_idx = randi(x.max_start_idx, 1, 'uint32');
        xb = x.pool(:, double(start_idx):double(start_idx)+x.steps-1);
        try
            [block_loss, ~] = feval('snn_time_loop_gpu_mex', 'train_primary_epoch', single(xb), logical(lambda), lr);
        catch ME
            if ~is_unknown_command(ME), rethrow(ME); end
            [block_loss, ~] = feval('snn_time_loop_gpu_mex', ...
                'train_epoch', single(xb), logical(lambda), lr, false, true, false);
        end
        loss_sum = loss_sum + single(block_loss);
    end
    Pg.B = dynamics_get_bias_gpu();
    return;
end
if iscell(x)
    loss_sum = single(0);
    lr = single(get_lr(ep, opts.epochs, opts.SCHED));
    for bb = 1:numel(x)
        try
            [block_loss, ~] = feval('snn_time_loop_gpu_mex', 'train_primary_epoch', single(x{bb}), logical(lambda{bb}), lr);
        catch ME
            if ~is_unknown_command(ME), rethrow(ME); end
            [block_loss, ~] = feval('snn_time_loop_gpu_mex', ...
                'train_epoch', single(x{bb}), logical(lambda{bb}), lr, false, true, false);
        end
        loss_sum = loss_sum + single(block_loss);
    end
    Pg.B = dynamics_get_bias_gpu();
    return;
end
lr = single(get_lr(ep, opts.epochs, opts.SCHED));
try
    [loss_sum, ~] = feval('snn_time_loop_gpu_mex', 'train_primary_epoch', single(x), logical(lambda), lr);
catch ME
    if ~is_unknown_command(ME), rethrow(ME); end
    [loss_sum, ~] = feval('snn_time_loop_gpu_mex', ...
        'train_epoch', single(x), logical(lambda), lr, false, true, false);
end
Pg.B = dynamics_get_bias_gpu();
end

