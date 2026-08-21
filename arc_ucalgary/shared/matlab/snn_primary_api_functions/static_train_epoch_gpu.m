% static_train_epoch_gpu.m
function [loss_sum, metric, Pg] = static_train_epoch_gpu(domain, data, split, order, Pg, opts, ep)
lr = single(get_lr(ep, opts.epochs, opts.SCHED));
mex_name = static_mex_name(domain);
try
    [loss_sum, metric_raw, count] = feval(mex_name, 'train_primary_epoch', split, order, int32(opts.batch_size), lr);
catch ME
    if ~is_unknown_command(ME), rethrow(ME); end
    [X, Y] = split_arrays(data, split);
    [loss_sum, ~, metric_raw, ~, count, gB, ~] = feval(mex_name, 'train_epoch', X, Y, order, int32(opts.batch_size));
    Pg = adam_bias_update(Pg, gB, lr, count);
    static_update_bias_gpu(domain, Pg.B, Pg.W_out);
end
metric = normalize_static_metric(domain, metric_raw, count);
Pg.B = static_get_bias_gpu(domain);
end

