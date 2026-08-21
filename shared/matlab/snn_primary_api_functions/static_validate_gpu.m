% static_validate_gpu.m
function val = static_validate_gpu(domain, data, split, opts)
mex_name = static_mex_name(domain);
try
    [loss_sum, metric_raw, count] = feval(mex_name, 'validate_primary', split, int32(opts.batch_size));
catch ME
    if ~is_unknown_command(ME), rethrow(ME); end
    [X, Y] = split_arrays(data, split);
    [loss_sum, ~, metric_raw, ~, count] = feval(mex_name, 'validate', X, Y, int32(opts.batch_size));
end
metric = normalize_static_metric(domain, metric_raw, count);
val = struct('loss', single(loss_sum / max(1,count)), 'metric', single(metric), 'count', count);
end

