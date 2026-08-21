% static_predict_gpu.m
function pred = static_predict_gpu(domain, data, split, opts)
mex_name = static_mex_name(domain);
try
    [loss_sum, metric_raw, count, Z] = feval(mex_name, 'predict_primary', split, int32(opts.batch_size));
catch ME
    if ~is_unknown_command(ME), rethrow(ME); end
    if domain == "regression"
        try
            [loss_sum, ~, metric_raw, ~, count, Z, ~] = feval(mex_name, 'predict_gpu', split, int32(opts.batch_size));
        catch ME2
            if ~is_unknown_command(ME2), rethrow(ME2); end
            [X, Y] = split_arrays(data, split);
            [loss_sum, ~, metric_raw, ~, count] = feval(mex_name, 'validate', X, Y, int32(opts.batch_size));
            Z = nan(size(Y), 'single');
            warning('snn_primary_api:legacyRegressionPredict', ...
                ['The loaded regression MEX does not expose primary prediction outputs. ', ...
                 'Metrics were computed with the legacy validate command; recompile the GPU MEX for rigorous output checks.']);
        end
    else
        [X, Y] = split_arrays(data, split);
        [loss_sum, ~, metric_raw, ~, count] = feval(mex_name, 'validate', X, Y, int32(opts.batch_size));
        Z = nan(size(Y), 'single');
        warning('snn_primary_api:legacyClassificationPredict', ...
            ['The loaded classification MEX does not expose primary prediction outputs. ', ...
             'Metrics were computed with the legacy validate command; recompile the classification gpu MEX for rigorous output checks.']);
    end
end
metric = normalize_static_metric(domain, metric_raw, count);
[~, Ysplit] = split_arrays(data, split);
pred = struct('loss', single(loss_sum / max(1,count)), 'metric', single(metric), ...
    'count', count, 'Z', single(Z), 'Y', single(Ysplit));
if domain == "regression"
    pred = attach_regression_test_stats(pred, data, split);
end
end

