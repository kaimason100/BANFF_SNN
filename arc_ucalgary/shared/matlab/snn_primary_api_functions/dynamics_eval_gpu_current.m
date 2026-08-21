% dynamics_eval_gpu_current.m
function [summary, Pg] = dynamics_eval_gpu_current(x, lambda, Pg, opts)
if logical(get_opt(opts, 'gpu_diagnostics', false))
    try
        [Z, ~, ~, ~, ~, ~, ~, loss_sum, count] = feval('snn_time_loop_gpu_mex', ...
            'run_primary_diagnostic', single(x), logical(lambda), int32(0), int32(zeros(0,1)));
    catch ME
        if ~is_unknown_command(ME), rethrow(ME); end
        [Z, ~, ~, ~, ~, ~, ~, ~, loss_sum, count] = feval('snn_time_loop_gpu_mex', ...
            'run_diagnostic', single(x), logical(lambda), false, int32(0), int32(zeros(0,1)));
    end
    Z = valid_dynamics_predictions(single(Z), size(x,2));
else
    try
        [Z, loss_sum, count] = feval('snn_time_loop_gpu_mex', ...
            'run_primary_eval', single(x), logical(lambda));
        Z = valid_dynamics_predictions(single(Z), size(x,2));
    catch ME
        if ~is_unknown_command(ME), rethrow(ME); end
        if ~logical(get_opt(opts, 'allow_legacy_diagnostic_gpu_eval', false))
            error('snn_primary_api:staleDynamicsMex', ...
                ['The compiled dynamics MEX lacks run_primary_eval. Normal GPU dynamics evaluation ', ...
                 'will not fall back to diagnostic evaluation because that allocates large N_hidden x T buffers. ', ...
                 'Recompile the current dynamics MEX, or set opts.allow_legacy_diagnostic_gpu_eval=true only for small legacy checks.']);
        end
        warning('snn_primary_api:legacyDynamicsGpuEval', ...
            'Explicit allow_legacy_diagnostic_gpu_eval=true: using memory-heavy diagnostic evaluation. Recompile for run_primary_eval support.');
        [Z, ~, ~, ~, ~, ~, ~, loss_sum, count] = feval('snn_time_loop_gpu_mex', ...
            'run_primary_diagnostic', single(x), logical(lambda), int32(0), int32(zeros(0,1)));
        Z = valid_dynamics_predictions(single(Z), size(x,2));
    end
end
summary = struct('loss', single(loss_sum / max(1,double(count))), ...
    'Z', single(Z), 'num_valid_prediction_columns', size(Z,2));
Pg.B = dynamics_get_bias_gpu();
end

