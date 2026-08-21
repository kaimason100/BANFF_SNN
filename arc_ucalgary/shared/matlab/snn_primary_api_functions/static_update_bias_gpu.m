% static_update_bias_gpu.m
function static_update_bias_gpu(domain, B, W_out) %#ok<INUSD>
mex_name = static_mex_name(domain);
try
    feval(mex_name, 'update_bias', single(B));
catch ME
    if ~is_unknown_command(ME), rethrow(ME); end
    error('snn_primary_api:gpuMexOutOfDate', ...
        'The compiled %s MEX does not expose update_bias. Recompile the current primary-only CUDA source.', mex_name);
end
end
