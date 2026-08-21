% static_get_bias_gpu.m
function B = static_get_bias_gpu(domain)
mex_name = static_mex_name(domain);
try
    B = feval(mex_name, 'get_bias');
catch ME
    if ~is_unknown_command(ME), rethrow(ME); end
    error('snn_primary_api:gpuMexOutOfDate', ...
        'The compiled %s MEX does not expose get_bias. Recompile the current primary-only CUDA source.', mex_name);
end
end
