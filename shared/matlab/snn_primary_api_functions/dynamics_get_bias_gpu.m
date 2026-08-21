% dynamics_get_bias_gpu.m
function B = dynamics_get_bias_gpu()
try
    B = feval('snn_time_loop_gpu_mex', 'get_bias');
catch ME
    if ~is_unknown_command(ME), rethrow(ME); end
    error('snn_primary_api:gpuMexOutOfDate', ...
        'The compiled dynamical-system MEX does not expose get_bias. Recompile the current primary-only CUDA source.');
end
end
