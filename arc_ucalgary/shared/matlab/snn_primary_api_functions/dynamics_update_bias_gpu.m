% dynamics_update_bias_gpu.m
function dynamics_update_bias_gpu(B)
try
    feval('snn_time_loop_gpu_mex', 'update_bias', single(B));
catch ME
    if ~is_unknown_command(ME), rethrow(ME); end
    error('snn_primary_api:gpuMexOutOfDate', ...
        'The compiled dynamical-system MEX does not expose update_bias. Recompile the current primary-only CUDA source.');
end
end
