% dynamics_set_optimizer_state_gpu.m
function dynamics_set_optimizer_state_gpu(state)
try
    feval('snn_time_loop_gpu_mex', 'set_optim_state', single(state.m_b), single(state.v_b), single(state.vhat_b), double(state.t_adam));
catch ME
    if ~is_unknown_command(ME), rethrow(ME); end
    error('snn_primary_api:gpuCheckpointUnsupported', ...
        'The compiled dynamical-system GPU MEX does not expose set_optim_state. Recompile the current CUDA source before resuming an ARC checkpoint.');
end
end

