% dynamics_get_optimizer_state_gpu.m
function state = dynamics_get_optimizer_state_gpu()
try
    [state.m_b, state.v_b, state.vhat_b, state.t_adam] = feval('snn_time_loop_gpu_mex', 'get_optim_state');
catch ME
    if ~is_unknown_command(ME), rethrow(ME); end
    error('snn_primary_api:gpuCheckpointUnsupported', ...
        'The compiled dynamical-system GPU MEX does not expose get_optim_state. Recompile the current CUDA source before using ARC checkpoint resume.');
end
state.m_b = single(state.m_b);
state.v_b = single(state.v_b);
state.vhat_b = single(state.vhat_b);
state.t_adam = double(state.t_adam);
end

