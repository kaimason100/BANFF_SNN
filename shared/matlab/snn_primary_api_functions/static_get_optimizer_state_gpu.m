% static_get_optimizer_state_gpu.m
function state = static_get_optimizer_state_gpu(domain)
mex_name = static_mex_name(domain);
try
    [state.m_b, state.v_b, state.vhat_b, state.t_adam] = feval(mex_name, 'get_optim_state');
catch ME
    if ~is_unknown_command(ME), rethrow(ME); end
    error('snn_primary_api:gpuCheckpointUnsupported', ...
        'The compiled %s MEX does not expose get_optim_state. Recompile the current CUDA source before using ARC checkpoint resume.', mex_name);
end
state.m_b = single(state.m_b);
state.v_b = single(state.v_b);
state.vhat_b = single(state.vhat_b);
state.t_adam = double(state.t_adam);
end

