% static_set_optimizer_state_gpu.m
function static_set_optimizer_state_gpu(domain, state)
mex_name = static_mex_name(domain);
try
    feval(mex_name, 'set_optim_state', single(state.m_b), single(state.v_b), single(state.vhat_b), double(state.t_adam));
catch ME
    if ~is_unknown_command(ME), rethrow(ME); end
    error('snn_primary_api:gpuCheckpointUnsupported', ...
        'The compiled %s MEX does not expose set_optim_state. Recompile the current CUDA source before resuming an ARC checkpoint.', mex_name);
end
end

