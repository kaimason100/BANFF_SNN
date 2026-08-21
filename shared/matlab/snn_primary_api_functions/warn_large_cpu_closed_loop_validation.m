% warn_large_cpu_closed_loop_validation.m
function warn_large_cpu_closed_loop_validation(P, opts)
persistent warned
if isempty(warned)
    warned = false;
end
if warned
    return;
end
n_hidden = double(get_opt(P, 'N_hidden', 0));
t_val = double(get_opt(opts, 'closed_loop_validation_time', get_opt(opts, 'T_sim', 0)));
n_ic = double(get_opt(opts, 'closed_loop_validation_ics', 1));
if n_hidden >= 10000 && t_val >= 50 && n_ic >= 2
    warning('snn_primary_api:largeCpuClosedLoopValidation', ...
        ['GPU dynamics training is using CPU closed-loop validation with N_hidden=%g, ', ...
         'validation time=%g s and %g initial conditions. This can dominate ARC runtime. ', ...
         'Set opts.closed_loop_validate_every=inf for ARC-scale training and run long closed-loop tests from the saved model.'], ...
        n_hidden, t_val, n_ic);
    warned = true;
end
end

