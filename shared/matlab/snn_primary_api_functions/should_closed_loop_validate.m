% should_closed_loop_validate.m
function tf = should_closed_loop_validate(ep, opts)
every = max(1, round(get_opt(opts, 'closed_loop_validate_every', inf)));
tf = isfinite(every) && (ep == 1 || ep == opts.epochs || mod(ep, every) == 0);
end

