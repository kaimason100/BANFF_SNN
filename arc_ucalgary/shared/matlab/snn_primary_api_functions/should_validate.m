% should_validate.m
function tf = should_validate(ep, opts)
validate_every = max(1, round(get_opt(opts, 'validate_every', 1)));
tf = ep == 1 || ep == opts.epochs || mod(ep, validate_every) == 0;
end

