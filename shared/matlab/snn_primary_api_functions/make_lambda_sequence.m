% make_lambda_sequence.m
function lambda = make_lambda_sequence(steps, opts)
lambda = true(1, steps);
if ~isfield(opts, 'use_multistep') || ~opts.use_multistep, return; end
period = max(1, round(opts.W_warmup) + round(opts.H_free));
idx = 2:steps;
teacher = mod(idx-1, period) < round(opts.W_warmup);
lambda(idx(~teacher)) = false;
end

