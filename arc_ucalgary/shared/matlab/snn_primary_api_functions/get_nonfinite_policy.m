% get_nonfinite_policy.m
function policy = get_nonfinite_policy(opts)
%GET_NONFINITE_POLICY Return handling for NaN, +Inf and -Inf values.
%   opts.nonfinite_policy is the preferred name. opts.nan_policy is accepted
%   for backward compatibility with older scripts, but both apply to all
%   non-finite values, not just NaNs.
policy = lower(string(get_opt(opts, 'nonfinite_policy', get_opt(opts, 'nan_policy', 'error'))));
end

