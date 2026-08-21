% closed_loop_validation_initial_conditions.m
function x0_list = closed_loop_validation_initial_conditions(sys, opts)
%CLOSED_LOOP_VALIDATION_INITIAL_CONDITIONS Build a deterministic evaluation IC set.
%   Validation includes the reference default-plus-offset condition by
%   default. Testing disables that condition and draws every IC from its
%   separate seed, keeping the complete test set held out from validation.
explicit = get_opt(opts, 'closed_loop_x0_list', []);
if ~isempty(explicit)
    x0_list = single(explicit);
    if size(x0_list,1) ~= numel(sys.x0_default)
        error('snn_primary_api:badClosedLoopICs', ...
            'closed_loop_x0_list must be D x N_IC, where D=%d.', numel(sys.x0_default));
    end
    return;
end
n_ic = max(1, round(get_opt(opts, 'closed_loop_validation_ics', 5)));
jitter = single(get_opt(opts, 'closed_loop_ic_jitter', single(0.01)));
x0_default = single(sys.x0_default(:));
D = numel(x0_default);
x0_list = repmat(x0_default, 1, n_ic);
include_reference = logical(get_opt(opts, 'closed_loop_ic_include_reference', true));
seed = double(get_opt(opts, 'closed_loop_ic_seed', 1001));
rng(seed, 'twister');
if include_reference
    x0_list(1,1) = x0_list(1,1) + jitter;
    if n_ic > 1
        x0_list(:,2:end) = x0_default + jitter .* (rand(D, n_ic-1, 'single') - single(0.5));
    end
else
    x0_list = x0_default + jitter .* (rand(D, n_ic, 'single') - single(0.5));
end
end
