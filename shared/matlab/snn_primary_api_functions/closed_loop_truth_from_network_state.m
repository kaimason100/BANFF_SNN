% Package orientation: Shared implementation helper for closed-loop dynamics diagnostics.

function [truth, diagnostic] = closed_loop_truth_from_network_state(test_x0_norm, eval_set)
%CLOSED_LOOP_TRUTH_FROM_NETWORK_STATE Continue the true system from a network state.
%   This is used after closed-loop warmup. If the network state maps outside
%   the true system's numerically stable region, the diagnostic records where
%   the true trajectory becomes non-finite instead of hiding the source.

opts_truth = eval_set.opts;
opts_truth.T_sim = single(get_eval_set_field_local(eval_set, 'test_time', opts_truth.T_sim));
opts_truth.closed_loop_warmup_time = single(0);
mu = single(opts_truth.dynamics_mu(:));
sigma = single(opts_truth.dynamics_sigma(:));
test_x0_norm = single(test_x0_norm(:));
if numel(test_x0_norm) ~= numel(mu) || numel(sigma) ~= numel(mu)
    raw_x0 = nan(size(mu), 'single');
    diagnostic = base_diagnostic(opts_truth, test_x0_norm, raw_x0, eval_set);
    diagnostic.status = 'network_state_dimension_mismatch';
    diagnostic.message = sprintf(['Network post-warmup state has %d elements, but the true-system ', ...
        'normalization expects %d. This usually indicates incompatible closed-loop evaluation code.'], ...
        numel(test_x0_norm), numel(mu));
    truth = [];
    return;
end
raw_x0 = single(test_x0_norm(:)) .* sigma + mu;
opts_truth.x0_override = raw_x0;

diagnostic = base_diagnostic(opts_truth, test_x0_norm, raw_x0, eval_set);
truth = [];

if any(~isfinite(test_x0_norm(:)))
    diagnostic.status = 'nonfinite_network_state';
    diagnostic.message = 'Network post-warmup normalized state contains non-finite values.';
    return;
end
if any(~isfinite(raw_x0(:)))
    diagnostic.status = 'nonfinite_raw_initial_condition';
    diagnostic.message = 'Network post-warmup state maps to a non-finite raw true-system initial condition.';
    return;
end

try
    sys = make_dynamics_system_for_api(opts_truth.system_name);
    ensure_dynamics_utility_on_path('simulate_dynamical_system');
    integrator = char(get_opt(opts_truth, 'integrator', 'euler'));
    raw = simulate_dynamical_system(sys.f, [0 opts_truth.T_sim], raw_x0, ...
        opts_truth.dt, sys.params, integrator, opts_truth.dyn_sys_rate);
catch ME
    diagnostic.status = 'true_integrator_error';
    diagnostic.message = ME.message;
    diagnostic.error_identifier = ME.identifier;
    return;
end

diagnostic.raw_trajectory_size = size(raw);
finite_raw = raw(isfinite(raw));
if isempty(finite_raw)
    diagnostic.raw_trajectory_finite_min = NaN;
    diagnostic.raw_trajectory_finite_max = NaN;
    diagnostic.raw_trajectory_finite_max_abs = Inf;
else
    diagnostic.raw_trajectory_finite_min = double(min(finite_raw(:)));
    diagnostic.raw_trajectory_finite_max = double(max(finite_raw(:)));
    diagnostic.raw_trajectory_finite_max_abs = double(max(abs(finite_raw(:))));
end

bad_by_col = any(~isfinite(raw), 1);
if any(bad_by_col)
    first_col = find(bad_by_col, 1, 'first');
    bad_dims = find(~isfinite(raw(:, first_col)));
    diagnostic.status = 'nonfinite_true_trajectory';
    diagnostic.message = 'True-system continuation from the network post-warmup state became non-finite.';
    diagnostic.first_nonfinite_column = first_col;
    diagnostic.first_nonfinite_time = double(first_col - 1) * double(opts_truth.dt);
    diagnostic.first_nonfinite_dimensions = bad_dims(:).';
    diagnostic.finite_prefix_columns = max(0, first_col - 1);
    return;
end

x_norm = single((raw - mu) ./ sigma);
bad_norm_by_col = any(~isfinite(x_norm), 1);
if any(bad_norm_by_col)
    first_col = find(bad_norm_by_col, 1, 'first');
    bad_dims = find(~isfinite(x_norm(:, first_col)));
    diagnostic.status = 'nonfinite_normalized_truth_trajectory';
    diagnostic.message = 'True-system continuation was finite in raw coordinates but became non-finite after normalization.';
    diagnostic.first_nonfinite_column = first_col;
    diagnostic.first_nonfinite_time = double(first_col - 1) * double(opts_truth.dt);
    diagnostic.first_nonfinite_dimensions = bad_dims(:).';
    diagnostic.finite_prefix_columns = max(0, first_col - 1);
    return;
end
truth = single(x_norm(:, 2:end).');
diagnostic.status = 'ok';
diagnostic.message = 'True-system continuation remained finite.';
diagnostic.truth_size = size(truth);
end

function diagnostic = base_diagnostic(opts_truth, test_x0_norm, raw_x0, eval_set)
diagnostic = struct();
diagnostic.status = 'not_run';
diagnostic.message = '';
diagnostic.error_identifier = '';
diagnostic.system_name = char(opts_truth.system_name);
diagnostic.integrator = char(get_opt(opts_truth, 'integrator', 'euler'));
diagnostic.dt = double(opts_truth.dt);
diagnostic.dyn_sys_rate = double(opts_truth.dyn_sys_rate);
diagnostic.test_time = double(opts_truth.T_sim);
diagnostic.closed_loop_ic_seed = get_eval_set_field_local(eval_set, 'closed_loop_ic_seed', []);
diagnostic.closed_loop_ic_jitter = get_eval_set_field_local(eval_set, 'closed_loop_ic_jitter', []);
diagnostic.network_x0_norm = single(test_x0_norm(:));
diagnostic.raw_x0 = single(raw_x0(:));
diagnostic.network_x0_norm_max_abs = max_abs_finite_or_inf(test_x0_norm);
diagnostic.raw_x0_max_abs = max_abs_finite_or_inf(raw_x0);
diagnostic.raw_trajectory_size = [0 0];
diagnostic.raw_trajectory_finite_min = NaN;
diagnostic.raw_trajectory_finite_max = NaN;
diagnostic.raw_trajectory_finite_max_abs = NaN;
diagnostic.first_nonfinite_column = [];
diagnostic.first_nonfinite_time = [];
diagnostic.first_nonfinite_dimensions = [];
diagnostic.finite_prefix_columns = [];
diagnostic.truth_size = [0 0];
end

function value = max_abs_finite_or_inf(x)
x = double(x(:));
if any(~isfinite(x))
    value = Inf;
else
    value = max(abs(x));
end
end

function value = get_eval_set_field_local(eval_set, name, default_value)
if isstruct(eval_set) && isfield(eval_set, name) && ~isempty(eval_set.(name))
    value = eval_set.(name);
else
    value = default_value;
end
end
