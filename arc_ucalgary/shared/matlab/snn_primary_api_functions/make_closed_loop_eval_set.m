% make_closed_loop_eval_set.m
function eval_set = make_closed_loop_eval_set(opts)
%MAKE_CLOSED_LOOP_EVAL_SET Build deterministic network warmup inputs for closed-loop tests.
%   The same set can be reused across validation epochs so validation does not
%   change because of fresh random initial-condition jitter. The reference
%   trajectories provide the initial teacher-forced state and the autonomous
%   network warmup carrier. Evaluation starts the true system from the
%   network's terminal warmup state for the scored test interval.
previous_rng = rng;
cleanup_rng = onCleanup(@() rng(previous_rng)); %#ok<NASGU>
opts_base = opts;
test_time = single(opts.T_sim);
warmup_time = single(max(0, double(get_opt(opts_base, 'closed_loop_warmup_time', 0))));
opts_base.closed_loop_test_time_effective = test_time;
opts_base.closed_loop_warmup_time = warmup_time;
opts_base.T_sim = single(double(test_time) + double(warmup_time));
opts_base.dynamics_split = 'closed_loop';
sys = make_dynamics_system_for_api(opts_base.system_name);
has_saved_norm = isfield(opts_base, 'dynamics_mu') && isfield(opts_base, 'dynamics_sigma') && ...
    ~isempty(opts_base.dynamics_mu) && ~isempty(opts_base.dynamics_sigma);
recompute_norm = logical(get_opt(opts_base, 'recompute_dynamics_norm', false));
if ~has_saved_norm || recompute_norm
    % If training normalization is not already attached to opts, rebuild it
    % from the training pool. Saved model testing normally provides these.
    opts_norm = opts_base;
    opts_norm.dynamics_split = 'train';
    [x_pool, ~] = make_dynamics_problem(opts_norm);
    if has_saved_norm && recompute_norm
        warn_if_dynamics_norm_differs(opts_base.dynamics_mu, opts_base.dynamics_sigma, x_pool.mu, x_pool.sigma);
    end
    opts_base.dynamics_mu = x_pool.mu;
    opts_base.dynamics_sigma = x_pool.sigma;
end
% Deterministic list of validation/test initial conditions.
x0_list = closed_loop_validation_initial_conditions(sys, opts_base);
n_ic = size(x0_list, 2);
eval_set = struct();
eval_set.opts = opts_base;
eval_set.x0_list = x0_list;
eval_set.closed_loop_ic_seed = get_opt(opts_base, 'closed_loop_ic_seed', 1001);
eval_set.closed_loop_ic_jitter = get_opt(opts_base, 'closed_loop_ic_jitter', single(0.01));
eval_set.closed_loop_ic_include_reference = logical(get_opt(opts_base, 'closed_loop_ic_include_reference', true));
eval_set.closed_loop_ic_role = char(get_opt(opts_base, 'closed_loop_ic_role', 'validation'));
eval_set.warmup_time = warmup_time;
eval_set.test_time = test_time;
eval_set.warmup_steps = max(0, round(double(warmup_time) / double(opts_base.dt)));
eval_set.x_true = cell(n_ic, 1);
eval_set.lambda = cell(n_ic, 1);
eval_set.truth = cell(n_ic, 1);
eval_set.test_x0_norm = cell(n_ic, 1);
for ic = 1:n_ic
    opts_eval = opts_base;
    opts_eval.x0_override = x0_list(:,ic);
    % Reference normalized trajectory: lambda uses only its initial state,
    % then the network runs autonomously throughout warmup and testing.
    [x_true, ~] = make_dynamics_problem(opts_eval);
    % Closed-loop lambda: only the first state is teacher-forced; the rest of
    % the rollout feeds back network predictions.
    lambda_closed = make_closed_loop_lambda(size(x_true,2));
    validate_dynamics_data(x_true, lambda_closed, sprintf('closed_loop_ic_%d', ic));
    eval_set.x_true{ic} = x_true;
    eval_set.lambda{ic} = lambda_closed;
    % Retain the reference trajectory for backwards-compatible diagnostics.
    % It is not the scored true-system trajectory when warmup is nonzero.
    eval_set.truth{ic} = single(x_true(:,2:end).');
    eval_set.test_x0_norm{ic} = single(x_true(:, 1));
end
end
