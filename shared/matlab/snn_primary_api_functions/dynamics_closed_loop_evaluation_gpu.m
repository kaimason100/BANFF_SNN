% dynamics_closed_loop_evaluation_gpu.m
function closed = dynamics_closed_loop_evaluation_gpu(P, opts, eval_set)
if nargin < 3 || isempty(eval_set)
    eval_set = make_closed_loop_eval_set(opts);
end
n_ic = numel(eval_set.x_true);
Pg = init_dynamics_gpu(P, opts, max(cellfun(@(x) size(x,2), eval_set.x_true)));
cleanup = onCleanup(@() clear_dynamics_gpu()); %#ok<NASGU>
dynamics_update_bias_gpu(P.B);
wd_by_ic = nan(n_ic, 1, 'single');
pred_by_ic = cell(n_ic, 1);
truth_by_ic = cell(n_ic, 1);
test_x0_norm_by_ic = cell(n_ic, 1);
truth_diagnostic_by_ic = cell(n_ic, 1);
for ic = 1:n_ic
    [summary, Pg] = dynamics_eval_gpu_current(eval_set.x_true{ic}, eval_set.lambda{ic}, Pg, eval_set.opts);
    if isfield(summary, 'num_valid_prediction_columns')
        pred = single(summary.Z(:,1:summary.num_valid_prediction_columns).');
    else
        pred = valid_dynamics_predictions(summary.Z).';
    end
    [pred, truth, test_x0_norm, truth_diagnostic] = closed_loop_test_segment(pred, eval_set, ic);
    wd_by_ic(ic) = single(phase_portrait_wasserstein_distance(double(pred), double(truth), opts.wd));
    pred_by_ic{ic} = pred;
    truth_by_ic{ic} = truth;
    test_x0_norm_by_ic{ic} = test_x0_norm;
    truth_diagnostic_by_ic{ic} = truth_diagnostic;
end
closed = struct();
closed.wasserstein_distance = single(mean(wd_by_ic, 'omitnan'));
closed.wasserstein_distance_by_ic = wd_by_ic;
closed.x0_list = eval_set.x0_list;
closed.test_x0_norm_by_ic = test_x0_norm_by_ic;
closed.truth_diagnostic_by_ic = truth_diagnostic_by_ic;
closed.truth_simulation_failed_by_ic = cellfun(@truth_diagnostic_failed, truth_diagnostic_by_ic);
closed.closed_loop_ic_seed = get_eval_set_field(eval_set, 'closed_loop_ic_seed', []);
closed.closed_loop_test_ic_seed = closed.closed_loop_ic_seed;
closed.closed_loop_ic_jitter = get_eval_set_field(eval_set, 'closed_loop_ic_jitter', []);
closed.closed_loop_ic_include_reference = logical(get_eval_set_field(eval_set, 'closed_loop_ic_include_reference', true));
closed.closed_loop_ic_role = char(get_eval_set_field(eval_set, 'closed_loop_ic_role', 'unspecified'));
closed.closed_loop_warmup_time = single(get_eval_set_field(eval_set, 'warmup_time', 0));
closed.closed_loop_test_time = single(get_eval_set_field(eval_set, 'test_time', NaN));
closed.closed_loop_warmup_steps = int32(get_eval_set_field(eval_set, 'warmup_steps', 0));
closed.closed_loop_truth_initialization = 'network_terminal_warmup_state';
closed.pred_norm = pred_by_ic{1};
closed.true_norm = truth_by_ic{1};
closed.pred_norm_by_ic = pred_by_ic;
closed.true_norm_by_ic = truth_by_ic;
closed.n_initial_conditions = n_ic;
closed.gpu_bias = Pg.B;
end

function [pred_trim, truth_trim, test_x0_norm, truth_diagnostic] = closed_loop_test_segment(pred, eval_set, ic)
warmup_steps = max(0, round(double(get_eval_set_field(eval_set, 'warmup_steps', 0))));
if warmup_steps >= size(pred, 1)
    error('snn_primary_api:closedLoopWarmupTooLong', ...
        'Closed-loop warmup requires one of %d prediction steps as its terminal state. Reduce closed_loop_warmup_time or increase test duration.', size(pred, 1));
end

% Warm up only the autonomous network. The true system then starts from the
% network's decoded terminal warmup state, so both scored trajectories share
% the same initial condition at the start of the scored interval.
if warmup_steps > 0
    pred_trim = pred(warmup_steps + 1:end, :);
    test_x0_norm = single(pred(warmup_steps, :).');
else
    pred_trim = pred;
    initial_states = get_eval_set_field(eval_set, 'test_x0_norm', {});
    if iscell(initial_states) && numel(initial_states) >= ic && ~isempty(initial_states{ic})
        test_x0_norm = single(initial_states{ic});
    else
        test_x0_norm = single([]);
    end
end

[truth_trim, truth_diagnostic] = closed_loop_truth_from_network_state(test_x0_norm, eval_set);
if isempty(truth_trim)
    % Preserve the prediction length for diagnostics; the non-finite truth
    % continuation is represented by an infinite Wasserstein distance.
    truth_trim = nan(size(pred_trim), 'single');
    return;
end
n = min(size(pred_trim, 1), size(truth_trim, 1));
if n < 1
    error('snn_primary_api:emptyClosedLoopTestSegment', ...
        'Closed-loop test segment is empty after warmup trimming.');
end
pred_trim = pred_trim(1:n, :);
truth_trim = truth_trim(1:n, :);
end

function value = get_eval_set_field(eval_set, name, default_value)
if isstruct(eval_set) && isfield(eval_set, name) && ~isempty(eval_set.(name))
    value = eval_set.(name);
else
    value = default_value;
end
end

function tf = truth_diagnostic_failed(diagnostic)
tf = isstruct(diagnostic) && isfield(diagnostic, 'status') && ...
    ~(strcmp(diagnostic.status, 'ok') || strcmp(diagnostic.status, 'not_required'));
end
