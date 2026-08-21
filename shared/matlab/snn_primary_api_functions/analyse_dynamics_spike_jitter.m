% Package orientation: Optional Gaussian-jitter diagnostic retained separately
% from the publication rate-preserving within-window timing-shuffle analysis.

function result = analyse_dynamics_spike_jitter(opts)
%ANALYSE_DYNAMICS_SPIKE_JITTER Simulate post-hoc spike-time jitter for saved models.
%   This diagnostic is not used by analyse_jitter.mlx or the publication figure.
%   The recurrent network is run once without jitter. Recorded spike events
%   are then independently time-jittered and passed through the fixed decoder
%   filter.  This quantifies decoder sensitivity; it is not an online noisy
%   recurrent simulation, because jittered readout states are not fed back.

if ~isfield(opts, 'model_files') || isempty(opts.model_files)
    if ~isfield(opts, 'model_file') || isempty(opts.model_file)
        error('snn_primary_api:modelFile', ...
            'analyse_dynamics_spike_jitter requires opts.model_file or opts.model_files.');
    end
    model_files = {char(opts.model_file)};
else
    model_files = cellstr(opts.model_files);
end
seed_list = double(get_opt(opts, 'seed_list', 1:numel(model_files)));
if numel(seed_list) ~= numel(model_files)
    error('snn_primary_api:seedFileMismatch', ...
        'opts.seed_list must have one entry for each model file.');
end

result = struct();
result.analysis_kind = 'posthoc_spike_time_jitter';
result.jitter_is_recurrent = false;
result.description = ['The baseline closed-loop network is simulated once. ', ...
    'Jitter is applied to recorded spike events during decoder reconstruction only.'];
result.seed_results = struct([]);
result.seed_list = seed_list(:).';
for ii = 1:numel(model_files)
    one_opts = opts;
    one_opts.model_file = model_files{ii};
    one_opts.init_seed = seed_list(ii);
    one = analyse_one_model(one_opts);
    one.init_seed = seed_list(ii);
    result.seed_results(end + 1) = one; %#ok<AGROW>
end
end

function out = analyse_one_model(opts)
train_result = load_training_result(opts.model_file);
if isfield(train_result, 'seed_results')
    seed_index = max(1, min(numel(train_result.seed_results), round(get_opt(opts, 'seed_index', 1))));
    train_result = train_result.seed_results(seed_index);
end
if ~isfield(train_result, 'options')
    error('snn_primary_api:modelFile', 'Saved training result does not contain options.');
end
opts_eval = merge_options_with_seed(train_result.options, rmfield_if_present(opts, 'model_file'));
opts_eval = apply_saved_architecture_metadata(opts_eval, train_result);
opts_eval.T_sim = single(get_opt(opts_eval, 'closed_loop_test_time', opts_eval.T_sim));
opts_eval.closed_loop_warmup_time = single(get_opt(opts_eval, 'closed_loop_test_warmup_time', ...
    get_opt(opts_eval, 'closed_loop_warmup_time', 5)));
opts_eval.closed_loop_validation_ics = get_opt(opts_eval, 'closed_loop_test_ics', ...
    get_opt(opts_eval, 'closed_loop_validation_ics', 1));
opts_eval.closed_loop_ic_seed = get_opt(opts_eval, 'closed_loop_test_ic_seed', 123);
opts_eval.closed_loop_ic_include_reference = logical(get_opt(opts_eval, ...
    'closed_loop_test_include_reference', false));
opts_eval.closed_loop_ic_role = 'test';
if isfield(train_result, 'dynamics') && isstruct(train_result.dynamics)
    opts_eval = apply_saved_dynamics_metadata(opts_eval, train_result.dynamics);
end
if isfield(train_result, 'model') && isstruct(train_result.model)
    P = train_result.model;
else
    sys = make_dynamics_system_for_api(opts_eval.system_name);
    P = make_primary_model(sys.dim, sys.dim, opts_eval);
end
P = ensure_model_architecture_fields(P, opts_eval);
P.B = best_bias_from_result(train_result);

std_list = double(get_opt(opts_eval, 'spike_jitter_std_list', [0.001 0.005 0.010]));
if isempty(std_list) || any(~isfinite(std_list)) || any(std_list < 0)
    error('analyse_dynamics_spike_jitter:badStdList', ...
        'spike_jitter_std_list must contain finite nonnegative values in seconds.');
end
eval_set = make_closed_loop_eval_set(opts_eval);
n_ic = numel(eval_set.x_true);
cases = repmat(empty_case(), n_ic, 1);
for ic = 1:n_ic
    [pred_all, event_rows, event_steps] = simulate_baseline_events(P, eval_set.x_true{ic}, eval_set.lambda{ic});
    warmup_steps = double(eval_set.warmup_steps);
    if warmup_steps >= size(pred_all, 1)
        error('snn_primary_api:closedLoopWarmupTooLong', 'Warmup removes all simulated prediction steps.');
    end
    % The true system is initialized from the network state after warmup,
    % matching the standard closed-loop test protocol.
    if warmup_steps > 0
        baseline = pred_all(warmup_steps + 1:end, :);
        test_x0_norm = single(pred_all(warmup_steps, :).');
    else
        baseline = pred_all;
        test_x0_norm = single(eval_set.test_x0_norm{ic});
    end
    [truth, truth_diagnostic] = closed_loop_truth_from_network_state(test_x0_norm, eval_set);
    if isempty(truth)
        truth = nan(size(baseline), 'single');
    end
    n = min(size(baseline, 1), size(truth, 1));
    baseline = single(baseline(1:n, :));
    truth = single(truth(1:n, :));
    jitter = repmat(struct('std_s', [], 'pred_norm', [], 'mse_to_true', [], ...
        'mse_to_baseline', [], 'wasserstein_distance_to_true', []), 1, numel(std_list));
    for jj = 1:numel(std_list)
        rng_seed = double(get_opt(opts_eval, 'spike_jitter_seed', 999)) + 100003 * ic + 7919 * jj;
        pred_jitter_all = reconstruct_jittered_decoder(P, event_rows, event_steps, ...
            size(pred_all, 1), opts_eval.dt, std_list(jj), rng_seed);
        if warmup_steps > 0
            pred_jitter = pred_jitter_all(warmup_steps + 1:end, :);
        else
            pred_jitter = pred_jitter_all;
        end
        pred_jitter = single(pred_jitter(1:n, :));
        jitter(jj).std_s = std_list(jj);
        jitter(jj).pred_norm = pred_jitter;
        jitter(jj).mse_to_true = single(mean((pred_jitter - truth).^2, 'all', 'omitnan'));
        jitter(jj).mse_to_baseline = single(mean((pred_jitter - baseline).^2, 'all', 'omitnan'));
        jitter(jj).wasserstein_distance_to_true = single(phase_portrait_wasserstein_distance( ...
            double(pred_jitter), double(truth), opts_eval.wd));
    end
    cases(ic).baseline_pred_norm = baseline;
    cases(ic).true_norm = truth;
    cases(ic).test_x0_norm = test_x0_norm;
    cases(ic).truth_diagnostic = truth_diagnostic;
    cases(ic).jitter = jitter;
end
out = struct();
out.model_file = char(opts.model_file);
out.options = opts_eval;
out.jitter_is_recurrent = false;
out.jitter_method = 'posthoc_decoder_reconstruction_with_discrete_time_event_shifts';
out.spike_jitter_std_list = std_list;
out.closed_loop_cases = cases;
end

function [pred, event_rows, event_steps] = simulate_baseline_events(P, x, lambda)
steps = size(x, 2);
u = single(zeros(P.N_hidden, 1) + P.E_L); w = zeros(P.N_hidden, 1, 'single');
x_syn = zeros(P.N_hidden, 1, 'single'); r = zeros(P.N_hidden, 1, 'single');
z_prev = zeros(P.N_out, 1, 'single'); pred = zeros(steps - 1, P.N_out, 'single');
row_chunks = cell(steps - 1, 1);
for k = 1:steps - 1
    if k == 1 || lambda(k), x_in = x(:, k); else, x_in = z_prev; end
    I_in = P.W_in * (P.INPUT_SCALE * single(x_in));
    [u, w, ~, spike, ~, x_syn, r] = primary_step(P, I_in, u, w, x_syn, r);
    z_prev = P.W_out * r; pred(k, :) = z_prev.';
    row_chunks{k} = find(spike);
end
counts = cellfun(@numel, row_chunks);
event_rows = vertcat(row_chunks{:});
event_steps = repelem((1:steps - 1).', counts);
end

function pred = reconstruct_jittered_decoder(P, event_rows, event_steps, n_steps, dt, std_s, rng_seed)
% Event shifts are quantized at the model timestep, matching the discrete simulator.
old_rng = rng; cleanup = onCleanup(@() rng(old_rng)); %#ok<NASGU>
rng(rng_seed, 'twister');
if isempty(event_rows)
    events = sparse(P.N_hidden, n_steps);
else
    shifts = round((std_s / double(dt)) .* randn(numel(event_steps), 1));
    event_steps = min(max(event_steps + shifts, 1), n_steps);
    events = sparse(double(event_rows), double(event_steps), 1, P.N_hidden, n_steps);
end
x_syn = zeros(P.N_hidden, 1, 'single'); r = zeros(P.N_hidden, 1, 'single');
pred = zeros(n_steps, P.N_out, 'single');
for k = 1:n_steps
    x_syn = x_syn + P.spike_jump_sr .* single(full(events(:, k)));
    [x_syn, r] = cascade_advance(P, single(1), x_syn, r);
    pred(k, :) = (P.W_out * r).';
end
end

function value = get_opt(S, name, default_value)
if isstruct(S) && isfield(S, name) && ~isempty(S.(name))
    value = S.(name);
else
    value = default_value;
end
end

function out = empty_case()
out = struct('baseline_pred_norm', [], 'true_norm', [], 'test_x0_norm', [], ...
    'truth_diagnostic', struct(), 'jitter', struct([]));
end
