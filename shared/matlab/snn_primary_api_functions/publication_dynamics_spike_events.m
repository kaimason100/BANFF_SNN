% Package orientation: Shared implementation helper for publication analysis exports.

function events = publication_dynamics_spike_events(P, opts)
%PUBLICATION_DYNAMICS_SPIKE_EVENTS Save sparse events for offline timing tests.
%   Events span both the closed-loop warmup and test period.  EVENT_RHO is
%   retained because the two-stage synaptic filter uses each spike's
%   fractional within-step time. The CPU recording also stores its own
%   unperturbed decoder output and a true continuation initialised from that
%   CPU rollout's terminal warmup state. Later timing analyses therefore use
%   one backend consistently rather than mixing CPU events with GPU baselines.

events = empty_events();
try
    eval_set = make_closed_loop_eval_set(opts);
    n_ic = numel(eval_set.x_true);
    by_ic = repmat(empty_ic_events(), n_ic, 1);
    for ic = 1:n_ic
        [one, pred_all] = record_one_rollout(P, eval_set.x_true{ic}, eval_set.lambda{ic});
        one = attach_cpu_matched_test(one, pred_all, eval_set, ic);
        by_ic(ic) = one;
    end
    events.status = 'ok';
    events.schema_version = 2;
    events.recording_backend = 'cpu';
    events.baseline_backend = 'cpu_event_recording';
    events.truth_backend = 'cpu_true_continuation_from_cpu_warmup_state';
    events.dt = double(eval_set.opts.dt);
    events.n_hidden = double(P.N_hidden);
    events.n_output = double(P.N_out);
    events.closed_loop_warmup_steps = double(eval_set.warmup_steps);
    events.n_initial_conditions = n_ic;
    events.events_by_ic = by_ic;
    events.calculation = struct( ...
        'context', 'full_closed_loop_rollout_including_warmup', ...
        'event_step_definition', 'one-based prediction/update step', ...
        'event_neuron_definition', 'one-based hidden-neuron index', ...
        'event_rho_definition', 'fractional within-step spike time from primary_step', ...
        'baseline_definition', 'unperturbed CPU decoder output after warmup', ...
        'truth_definition', 'true continuation from the same CPU terminal warmup state', ...
        'storage', 'uint32 event steps and neurons; single event rho');
catch ME
    events.status = 'failed';
    events.message = ME.message;
    warning('publication_dynamics_spike_events:recordingFailed', ...
        'Could not save dynamics spike events: %s', ME.message);
end
end

function [entry, pred_all] = record_one_rollout(P, x, lambda)
n_steps = size(x, 2) - 1;
u = single(zeros(P.N_hidden, 1) + P.E_L);
w = zeros(P.N_hidden, 1, 'single');
x_syn = zeros(P.N_hidden, 1, 'single');
r = zeros(P.N_hidden, 1, 'single');
z_prev = zeros(P.N_out, 1, 'single');
neurons_by_step = cell(n_steps, 1);
rho_by_step = cell(n_steps, 1);
pred_all = zeros(n_steps, P.N_out, 'single');

for k = 1:n_steps
    if k == 1 || lambda(k)
        x_in = x(:, k);
    else
        x_in = z_prev;
    end
    I_in = P.W_in * (P.INPUT_SCALE * single(x_in));
    [u, w, rho, spike, ~, x_syn, r] = primary_step(P, I_in, u, w, x_syn, r);
    z_prev = P.W_out * r;
    pred_all(k, :) = z_prev.';
    neurons = find(spike);
    neurons_by_step{k} = uint32(neurons(:));
    rho_by_step{k} = single(rho(neurons));
end

counts = cellfun(@numel, neurons_by_step);
entry = empty_ic_events();
entry.n_steps = uint32(n_steps);
entry.n_events = uint64(sum(counts));
if entry.n_events == 0
    return;
end
entry.event_neurons = vertcat(neurons_by_step{:});
entry.event_rho = vertcat(rho_by_step{:});
entry.event_steps = uint32(repelem((1:n_steps).', counts));
end

function entry = attach_cpu_matched_test(entry, pred_all, eval_set, ic)
warmup_steps = max(0, round(double(eval_set.warmup_steps)));
if warmup_steps >= size(pred_all, 1)
    error('publication_dynamics_spike_events:warmupTooLong', ...
        'CPU event recording contains no scored samples after warmup.');
end
if warmup_steps > 0
    test_x0_norm = single(pred_all(warmup_steps, :).');
    baseline = single(pred_all(warmup_steps + 1:end, :));
else
    test_x0_norm = single(eval_set.test_x0_norm{ic});
    baseline = single(pred_all);
end
[truth, diagnostic] = closed_loop_truth_from_network_state(test_x0_norm, eval_set);
if isempty(truth)
    truth = nan(size(baseline), 'single');
else
    n = min(size(baseline, 1), size(truth, 1));
    baseline = baseline(1:n, :);
    truth = single(truth(1:n, :));
end
entry.baseline_norm = baseline;
entry.true_norm = truth;
entry.test_x0_norm = test_x0_norm;
entry.truth_diagnostic = diagnostic;
end

function events = empty_events()
events = struct('status', 'not_recorded', 'message', '', 'schema_version', 2, ...
    'recording_backend', '', 'baseline_backend', '', 'truth_backend', '', ...
    'dt', [], 'n_hidden', [], 'n_output', [], 'closed_loop_warmup_steps', [], ...
    'n_initial_conditions', [], 'events_by_ic', repmat(empty_ic_events(), 0, 1), ...
    'calculation', struct());
end

function entry = empty_ic_events()
entry = struct('event_steps', zeros(0, 1, 'uint32'), ...
    'event_neurons', zeros(0, 1, 'uint32'), 'event_rho', zeros(0, 1, 'single'), ...
    'n_steps', uint32(0), 'n_events', uint64(0), ...
    'baseline_norm', zeros(0, 0, 'single'), 'true_norm', zeros(0, 0, 'single'), ...
    'test_x0_norm', zeros(0, 1, 'single'), 'truth_diagnostic', struct());
end
