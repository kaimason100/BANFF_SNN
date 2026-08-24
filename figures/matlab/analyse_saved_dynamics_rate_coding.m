% Package orientation: Offline publication analysis of saved dynamics spike events.

function result = analyse_saved_dynamics_rate_coding(opts)
%ANALYSE_SAVED_DYNAMICS_RATE_CODING Analyse saved events without simulation.
%   Full baseline spike events, decoder outputs, matched network-started
%   true trajectories, models, and metadata are loaded from timestamped test
%   files. Warmup events remain unchanged.
%   Within the scored interval, each neuron's event count is preserved in every
%   shuffle window while its timing is randomized. The fixed filtered-spike
%   readout is then reconstructed offline.

if nargin < 1 || isempty(opts), opts = struct(); end
analyses = load_rate_coding_analyses(opts);
result = struct('analysis_kind', 'rate_preserving_within_window_shuffle', ...
    'analysis_scope', 'offline_fixed_decoder_only', ...
    'description', ['Warmup spike events are replayed unchanged. Scored-interval events are shuffled ', ...
        'within windows aligned to the scored-interval boundary, preserving every neuron''s count ', ...
        'in every window. The unshuffled event replay is verified against its saved decoder ', ...
        'output, and all conditions use the true continuation from the same warmup state.'], ...
    'systems', repmat(empty_system_result(), 0, 1), 'source_analysis_files', {cell(0, 1)});

for aa = 1:numel(analyses)
    A = analyses(aa).analysis;
    seed_indices = select_seed_indices(A.seeds, opts);
    system_result = empty_system_result();
    system_result.task_id = char(A.task_id);
    system_result.system_name = system_name_from_analysis(A);
    system_result.source_analysis_file = analyses(aa).path;
    system_result.created_at = get_field(A, 'created_at', '');
    system_result.seed_results = repmat(empty_seed_result(), 0, 1);
    for ss = seed_indices
        one = analyse_saved_seed(A.seeds(ss), opts);
        system_result.seed_results(end + 1, 1) = one;
    end
    require_metric_seed_count(system_result, opts);
    result.systems(end + 1, 1) = system_result;
    result.source_analysis_files{end + 1, 1} = analyses(aa).path;
end
end

function analyses = load_rate_coding_analyses(opts)
analysis_dir = get_opt(opts, 'analysis_dir', fullfile(project_root(), 'outputs', 'publication_analysis'));
task_ids = string(get_opt(opts, 'task_ids', strings(0, 1)));
if exist(analysis_dir, 'dir') ~= 7
    error('analyse_saved_dynamics_rate_coding:missingDirectory', 'Saved test-data directory does not exist.');
end
files = dir(fullfile(analysis_dir, '*_test_analysis_*.mat'));
latest = struct('task_id', {}, 'analysis', {}, 'path', {}, 'time', {});
for ii = 1:numel(files)
    path = fullfile(files(ii).folder, files(ii).name);
    try
        A = load_publication_analysis_file(path);
    catch ME
        warning('analyse_saved_dynamics_rate_coding:unreadableFile', 'Skipping %s: %s', files(ii).name, ME.message);
        continue;
    end
    if ~isstruct(A) || ~strcmp(char(get_field(A, 'task_family', '')), 'dynamical_systems') || ...
            ~isfield(A, 'seeds') || isempty(A.seeds)
        continue;
    end
    task_id = string(get_field(A, 'task_id', ''));
    if strlength(task_id) == 0 || (~isempty(task_ids) && ~any(task_id == task_ids))
        continue;
    end
    created = double(get_field(A, 'created_at_datenum', files(ii).datenum));
    previous = find(arrayfun(@(x) x.task_id == task_id, latest), 1);
    entry = struct('task_id', task_id, 'analysis', A, 'path', path, 'time', created);
    if isempty(previous)
        latest(end + 1, 1) = entry; %#ok<AGROW>
    elseif created > latest(previous).time
        latest(previous) = entry;
    end
end
if isempty(latest)
    error('analyse_saved_dynamics_rate_coding:noSavedTests', ...
        'No saved dynamics test matched task_ids. Run the matching test script first.');
end
analyses = latest;
end

function indices = select_seed_indices(seeds, opts)
requested = double(get_opt(opts, 'network_seed_indices', []));
if isempty(requested)
    indices = 1:numel(seeds);
    return;
end
actual = arrayfun(@(S, ii) double(get_field(S, 'seed_index', ii)), seeds, 1:numel(seeds));
missing = setdiff(requested(:).', actual);
if ~isempty(missing)
    error('analyse_saved_dynamics_rate_coding:missingSeeds', ...
        'Saved test data are missing requested network seed index/indices: %s.', mat2str(missing));
end
indices = find(ismember(actual, requested));
end

function out = analyse_saved_seed(seed, opts)
require_saved_events(seed);
P = regenerate_model(seed);
events = seed.spike_events;
% The unperturbed network output is retained separately as the sole baseline.
% Do not insert a one-timestep/no-shuffle condition here: it would duplicate
% that baseline in every figure without adding an independent perturbation.
window_s = unique(max(double(events.dt), double(get_opt(opts, ...
    'rate_shuffle_window_s', [.025 .050 .100 .200 .400 .800]))), 'stable');
window_steps = max(1, round(window_s ./ double(events.dt)));
n_ic = numel(events.events_by_ic);
conditions = repmat(empty_condition(), 1, numel(window_s));
for ww = 1:numel(conditions)
    conditions(ww).window_s = window_s(ww);
    conditions(ww).window_steps = window_steps(ww);
    conditions(ww).wd_by_ic = nan(n_ic, 1);
    conditions(ww).mse_by_ic = nan(n_ic, 1);
    conditions(ww).max_count_error_by_ic = nan(n_ic, 1);
end
example = struct('truth', [], 'baseline', [], 'shuffled', {cell(1, numel(conditions))}, ...
    'dt', double(events.dt), 'ic_index', []);
valid_truth_ic = false(n_ic, 1);
truth_diagnostic_by_ic = cell(n_ic, 1);
baseline_replay_max_abs_error_by_ic = nan(n_ic, 1);
shuffle_seed = double(get_opt(opts, 'rate_shuffle_seed', 2026));
replay_tolerance = double(get_opt(opts, 'baseline_replay_abs_tolerance', 1e-5));

for ic = 1:n_ic
    event = events.events_by_ic(ic);
    truth = single(event.true_norm);
    diagnostic = event.truth_diagnostic;
    truth_diagnostic_by_ic{ic} = diagnostic;
    start = double(events.closed_loop_warmup_steps) + 1;
    if start > double(event.n_steps)
        error('analyse_saved_dynamics_rate_coding:badWarmup', 'Saved warmup length removes all recorded decoder steps.');
    end
    original_events = event_maps(event, events.n_hidden);
    reconstructed_all = reconstruct_filtered_readout(P, original_events);
    saved_baseline = single(event.baseline_norm);
    n = min([size(truth, 1), size(saved_baseline, 1), size(reconstructed_all, 1) - start + 1]);
    if n < 1
        error('analyse_saved_dynamics_rate_coding:emptyTrajectory', ...
            'Saved event and trajectory lengths leave no post-warmup samples for IC %d.', ic);
    end
    truth = truth(1:n, :);
    baseline = reconstructed_all(start:start + n - 1, :);
    saved_baseline = saved_baseline(1:n, :);
    replay_error = max(abs(double(baseline) - double(saved_baseline)), [], 'all');
    baseline_replay_max_abs_error_by_ic(ic) = replay_error;
    if ~isfinite(replay_error) || replay_error > replay_tolerance
        error('analyse_saved_dynamics_rate_coding:baselineReplayMismatch', ...
            ['Unshuffled event reconstruction differs from the saved decoder output ', ...
             'for IC %d (max abs error %.9g; tolerance %.9g).'], ...
            ic, replay_error, replay_tolerance);
    end
    if ~all(isfinite(truth), 'all')
        for ww = 1:numel(conditions), conditions(ww).wd_by_ic(ic) = Inf; end
        continue;
    end
    valid_truth_ic(ic) = true;
    if isempty(example.truth)
        example.truth = truth; example.baseline = baseline; example.ic_index = ic;
    end
    for ww = 1:numel(conditions)
        if window_steps(ww) == 1
            shuffled = baseline;
            shuffled_events = original_events;
        else
            shuffled_events = shuffle_scored_interval(event, events.n_hidden, ...
                window_steps(ww), start, shuffle_seed + 100003 * ic + 7919 * ww);
            shuffled_all = reconstruct_filtered_readout(P, shuffled_events);
            shuffled = shuffled_all(start:start + n - 1, :);
        end
        conditions(ww).wd_by_ic(ic) = banff_plot('phase_distance', double(shuffled), double(truth), get_opt(seed.options, 'wd', struct()));
        conditions(ww).mse_by_ic(ic) = mean((double(shuffled) - double(truth)).^2, 'all', 'omitnan');
        conditions(ww).max_count_error_by_ic(ic) = max_window_count_error( ...
            event, shuffled_events, window_steps(ww), start);
        if example.ic_index == ic, example.shuffled{ww} = shuffled; end
    end
end
if isempty(example.truth)
    error('analyse_saved_dynamics_rate_coding:noFiniteTruth', 'No saved test IC has a finite true continuation.');
end
for ww = 1:numel(conditions)
    conditions(ww).wd = mean(conditions(ww).wd_by_ic, 'omitnan');
    conditions(ww).mse = mean(conditions(ww).mse_by_ic, 'omitnan');
    conditions(ww).max_count_error = max(conditions(ww).max_count_error_by_ic, [], 'omitnan');
end
out = empty_seed_result();
out.init_seed = get_field(seed, 'init_seed', get_field(seed, 'seed_index', NaN));
out.seed_index = get_field(seed, 'seed_index', NaN);
out.model_file = get_field(seed, 'model_file', '');
out.options = seed.options;
out.analysis_scope = 'offline_fixed_decoder_only';
out.decoder_type = 'fixed_linear_readout_of_two_stage_filtered_saved_spikes';
out.baseline_source = 'verified_unshuffled_event_reconstruction';
out.truth_source = 'true_continuation_from_same_network_warmup_state';
out.baseline_replay_abs_tolerance = replay_tolerance;
out.baseline_replay_max_abs_error_by_ic = baseline_replay_max_abs_error_by_ic;
out.conditions = conditions;
out.example = example;
out.valid_truth_ic = valid_truth_ic;
out.truth_diagnostic_by_ic = truth_diagnostic_by_ic;
end

function require_saved_events(seed)
if ~isfield(seed, 'spike_events') || ~isstruct(seed.spike_events) || ...
        ~strcmp(get_field(seed.spike_events, 'status', ''), 'ok')
    error('analyse_saved_dynamics_rate_coding:eventsUnavailable', ...
        ['Saved spike events are unavailable. Rerun this dynamics test once with ', ...
         'save_publication_spike_events enabled (the default).']);
end
required = {'event_neurons', 'event_steps', 'event_rho', 'n_steps'};
if ~isfield(seed.spike_events, 'events_by_ic') || isempty(seed.spike_events.events_by_ic) || ...
        ~isfield(seed, 'architecture') || ~isstruct(seed.architecture) || ...
        ~isfield(seed, 'options') || ~isstruct(seed.options) || ...
        ~all(isfield(seed.spike_events.events_by_ic, [required, ...
            {'baseline_norm', 'true_norm', 'test_x0_norm', 'truth_diagnostic'}])) || ...
        double(get_field(seed.spike_events, 'schema_version', 0)) < 2 || ...
        ~any(strcmp(get_field(seed.spike_events, 'recording_backend', ''), ...
        {'cpu', 'matlab_gpu_arrayfun'}))
    error('analyse_saved_dynamics_rate_coding:invalidEvents', ...
        ['Saved spike-event data predate the single-backend baseline fix or are incomplete. ', ...
         'Rerun the corresponding dynamical-system test scripts before this analysis.']);
end
end

function maps = event_maps(event, n_hidden)
maps = struct();
maps.rho = sparse(double(event.event_neurons), double(event.event_steps), double(event.event_rho), ...
    double(n_hidden), double(event.n_steps));
% A crossing can occur at rho = 0. Sparse rho would omit it, so spike
% presence is retained independently of the fractional event time.
maps.spike = sparse(double(event.event_neurons), double(event.event_steps), ones(numel(event.event_steps), 1), ...
    double(n_hidden), double(event.n_steps));
end

function shuffled = shuffle_scored_interval(event, n_hidden, window_steps, scored_start, rng_seed)
old_rng = rng; cleanup = onCleanup(@() rng(old_rng));
rng(rng_seed, 'twister');
n_steps = double(event.n_steps); rows = double(event.event_neurons); cols = double(event.event_steps); rhos = double(event.event_rho);
scored_start = max(1, min(n_steps + 1, round(double(scored_start))));
new_rows = zeros(size(rows)); new_cols = zeros(size(cols)); new_rhos = zeros(size(rhos)); cursor = 0;

% Replay every warmup event at its original neuron, timestep, and event fraction.
is_warmup = cols < scored_start;
n_warmup = nnz(is_warmup);
if n_warmup > 0
    write = 1:n_warmup;
    new_rows(write) = rows(is_warmup);
    new_cols(write) = cols(is_warmup);
    new_rhos(write) = rhos(is_warmup);
    cursor = n_warmup;
end

% Start non-overlapping shuffle windows at the scored-interval boundary so no
% window can move an event between warmup and the scored test interval.
for start_col = scored_start:window_steps:n_steps
    stop_col = min(n_steps, start_col + window_steps - 1); in_window = cols >= start_col & cols <= stop_col;
    window_rows = rows(in_window); window_rhos = rhos(in_window);
    for neuron = unique(window_rows).'
        idx = find(window_rows == neuron); count = numel(idx);
        write = cursor + (1:count);
        new_rows(write) = neuron;
        new_cols(write) = randperm(stop_col - start_col + 1, count) + start_col - 1;
        new_rhos(write) = window_rhos(idx);
        cursor = cursor + count;
    end
end
if cursor ~= numel(rows), error('analyse_saved_dynamics_rate_coding:shuffleEventLoss', 'Spike-event shuffle lost events.'); end
shuffled = struct();
shuffled.rho = sparse(new_rows, new_cols, new_rhos, double(n_hidden), n_steps);
shuffled.spike = sparse(new_rows, new_cols, ones(numel(new_rows), 1), double(n_hidden), n_steps);
end

function pred = reconstruct_filtered_readout(P, events)
n_steps = size(events.rho, 2); x_syn = zeros(P.N_hidden, 1, 'single'); r = zeros(P.N_hidden, 1, 'single');
pred = zeros(n_steps, P.N_output, 'single');
for k = 1:n_steps
    rho = single(full(events.rho(:, k)));
    [x_syn, r] = banff_plot('cascade', P, rho, x_syn, r);
    spike = full(events.spike(:, k)) > 0;
    x_syn(spike) = x_syn(spike) + P.synapticJump;
    [x_syn, r] = banff_plot('cascade', P, single(1) - rho, x_syn, r);
    pred(k, :) = (P.W_out * r).';
end
end

function value = max_window_count_error(event, shuffled, width, scored_start)
original = sparse(double(event.event_neurons), double(event.event_steps), ones(numel(event.event_steps), 1), ...
    size(shuffled.spike, 1), size(shuffled.spike, 2));
value = 0;
for start_col = scored_start:width:size(shuffled.spike, 2)
    stop_col = min(size(shuffled.spike, 2), start_col + width - 1);
    value = max(value, full(max(abs(sum(original(:, start_col:stop_col), 2) - ...
        sum(shuffled.spike(:, start_col:stop_col) > 0, 2)))));
end
end

function require_metric_seed_count(system_result, opts)
required = double(get_opt(opts, 'required_metric_network_seeds', 3));
if numel(system_result.seed_results) ~= required
    error('analyse_saved_dynamics_rate_coding:metricSeedCount', ...
        'Task %s has %d saved selected network seeds; this analysis requires exactly %d.', ...
        system_result.task_id, numel(system_result.seed_results), required);
end
end

function name = system_name_from_analysis(A)
name = char(A.task_id);
if isfield(A, 'seeds') && ~isempty(A.seeds) && isfield(A.seeds(1), 'options')
    name = char(get_opt(A.seeds(1).options, 'system_name', name));
end
end

function out = empty_system_result()
out = struct('task_id', '', 'system_name', '', 'source_analysis_file', '', 'created_at', '', 'seed_results', repmat(empty_seed_result(), 0, 1));
end

function out = empty_seed_result()
out = struct('init_seed', [], 'seed_index', [], 'model_file', '', 'options', struct(), ...
    'analysis_scope', '', 'decoder_type', '', 'baseline_source', '', 'truth_source', '', ...
    'baseline_replay_abs_tolerance', [], 'baseline_replay_max_abs_error_by_ic', [], ...
    'conditions', repmat(empty_condition(), 0, 1), ...
    'example', struct(), 'valid_truth_ic', [], 'truth_diagnostic_by_ic', {cell(0, 1)});
end

function condition = empty_condition()
condition = struct('window_s', [], 'window_steps', [], 'wd_by_ic', [], 'mse_by_ic', [], ...
    'max_count_error_by_ic', [], 'wd', [], 'mse', [], 'max_count_error', []);
end

function value = get_opt(S, name, default_value)
if isstruct(S) && isfield(S, name) && ~isempty(S.(name)), value = S.(name); else, value = default_value; end
end

function value = get_field(S, name, default_value)
if isstruct(S) && isfield(S, name) && ~isempty(S.(name)), value = S.(name); else, value = default_value; end
end
function P = regenerate_model(seed)
if ~isfield(seed, 'architecture') || ~isfield(seed, 'options') || ~isfield(seed, 'bias')
    error('analyse_saved_dynamics_rate_coding:modelMetadata', ...
        'Saved analysis lacks the metadata needed to regenerate fixed matrices.');
end
P = banff_model('create', seed.architecture.N_in, seed.architecture.N_out, seed.options);
P.B = single(seed.bias(:));
end

