% Package orientation: Loader for saved Yacht and breast-cancer neuron-sweep analyses.

function sweep = load_static_neuron_sweep_publication_data(task_id, opts)
%LOAD_STATIC_NEURON_SWEEP_PUBLICATION_DATA Load saved low-rank static-task sweeps.
% task_id is 'classification_BC' or 'regression_yacht'.

if nargin < 2 || isempty(opts), opts = struct(); end
task_id = char(string(task_id));
if ~ismember(task_id, {'classification_BC', 'regression_yacht'})
    error('load_static_neuron_sweep_publication_data:unsupportedTask', ...
        'Supported tasks are classification_BC and regression_yacht.');
end
analysis_dir = get_opt_local(opts, 'analysis_dir', fullfile(project_root(), 'outputs', 'publication_analysis'));
neuron_counts = double(get_opt_local(opts, 'neuron_counts', [1000 2000 4000 8000 16000 32000]));
network_seed_index = double(get_opt_local(opts, 'network_seed_index', 1));
if exist(analysis_dir, 'dir') ~= 7
    error('load_static_neuron_sweep_publication_data:missingDirectory', 'Saved test-data directory does not exist: %s', analysis_dir);
end

files = dir(fullfile(analysis_dir, '*_test_analysis_*.mat'));
latest = repmat(struct('analysis', [], 'path', '', 'time', -inf), 1, numel(neuron_counts));
for ii = 1:numel(files)
    path = fullfile(files(ii).folder, files(ii).name);
    try
        A = load_publication_analysis_file(path);
    catch ME
        warning('load_static_neuron_sweep_publication_data:unreadableFile', 'Skipping %s: %s', files(ii).name, ME.message);
        continue;
    end
    if ~is_requested_lowrank_analysis(A, task_id), continue; end
    n_hidden = analysis_n_hidden(A);
    index = find(neuron_counts == n_hidden, 1);
    if isempty(index), continue; end
    t = double(get_field_local(A, 'created_at_datenum', files(ii).datenum));
    if isempty(latest(index).analysis) || t > latest(index).time
        latest(index) = struct('analysis', A, 'path', path, 'time', t);
    end
end

missing = neuron_counts(arrayfun(@(x) isempty(x.analysis), latest));
if ~isempty(missing)
    error('load_static_neuron_sweep_publication_data:missingCounts', ...
        'No saved %s low-rank test analysis was found for N_hidden = [%s].', task_id, num2str(missing));
end

conditions = repmat(empty_condition(), 1, numel(neuron_counts));
for ii = 1:numel(neuron_counts)
    A = latest(ii).analysis;
    [accuracy, rmse, pearson_r, pearson_p, seeds] = extract_metrics(A.seeds, network_seed_index);
    conditions(ii).n_hidden = neuron_counts(ii);
    conditions(ii).label = sprintf('%dk', neuron_counts(ii) / 1000);
    conditions(ii).source_file = latest(ii).path;
    conditions(ii).seed_indices = seeds;
    conditions(ii).accuracy = accuracy;
    conditions(ii).rmse = rmse;
    conditions(ii).pearson_r = pearson_r;
    conditions(ii).pearson_p = pearson_p;
end
sweep = struct('task_id', task_id, 'neuron_counts', neuron_counts, ...
    'network_seed_index', network_seed_index, 'conditions', conditions);
end

function tf = is_requested_lowrank_analysis(A, task_id)
% Select by structured publication metadata, never by historical filenames.
tf = isstruct(A) && isfield(A,'task_id') && isfield(A,'seeds') && ~isempty(A.seeds);
if ~tf, return; end
savedTask = char(string(A.task_id));
validTaskId = strcmp(savedTask, task_id) || ...
    ~isempty(regexp(savedTask, ['^' regexptranslate('escape',task_id) '_neuron_sweep_N[0-9]+$'], 'once'));
if ~validTaskId, tf = false; return; end
seed = A.seeds(1);
arch = get_field_local(seed,'architecture',struct());
options = get_field_local(seed,'options',struct());
if lower(string(get_field_local(arch,'recurrent_mode',''))) ~= "low_rank"
    tf = false; return;
end
if lower(string(get_field_local(options,'method','eprop'))) ~= "eprop"
    tf = false; return;
end
family = char(string(get_field_local(A,'task_family','')));
if strcmp(task_id,'classification_BC')
    tf = strcmp(family,'classification');
else
    tf = strcmp(family,'regression');
end
end

function n_hidden = analysis_n_hidden(A)
seed = A.seeds(1);
arch = get_field_local(seed, 'architecture', struct());
n_hidden = double(get_field_local(arch, 'N_hidden', NaN));
end

function [accuracy, rmse, pearson_r, pearson_p, seed_indices] = extract_metrics(seeds, wanted_seed_index)
seed_position = find(arrayfun(@(x) scalar_field(x, 'seed_index', NaN) == wanted_seed_index, seeds), 1);
if isempty(seed_position)
    available = arrayfun(@(x) scalar_field(x, 'seed_index', NaN), seeds);
    error('load_static_neuron_sweep_publication_data:missingSeed', ...
        'Requested network seed %d is unavailable; saved seeds are [%s].', wanted_seed_index, num2str(available));
end
seed = seeds(seed_position); test = get_field_local(seed, 'test', struct());
metrics = get_field_local(seed, 'metrics', struct());
accuracy = scalar_field(test, 'metric', scalar_field(metrics, 'metric', NaN));
regression = get_field_local(test, 'regression', get_field_local(metrics, 'regression', struct()));
rmse = scalar_field(regression, 'rmse', NaN);
pearson_r = scalar_field(regression, 'pearson_r', scalar_field(regression, 'r', NaN));
pearson_p = scalar_field(regression, 'pearson_p', scalar_field(regression, 'p', NaN));
seed_indices = scalar_field(seed, 'seed_index', seed_position);
end

function condition = empty_condition()
condition = struct('n_hidden', [], 'label', '', 'source_file', '', 'seed_indices', [], ...
    'accuracy', [], 'rmse', [], 'pearson_r', [], 'pearson_p', []);
end

function value = scalar_field(S, name, default_value)
value = get_field_local(S, name, default_value);
if isempty(value) || ~isnumeric(value), value = default_value; else, value = double(value(1)); end
end
function value = get_opt_local(S, name, default_value), value = get_field_local(S, name, default_value); end
function value = get_field_local(S, name, default_value), if isstruct(S) && isfield(S, name) && ~isempty(S.(name)), value = S.(name); else, value = default_value; end, end
