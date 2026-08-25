% Package orientation: Shared loader for plotting scripts that consume saved dynamics tests.

function result = load_saved_dynamics_test_results(opts)
%LOAD_SAVED_DYNAMICS_TEST_RESULTS Load the newest valid analysis file per task.
%   RESULT has the same seed_results/test/options layout expected by the
%   shared dynamical-system plotting helpers.  It never evaluates a model.

if nargin < 1 || isempty(opts)
    opts = struct();
end
analysis_dir = get_opt_local(opts, 'analysis_dir', ...
    fullfile(project_root(), 'outputs', 'publication_analysis'));
task_ids = string(get_opt_local(opts, 'task_ids', strings(0, 1)));
seed_indices = double(get_opt_local(opts, 'seed_indices', []));
first_seed_only = logical(get_opt_local(opts, 'first_seed_only', false));
deduplicate_systems = logical(get_opt_local(opts, 'deduplicate_systems', false));
required_n_hidden = double(get_opt_local(opts, 'required_n_hidden', []));

if exist(analysis_dir, 'dir') ~= 7
    error('load_saved_dynamics_test_results:missingDirectory', ...
        'Saved test-data directory does not exist: %s', analysis_dir);
end

files = dir(fullfile(analysis_dir, '*_test_analysis_*.mat'));
if isempty(files)
    error('load_saved_dynamics_test_results:noFiles', ...
        'No timestamped saved test-data files were found in %s.', analysis_dir);
end

latest = struct('analysis', {}, 'path', {}, 'time', {}, 'key', {});
for ii = 1:numel(files)
    file_path = fullfile(files(ii).folder, files(ii).name);
    try
        A = load_publication_analysis_file(file_path);
    catch ME
        warning('load_saved_dynamics_test_results:unreadableFile', ...
            'Skipping unreadable saved test-data file %s: %s', files(ii).name, ME.message);
        continue;
    end
    if ~isstruct(A) || ~strcmp(char(get_field_local(A, 'task_family', '')), 'dynamical_systems') || ...
            ~isfield(A, 'seeds') || isempty(A.seeds)
        continue;
    end
    task_id = string(get_field_local(A, 'task_id', ''));
    if strlength(task_id) == 0 || (~isempty(task_ids) && ~any(task_id == task_ids))
        continue;
    end
    if ~analysis_matches_hidden_size(A, required_n_hidden)
        continue;
    end
    candidate_time = double(get_field_local(A, 'created_at_datenum', files(ii).datenum));
    if deduplicate_systems
        selection_key = system_key_from_analysis(A, task_id);
    else
        selection_key = task_id;
    end
    existing = find(arrayfun(@(x) x.key == selection_key, latest), 1);
    if isempty(existing)
        latest(end + 1) = struct('analysis', A, 'path', file_path, 'time', candidate_time, 'key', selection_key); %#ok<AGROW>
    elseif candidate_time > latest(existing).time
        latest(existing) = struct('analysis', A, 'path', file_path, 'time', candidate_time, 'key', selection_key);
    end
end

if isempty(latest)
    error('load_saved_dynamics_test_results:noMatchingFiles', ...
        'No valid saved dynamical-system tests matched the requested task selection.');
end

result = struct();
% Cell wrapping keeps this outer struct scalar even when a saved test/options
% field is itself a struct array with task-specific subfields.
result.seed_results = struct('test', {}, 'options', {}, 'init_seed', {}, ...
    'seed_index', {}, 'model_file', {}, 'task_id', {});
result.seed_list = [];
result.source_analysis_files = {latest.path};
for ii = 1:numel(latest)
    A = latest(ii).analysis;
    for jj = 1:numel(A.seeds)
        seed = A.seeds(jj);
        if first_seed_only && jj ~= 1
            continue;
        end
        if ~isempty(seed_indices) && ~any(double(get_field_local(seed, 'seed_index', jj)) == seed_indices)
            continue;
        end
        if ~isfield(seed, 'test') || ~isstruct(seed.test) || ...
                ~isfield(seed, 'options') || ~isstruct(seed.options)
            warning('load_saved_dynamics_test_results:missingFields', ...
                'Skipping incomplete saved test entry for task %s, seed index %d.', A.task_id, jj);
            continue;
        end
        entry = struct( ...
            'test', {seed.test}, ...
            'options', {seed.options}, ...
            'init_seed', get_field_local(seed, 'init_seed', get_field_local(seed, 'seed_index', jj)), ...
            'seed_index', get_field_local(seed, 'seed_index', jj), ...
            'model_file', {get_field_local(seed, 'model_file', '')}, ...
            'task_id', {A.task_id});
        result.seed_results(end + 1, 1) = entry;
        result.seed_list(end + 1, 1) = entry.init_seed;
    end
end

if isempty(result.seed_results)
    error('load_saved_dynamics_test_results:noSeeds', ...
        'The matching saved test-data files did not contain the requested seed entries.');
end
end

function value = get_opt_local(S, name, default_value)
if isstruct(S) && isfield(S, name) && ~isempty(S.(name))
    value = S.(name);
else
    value = default_value;
end
end

function value = get_field_local(S, name, default_value)
if isstruct(S) && isfield(S, name) && ~isempty(S.(name))
    value = S.(name);
else
    value = default_value;
end
end

function key = system_key_from_analysis(analysis, fallback_task_id)
key = string(fallback_task_id);
if ~isstruct(analysis) || ~isfield(analysis, 'seeds') || isempty(analysis.seeds)
    return;
end
seed = analysis.seeds(1);
if ~isfield(seed, 'options') || ~isstruct(seed.options) || ...
        ~isfield(seed.options, 'system_name') || isempty(seed.options.system_name)
    return;
end
name = lower(string(seed.options.system_name));
name = regexprep(name, '[^a-z0-9]+', '');
if contains(name, 'vanderpol') || contains(name, 'vdp')
    key = "vanderpol";
elseif contains(name, 'lorenz')
    key = "lorenz";
elseif contains(name, 'sprott')
    key = "sprotts";
else
    key = name;
end
end

function tf = analysis_matches_hidden_size(analysis, required_n_hidden)
if isempty(required_n_hidden), tf = true; return; end
tf = false;
if ~isstruct(analysis) || ~isfield(analysis,'seeds') || isempty(analysis.seeds), return; end
seed = analysis.seeds(1);
if ~isfield(seed,'architecture') || ~isfield(seed.architecture,'N_hidden'), return; end
n_hidden = double(seed.architecture.N_hidden);
tf = isfinite(n_hidden) && any(n_hidden == required_n_hidden(:));
end
