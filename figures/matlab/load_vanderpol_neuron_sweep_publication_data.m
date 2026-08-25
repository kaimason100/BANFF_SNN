% Package orientation: Shared loader for Van der Pol neuron-count figures.

function sweep = load_vanderpol_neuron_sweep_publication_data(opts)
%LOAD_VANDERPOL_NEURON_SWEEP_PUBLICATION_DATA Load saved Van der Pol tests.

if nargin < 1 || isempty(opts), opts = struct(); end
analysis_dir = get_opt_local(opts, 'analysis_dir', ...
    fullfile(project_root(), 'outputs', 'publication_analysis'));
neuron_counts = double(get_opt_local(opts, 'neuron_counts', [1000 2000 4000 8000 16000 32000]));
example_seed_index = double(get_opt_local(opts, 'example_seed_index', 1));
required_test_ic_count = double(get_opt_local(opts, 'required_test_ic_count', 5));
require_complete_test_ics = logical(get_opt_local(opts, 'require_complete_test_ics', true));

if exist(analysis_dir, 'dir') ~= 7
    error('load_vanderpol_neuron_sweep_publication_data:missingDirectory', ...
        'Saved test-data directory does not exist: %s', analysis_dir);
end
files = dir(fullfile(analysis_dir, '*_test_analysis_*.mat'));
latest = repmat(struct('analysis', [], 'path', '', 'time', -inf), 1, numel(neuron_counts));
for ii = 1:numel(files)
    path = fullfile(files(ii).folder, files(ii).name);
    try
        A = load_publication_analysis_file(path);
    catch ME
        warning('load_vanderpol_neuron_sweep_publication_data:unreadableFile', ...
            'Skipping unreadable saved test-data file %s: %s', files(ii).name, ME.message);
        continue;
    end
    if ~is_vanderpol_analysis(A), continue; end
    n_hidden = analysis_n_hidden(A);
    if ~is_expected_profile(A,n_hidden), continue; end
    index = find(neuron_counts == n_hidden, 1);
    if isempty(index), continue; end
    t = double(get_field_local(A, 'created_at_datenum', files(ii).datenum));
    if isempty(latest(index).analysis) || t > latest(index).time
        latest(index) = struct('analysis', A, 'path', path, 'time', t);
    end
end

missing_mask = false(size(neuron_counts));
for ii = 1:numel(latest)
    missing_mask(ii) = isempty(latest(ii).analysis);
end
missing = neuron_counts(missing_mask);
if ~isempty(missing)
    error('load_vanderpol_neuron_sweep_publication_data:missingCounts', ...
        'No saved Van der Pol test analysis was found for N_hidden = [%s].', num2str(missing));
end

conditions = repmat(empty_condition(), 1, numel(neuron_counts));
for ii = 1:numel(neuron_counts)
    A = latest(ii).analysis;
    seeds = A.seeds;
    [example, example_position] = find_seed_entry(seeds, example_seed_index);
    [pred, truth] = first_closed_loop_trajectory(example.test);
    if ~isequal(size(pred),size(truth))
        error('load_vanderpol_neuron_sweep_publication_data:trajectoryShape', ...
            'Van der Pol N_hidden=%d prediction and truth shapes differ.',neuron_counts(ii));
    end
    n = size(pred,1);
    if n < 2 || any(~isfinite(pred), 'all') || any(~isfinite(truth), 'all')
        error('load_vanderpol_neuron_sweep_publication_data:badExample', ...
            'Van der Pol N_hidden=%d has no finite example trajectory for seed index %d.', neuron_counts(ii), example_seed_index);
    end
    % A neuron-count sweep compares one network seed at every size. Do not
    % mix the three 32k primary-model seeds into the five IC swarm.
    [wd, n_ic] = phase_portrait_wd_values(example.test, example.options);
    finite = isfinite(wd);
    if require_complete_test_ics && n_ic ~= required_test_ic_count
        error('load_vanderpol_neuron_sweep_publication_data:incompleteTestIcs', ...
            ['Van der Pol N_hidden=%d, network seed %d has %d saved test ICs; this figure requires %d. ', ...
             'Run/save the missing full tests or set require_complete_test_ics=false.'], ...
            neuron_counts(ii), example_seed_index, n_ic, required_test_ic_count);
    end
    conditions(ii).n_hidden = neuron_counts(ii);
    conditions(ii).label = sprintf('%dk', neuron_counts(ii) / 1000);
    conditions(ii).source_file = latest(ii).path;
    conditions(ii).task_id = char(get_field_local(A, 'task_id', 'vanderpol'));
    conditions(ii).wd = wd(finite);
    conditions(ii).example_seed_index = double(get_field_local(example, 'seed_index', example_position));
    conditions(ii).dt = double(get_option_field(example.options, 'dt', 1));
    conditions(ii).pred_norm = double(pred(1:n, :));
    conditions(ii).true_norm = double(truth(1:n, :));
end
sweep = struct('neuron_counts', neuron_counts, 'conditions', conditions, ...
    'example_seed_index', example_seed_index, 'required_test_ic_count', required_test_ic_count);
end

function tf = is_vanderpol_analysis(A)
tf = isstruct(A) && strcmp(char(get_field_local(A, 'task_family', '')), 'dynamical_systems') && ...
    isfield(A, 'seeds') && ~isempty(A.seeds);
if ~tf, return; end
opts = get_field_local(A.seeds(1), 'options', struct());
name = lower(string(get_option_field(opts, 'system_name', '')));
tf = contains(name, 'vanderpol') || contains(name, 'van_der_pol');
end

function tf = is_expected_profile(A,n_hidden)
taskId = string(get_field_local(A,'task_id',''));
if n_hidden == 32000
    tf = taskId == "dynamical_systems_vanderpol";
else
    expected = "dynamical_systems_vanderpol_neuron_sweep_N" + string(n_hidden);
    tf = taskId == expected;
end
end

function n_hidden = analysis_n_hidden(A)
seed = A.seeds(1); n_hidden = NaN;
arch = get_field_local(seed, 'architecture', struct());
model = get_field_local(seed, 'model', struct());
opts = get_field_local(seed, 'options', struct());
for candidate = {get_option_field(arch, 'N_hidden', []), get_option_field(model, 'N_hidden', []), get_option_field(opts, 'N_hidden', [])}
    value = candidate{1};
    if ~isempty(value) && isfinite(double(value(1)))
        n_hidden = double(value(1)); return;
    end
end
end

function [seed, position] = find_seed_entry(seeds, wanted)
position = find(arrayfun(@(s) double(get_field_local(s, 'seed_index', NaN)) == wanted, seeds), 1);
if isempty(position)
    error('load_vanderpol_neuron_sweep_publication_data:missingExampleSeed', ...
        'Saved analysis does not contain example seed index %d.', wanted);
end
seed = seeds(position);
end

function [pred, truth] = first_closed_loop_trajectory(test)
if isfield(test, 'closed_loop') && isstruct(test.closed_loop), test = test.closed_loop; end
if isfield(test, 'pred_norm_by_ic') && isfield(test, 'true_norm_by_ic')
    pred = first_cell(test.pred_norm_by_ic); truth = first_cell(test.true_norm_by_ic);
elseif isfield(test, 'pred_norm') && isfield(test, 'true_norm')
    pred = test.pred_norm; truth = test.true_norm;
else
    error('load_vanderpol_neuron_sweep_publication_data:missingTrajectory', ...
        'Saved test entry does not contain closed-loop trajectories.');
end
end

function [wd, n_ic] = phase_portrait_wd_values(test, options)
if isfield(test, 'closed_loop') && isstruct(test.closed_loop), test = test.closed_loop; end
if isfield(test, 'pred_norm_by_ic') && isfield(test, 'true_norm_by_ic')
    pred_by_ic = test.pred_norm_by_ic; truth_by_ic = test.true_norm_by_ic;
elseif isfield(test, 'pred_norm') && isfield(test, 'true_norm')
    pred_by_ic = {test.pred_norm}; truth_by_ic = {test.true_norm};
else
    wd = NaN; n_ic = 0; return;
end
if ~iscell(pred_by_ic), pred_by_ic = {pred_by_ic}; end
if ~iscell(truth_by_ic), truth_by_ic = {truth_by_ic}; end
if numel(pred_by_ic) ~= numel(truth_by_ic)
    error('load_vanderpol_neuron_sweep_publication_data:trajectoryCount', ...
        'Saved prediction and truth initial-condition counts differ.');
end
wd_options = get_option_field(options, 'wd', struct('NumProjections', 128, ...
    'TrimFraction', .10, 'Subsample', 5, 'TransientFraction', .10, 'MaxPoints', 1250));
wd = [];
n_ic = numel(pred_by_ic);
for ic = 1:n_ic
    pred = double(pred_by_ic{ic}); truth = double(truth_by_ic{ic});
    if ~isequal(size(pred),size(truth))
        error('load_vanderpol_neuron_sweep_publication_data:trajectoryShape', ...
            'Saved prediction and truth trajectories have different shapes.');
    end
    n = size(pred,1); d = size(pred,2);
    if n < 2 || d < 2, continue; end
    pairs = nchoosek(1:d, 2);
    pair_wd = nan(size(pairs, 1), 1);
    for pp = 1:size(pairs, 1)
        pair_wd(pp) = banff_plot('phase_distance', ...
            pred(1:n, pairs(pp, :)), truth(1:n, pairs(pp, :)), wd_options);
    end
    % Van der Pol is two-dimensional, so each IC contributes one phase-plane WD.
    wd(end + 1) = mean(pair_wd, 'omitnan'); %#ok<AGROW>
end
if isempty(wd), wd = NaN; end
end

function value = first_cell(value)
if iscell(value), value = value{1}; end
end

function condition = empty_condition()
condition = struct('n_hidden', [], 'label', '', 'source_file', '', 'task_id', '', ...
    'wd', [], 'example_seed_index', [], 'dt', [], ...
    'pred_norm', [], 'true_norm', []);
end

function value = get_opt_local(S, name, default_value)
if isstruct(S) && isfield(S, name) && ~isempty(S.(name)), value = S.(name); else, value = default_value; end
end

function value = get_field_local(S, name, default_value)
if isstruct(S) && isfield(S, name) && ~isempty(S.(name)), value = S.(name); else, value = default_value; end
end

function value = get_option_field(S, name, default_value)
value = get_field_local(S, name, default_value);
end
