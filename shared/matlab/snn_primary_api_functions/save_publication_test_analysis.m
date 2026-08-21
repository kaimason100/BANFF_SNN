% Package orientation: Shared implementation helper for publication analysis exports.

function analysis = save_publication_test_analysis(test_result, task_family, opts)
%SAVE_PUBLICATION_TEST_ANALYSIS Save reusable full-test data for figures.
%   A timestamp is included in the output filename and complete metadata are
%   stored inside the MAT file.

if nargin < 3 || isempty(opts)
    opts = struct();
end
output_dir = get_opt(opts, 'publication_analysis_dir', ...
    fullfile(project_root(), 'outputs', 'publication_analysis'));
if exist(output_dir, 'dir') ~= 7
    mkdir(output_dir);
end
batch_size = get_opt(opts, 'publication_static_rate_batch_size', 32);

[seed_results, seed_list] = unpack_publication_seed_results(test_result);
if isempty(seed_results)
    error('save_publication_test_analysis:emptyResult', 'No seed test results were supplied.');
end

task_family = char(task_family);
first = seed_results(1);
[task_id, trained_model_backend] = publication_task_id_from_model_file(first.model_file);
test_backend = get_result_field(first, 'test_backend', get_result_field(test_result, 'test_backend', 'unknown'));

analysis = struct();
analysis.schema_version = 2;
created_at_datenum = now;
created_at_ms = floor(rem(created_at_datenum * 86400000, 1000));
created_at_file_stamp = sprintf('%s_%03d', datestr(created_at_datenum, 'yyyymmdd_HHMMSS'), created_at_ms);
analysis.created_at = datestr(created_at_datenum, 31);
analysis.created_at_file_stamp = created_at_file_stamp;
analysis.created_at_datenum = created_at_datenum;
analysis.created_at_posix = (created_at_datenum - datenum(1970, 1, 1)) * 86400;
analysis.task_id = task_id;
analysis.task_family = task_family;
analysis.domain = get_result_field(first, 'domain', task_family);
analysis.trained_model_backend = trained_model_backend;
analysis.test_backend = char(test_backend);
analysis.seed_list = seed_list(:).';
analysis.n_seeds_saved = numel(seed_results);
analysis.expected_seed_count_for_publication_panel6 = 3;
analysis.source = struct();
analysis.source.function = 'save_publication_test_analysis';
analysis.source.output_file_policy = 'timestamped saved MAT file for this task/backend; plotting loads the newest matching file';
analysis.source.repository_root = project_root();
analysis.source.test_result_summary = rmfield_if_present(test_result, 'seed_results');
analysis.seeds = repmat(empty_publication_seed(), 1, numel(seed_results));

for ii = 1:numel(seed_results)
    R = seed_results(ii);
    train_result = load_training_result(R.model_file);
    [P, opts_eval, data_static] = publication_model_and_options(R, train_result, task_family);
    seed_entry = empty_publication_seed();
    seed_entry.seed_index = get_result_field(R, 'seed_index', ii);
    if numel(seed_list) >= ii
        seed_entry.init_seed = seed_list(ii);
    else
        seed_entry.init_seed = get_result_field(R, 'init_seed', ii);
    end
    seed_entry.model_file = char(R.model_file);
    seed_entry.model_sha256 = publication_file_sha256_safe(R.model_file);
    seed_entry.train_backend = get_result_field(R, 'train_backend', get_result_field(train_result, 'backend', 'unknown'));
    seed_entry.test_backend = get_result_field(R, 'test_backend', analysis.test_backend);
    seed_entry.options = opts_eval;
    seed_entry.architecture = publication_architecture_summary(P, opts_eval);
    seed_entry.model = P;
    seed_entry.bias = double(P.B(:));
    seed_entry.test = R.test;
    seed_entry.metrics = publication_metrics_from_test(R.test, task_family);
    if strcmp(task_family, 'dynamical_systems')
        seed_entry.neural_activity = publication_dynamics_rate_summary(P, opts_eval);
        if logical(get_opt(opts, 'save_publication_spike_events', true))
            % GPU evaluation does not expose per-neuron spike events.  Replay
            % once on CPU while the test is being exported so later timing
            % analyses can be performed entirely from this MAT file.
            seed_entry.spike_events = publication_dynamics_spike_events(P, opts_eval);
        end
    else
        seed_entry.neural_activity = publication_static_rate_summary(P, data_static, opts_eval, batch_size);
        seed_entry.data_summary = summarize_static_data(data_static);
    end
    analysis.seeds(ii) = seed_entry;
end

analysis.output_file = fullfile(output_dir, sprintf('%s_%s_test_analysis_%s.mat', ...
    task_id, analysis.test_backend, created_at_file_stamp));
save_publication_mat_file(analysis.output_file, analysis);
fprintf('Saved publication analysis data: %s%s', analysis.output_file, newline);
end

function save_publication_mat_file(output_file, analysis)
% Save under an unrelated temporary name in the destination directory, then
% rename it into place. Keeping both paths on the same volume avoids a
% cross-volume copy that can expose a partially copied v7.3/HDF5 file to a
% concurrent reader or OneDrive sync client.
output_dir = fileparts(output_file);
tmp_file = [tempname(output_dir) '.mat'];
try
    save(tmp_file, 'analysis', '-v7.3');
    if exist(output_file, 'file') == 2
        delete(output_file);
    end
    [ok, msg] = movefile(tmp_file, output_file, 'f');
    if ~ok
        error('save_publication_test_analysis:moveFile', ...
            'Could not move temporary MAT file into place: %s', msg);
    end
catch ME
    if exist(tmp_file, 'file') == 2
        delete(tmp_file);
    end
    rethrow(ME);
end
end

function [seed_results, seed_list] = unpack_publication_seed_results(test_result)
if isfield(test_result, 'seed_results')
    seed_results = test_result.seed_results;
    seed_list = double(test_result.seed_list(:));
else
    seed_results = test_result;
    seed_list = double(get_result_field(test_result, 'init_seed', 1));
end
end

function seed_entry = empty_publication_seed()
seed_entry = struct();
seed_entry.seed_index = [];
seed_entry.init_seed = [];
seed_entry.model_file = '';
seed_entry.model_sha256 = '';
seed_entry.train_backend = '';
seed_entry.test_backend = '';
seed_entry.options = struct();
seed_entry.architecture = struct();
seed_entry.model = struct();
seed_entry.bias = [];
seed_entry.test = struct();
seed_entry.metrics = struct();
seed_entry.neural_activity = struct();
seed_entry.spike_events = empty_publication_spike_events();
seed_entry.data_summary = struct();
end

function [P, opts_eval, data_static] = publication_model_and_options(R, train_result, task_family)
if isfield(R, 'options') && isstruct(R.options)
    opts_eval = merge_options_with_seed(train_result.options, R.options);
else
    opts_eval = train_result.options;
end
opts_eval = apply_saved_architecture_metadata(opts_eval, train_result);
data_static = [];
if strcmp(task_family, 'dynamical_systems')
    if isfield(R, 'options') && isstruct(R.options) && isfield(R.options, 'T_sim')
        opts_eval.T_sim = single(R.options.T_sim);
    else
        opts_eval.T_sim = single(get_opt(opts_eval, 'closed_loop_test_time', get_opt(opts_eval, 'T_sim', single(50))));
    end
    opts_eval.closed_loop_warmup_time = single(get_opt(opts_eval, 'closed_loop_test_warmup_time', ...
        get_opt(opts_eval, 'closed_loop_warmup_time', 5)));
    opts_eval.closed_loop_validation_ics = get_opt(opts_eval, 'closed_loop_test_ics', get_opt(opts_eval, 'closed_loop_validation_ics', 1));
    opts_eval.closed_loop_ic_seed = get_opt(opts_eval, 'closed_loop_test_ic_seed', ...
        get_opt(opts_eval, 'closed_loop_ic_seed', 123));
    if isfield(train_result, 'dynamics') && isstruct(train_result.dynamics)
        opts_eval = apply_saved_dynamics_metadata(opts_eval, train_result.dynamics);
    end
    if isfield(train_result, 'model') && isstruct(train_result.model)
        P = train_result.model;
    else
        sys = make_dynamics_system_for_api(opts_eval.system_name);
        P = make_primary_model(sys.dim, sys.dim, opts_eval);
    end
    opts_eval.closed_loop_ic_include_reference = logical(get_opt(opts_eval, ...
        'closed_loop_test_include_reference', false));
    opts_eval.closed_loop_ic_role = 'test';
    if isfield(opts_eval, 'closed_loop_x0_list') && ...
            ~(isfield(R, 'options') && isfield(R.options, 'closed_loop_x0_list'))
        opts_eval = rmfield(opts_eval, 'closed_loop_x0_list');
    end
else
    data_static = load_static_data(string(task_family), opts_eval);
    if isfield(train_result, 'model') && isstruct(train_result.model)
        P = train_result.model;
    else
        P = make_primary_model(size(data_static.X_train, 1), size(data_static.Y_train, 1), opts_eval);
    end
end

P = ensure_model_architecture_fields(P, opts_eval);
P.B = best_bias_from_result(train_result);
end

function events = empty_publication_spike_events()
events = struct();
events.status = 'not_recorded';
events.message = '';
events.schema_version = 2;
events.recording_backend = '';
events.baseline_backend = '';
events.truth_backend = '';
events.dt = [];
events.n_hidden = [];
events.n_output = [];
events.closed_loop_warmup_steps = [];
events.n_initial_conditions = [];
events.events_by_ic = struct('event_steps', {}, 'event_neurons', {}, ...
    'event_rho', {}, 'n_steps', {}, 'n_events', {}, 'baseline_norm', {}, ...
    'true_norm', {}, 'test_x0_norm', {}, 'truth_diagnostic', {});
events.calculation = struct();
end

function [task_id, backend] = publication_task_id_from_model_file(model_file)
[~, name] = fileparts(model_file);
tok_spsa = regexp(name, '^(?<task>.+)_lowrank_SPSA_GPU_primary_seed\d+$', 'names', 'ignorecase');
if ~isempty(tok_spsa)
    task_id = [tok_spsa.task '_spsa'];
    backend = 'spsa_gpu';
    return;
end
tok = regexp(name, '^(?<task>.+)_(?<backend>cpu|gpu)_primary_seed\d+$', 'names');
if isempty(tok)
    task_id = name;
    backend = 'unknown';
else
    task_id = tok.task;
    backend = tok.backend;
end
end

function arch = publication_architecture_summary(P, opts)
arch = struct();
arch.N_hidden = get_field_or(P, 'N_hidden', get_opt(opts, 'N_hidden', NaN));
arch.N_in = get_field_or(P, 'N_in', NaN);
arch.N_out = get_field_or(P, 'N_out', NaN);
arch.recurrent_mode = get_field_or(P, 'recurrent_mode', ...
    get_opt(get_opt(opts, 'arch', struct()), 'recurrent_mode', ''));
arch.low_rank_rank = get_field_or(P, 'N_rec', get_opt(opts, 'N_rec', NaN));
arch.dt = get_opt(opts, 'dt', NaN);
arch.steps_present = get_opt(opts, 'steps_present', NaN);
end

function metrics = publication_metrics_from_test(test, task_family)
metrics = struct();
for name = ["loss", "metric", "accuracy", "mse", "rmse", "mae", "r2", ...
        "wasserstein_distance", "wasserstein_distance_by_ic"]
    key = char(name);
    if isfield(test, key)
        metrics.(key) = test.(key);
    end
end
if isfield(test, 'regression')
    metrics.regression = test.regression;
end
if isfield(test, 'closed_loop')
    metrics.closed_loop = test.closed_loop;
end
metrics.task_family = task_family;
end

function value = get_field_or(S, field_name, fallback)
if isstruct(S) && isfield(S, field_name)
    value = S.(field_name);
else
    value = fallback;
end
end

function hash = publication_file_sha256_safe(file_name)
try
    hash = file_sha256(file_name);
catch
    hash = '';
end
end
