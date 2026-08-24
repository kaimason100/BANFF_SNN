function outputFile = banff_publication(results)
%BANFF_PUBLICATION Save compact, figure-facing tested results.
%   OUTPUTFILE = BANFF_PUBLICATION(RESULTS) exports tested seed results for
%   one scientifically identical configuration (apart from network seed).
%   Fixed matrices are not duplicated: they are deterministic and can be
%   regenerated from each seed's saved configuration and seed.

if isempty(results) || ~all(arrayfun(@(x) isfield(x, 'test'), results))
    error('banff:publicationResults', 'Every supplied result must have been tested.');
end
if ~all(arrayfun(@(x) isfield(x, 'provenance') && ...
        isfield(x.provenance, 'scientific_config_sha256'), results))
    error('banff:publicationProvenance', ...
        'Results must contain the publication-ready scientific config fingerprint.');
end
fingerprints = arrayfun(@(x) string(x.provenance.scientific_config_sha256), results);
if any(fingerprints ~= fingerprints(1))
    error('banff:publicationMixedResults', ...
        'All publication seeds must share one scientific configuration.');
end

first = results(1).config;
[taskId, family, backend] = publication_identity(first);
analysis = struct();
analysis.schema_version = 5;
createdNow = now;
analysis.created_at = datestr(createdNow, 31);
analysis.created_at_datenum = createdNow;
analysis.created_at_posix = (createdNow - datenum(1970, 1, 1)) * 86400;
analysis.task_id = taskId;
analysis.task_family = family;
analysis.domain = char(first.kind);
analysis.trained_model_backend = backend;
analysis.test_backend = backend;
analysis.seed_list = arrayfun(@(x) double(x.config.seed), results);
analysis.n_seeds_saved = numel(results);
analysis.expected_seed_count_for_publication_panel6 = 3;
analysis.scientific_config_sha256 = char(fingerprints(1));
analysis.source = struct('function', 'banff_publication', ...
    'simulator', 'MATLAB gpuArray with fused arrayfun kernel', ...
    'trainable_parameter', 'hidden bias B only', ...
    'eligibility', results(1).training, ...
    'core_source_sha256', results(1).provenance.core_source_sha256);

seeds = repmat(empty_seed(), 1, numel(results));
for index = 1:numel(results)
    R = results(index);
    cfg = R.config;
    seed = empty_seed();
    seed.seed_index = index;
    seed.init_seed = double(cfg.seed);
    seed.model_file = model_basename(cfg.model_file);
    seed.train_backend = backend;
    seed.test_backend = backend;
    seed.options = publication_options(cfg, R.data_information);
    seed.architecture = struct('N_hidden', cfg.N_hidden, ...
        'N_in', input_dimension(R), 'N_out', output_dimension(R), ...
        'recurrent_mode', char(cfg.recurrent_mode), ...
        'low_rank_rank', cfg.N_recurrent, ...
        'recurrent_storage', char(recurrent_storage(cfg)));
    seed.bias = single(R.best.B(:));
    seed.scientific_config_sha256 = R.provenance.scientific_config_sha256;
    seed.code_provenance = R.provenance;
    [seed.test, seed.metrics] = publication_test(R);
    if cfg.kind == "dynamics"
        [seed.neural_activity, seed.spike_events] = ...
            dynamics_activity(R.test, cfg, cfg.N_hidden);
    else
        seed.neural_activity = R.test.neural_activity;
        seed.spike_events = empty_spike_events();
        seed.data_summary = R.data_information;
    end
    seeds(index) = seed;
end
analysis.seeds = seeds;

folder = fullfile(fileparts(mfilename('fullpath')), 'outputs', 'publication_analysis');
% Keep the historical publication filename convention so the original figure
% scripts work unchanged. Optional continuous-surrogate runs are placed in a
% separate ablation folder and therefore cannot be selected accidentally by
% the manuscript figure loaders.
if first.method == "eprop" && first.eligibility_mode ~= "hard_spike"
    shortHash = char(fingerprints(1));
    shortHash = shortHash(1:min(12, numel(shortHash)));
    folder = fullfile(folder, 'ablations', ...
        sprintf('%s_cfg%s', char(first.eligibility_mode), shortHash));
end
if exist(folder, 'dir') ~= 7
    mkdir(folder);
end
stamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss_SSS'));
outputFile = fullfile(folder, sprintf('%s_%s_test_analysis_%s.mat', ...
    taskId, backend, stamp));
save(outputFile, 'analysis', '-v7.3');
fprintf('Saved publication analysis: %s\n', outputFile);
end

function seed = empty_seed()
seed = struct('seed_index', [], 'init_seed', [], 'model_file', '', ...
    'train_backend', '', 'test_backend', '', 'options', struct(), ...
    'architecture', struct(), 'bias', [], 'scientific_config_sha256', '', ...
    'code_provenance', struct(), 'test', struct(), 'metrics', struct(), ...
    'neural_activity', struct(), 'spike_events', empty_spike_events(), ...
    'data_summary', struct());
end

function [taskId, family, backend] = publication_identity(cfg)
switch cfg.task
    case "breast_cancer", taskId = 'classification_BC'; family = 'classification';
    case "mnist", taskId = 'classification_MNIST'; family = 'classification';
    case "afro_mnist_vai", taskId = 'classification_AfroMNIST_Vai'; family = 'classification';
    case "abalone", taskId = 'regression_abalone'; family = 'regression';
    case "toyota", taskId = 'regression_toyota'; family = 'regression';
    case "yacht", taskId = 'regression_yacht'; family = 'regression';
    case "lorenz", taskId = 'dynamical_systems_lorenz'; family = 'dynamical_systems';
    case "sprott_s", taskId = 'dynamical_systems_sprotts'; family = 'dynamical_systems';
    case "vanderpol", taskId = 'dynamical_systems_vanderpol'; family = 'dynamical_systems';
    otherwise, error('banff:publicationTask', 'Unknown task "%s".', cfg.task);
end
if cfg.recurrent_mode == "full_rank"
    if cfg.N_hidden == 6000
        taskId = sprintf('%s_full_rank6k', taskId); % preserve manuscript figure ID
    else
        taskId = sprintf('%s_full_rank_N%d', taskId, cfg.N_hidden);
    end
elseif cfg.method == "eprop" && cfg.training_profile == "neuron_sweep"
    taskId = sprintf('%s_neuron_sweep_N%d', taskId, cfg.N_hidden);
elseif cfg.method == "eprop" && cfg.N_hidden ~= 32000
    % Nonstandard-size ad hoc runs remain identifiable without pretending they
    % are part of the publication neuron sweep.
    taskId = sprintf('%s_N%d', taskId, cfg.N_hidden);
end
if cfg.method == "spsa"
    taskId = sprintf('%s_spsa', taskId);
    backend = 'spsa_gpu';
else
    backend = 'gpu';
end
end

function storage = recurrent_storage(cfg)
if cfg.recurrent_mode == "full_rank"
    storage = "sparse_double";
else
    storage = "factorized";
end
end

function name = model_basename(file)
[~, stem, extension] = fileparts(char(file));
name = [stem extension];
end

function options = publication_options(cfg, information)
options = cfg;
for field = {'checkpoint_hours','output_directory','model_file','verbose_every'}
    if isfield(options, field{1}), options = rmfield(options, field{1}); end
end
options.task_tag = char(cfg.task);
options.N_rec = cfg.N_recurrent;
options.steps_present = cfg.presentation_steps;
options.steps_avg = cfg.average_steps;
options.idx_train = field_or(information, 'train_index', []);
options.idx_val = field_or(information, 'validation_index', []);
options.idx_test = field_or(information, 'test_index', []);
options.feature_mean = field_or(information, 'feature_mean', []);
options.feature_std = field_or(information, 'feature_std', []);
options.target_mean = field_or(information, 'target_mean', []);
options.target_std = field_or(information, 'target_std', []);
options.dataset_sha256 = field_or(information, 'dataset_sha256', '');
options.split_policy = field_or(information, 'split_policy', '');
options.arch = struct('recurrent_mode', char(cfg.recurrent_mode));
if cfg.kind == "dynamics"
    if cfg.task == "sprott_s"
        options.system_name = 'sprotts';
    else
        options.system_name = char(cfg.task);
    end
    options.T_sim = cfg.test_time;
    options.long_sim_time = cfg.long_simulation_time;
    options.burn_in_time = cfg.burn_in_time;
    options.dyn_sys_rate = cfg.system_rate;
    options.dynamics_mu = information.mean;
    options.dynamics_sigma = information.std;
    options.closed_loop_test_time = cfg.test_time;
    options.closed_loop_test_warmup_time = cfg.test_warmup_time;
    options.closed_loop_warmup_time = cfg.test_warmup_time;
    options.closed_loop_test_ics = cfg.test_initial_conditions;
    options.closed_loop_test_ic_seed = cfg.test_initial_condition_seed;
    options.closed_loop_test_include_reference = false;
    options.closed_loop_ic_jitter = cfg.initial_condition_jitter;
    options.wd = struct('NumProjections', cfg.phase_metric.projections, ...
        'TrimFraction', cfg.phase_metric.trim_fraction, ...
        'Subsample', cfg.phase_metric.subsample, ...
        'TransientFraction', cfg.phase_metric.transient_fraction, ...
        'MaxPoints', cfg.phase_metric.max_points);
end
end

function [test, metrics] = publication_test(R)
if R.config.kind == "dynamics"
    test = struct('pred_norm_by_ic', {R.test.prediction}, ...
        'true_norm_by_ic', {R.test.truth}, ...
        'wasserstein_distance', R.test.phase_distance, ...
        'wasserstein_distance_by_ic', R.test.phase_distance_by_initial_condition, ...
        'metric', R.test.phase_distance);
    metrics = struct('phase_distance', R.test.phase_distance, ...
        'phase_distance_by_initial_condition', ...
        R.test.phase_distance_by_initial_condition);
elseif R.config.kind == "classification"
    test = struct('Z', R.test.output, ...
        'metric', R.test.statistics.accuracy_percent, ...
        'loss', R.test.loss, 'statistics', R.test.statistics);
    metrics = struct('metric', R.test.statistics.accuracy_percent, ...
        'accuracy_percent', R.test.statistics.accuracy_percent);
else
    regression = R.test.statistics;
    test = struct('Z', R.test.output, 'metric', regression.pearson_r, ...
        'loss', R.test.loss, 'regression', regression);
    metrics = struct('metric', regression.pearson_r, 'regression', regression);
end
end

function [neural, spikeEvents] = dynamics_activity(test, cfg, nHidden)
nConditions = numel(test.events);
counts = zeros(nHidden, 1);
postWarmupSteps = round(cfg.test_time / cfg.dt);
warmupSteps = round(cfg.test_warmup_time / cfg.dt);
entries = repmat(empty_event_entry(), nConditions, 1);
for index = 1:nConditions
    source = test.events{index};
    neuron = double(source.neuron(:));
    step = double(source.step(:));
    keep = step > warmupSteps;
    if any(keep)
        counts = counts + accumarray(neuron(keep), 1, [nHidden 1], @sum, 0);
    end
    entry = empty_event_entry();
    entry.event_steps = uint32(step);
    entry.event_neurons = uint32(neuron);
    entry.event_rho = single(source.rho(:));
    entry.n_steps = uint32(warmupSteps + postWarmupSteps);
    entry.n_events = uint64(numel(neuron));
    entry.baseline_norm = single(test.prediction{index});
    entry.true_norm = single(test.truth{index});
    entries(index) = entry;
end
rates = counts ./ max(1, nConditions * postWarmupSteps * double(cfg.dt));
active = rates > 0;
neural = struct('mean_firing_rate_by_neuron_hz', single(rates), ...
    'active_neuron_mask', active, 'active_fraction', double(mean(active)), ...
    'active_fraction_percent', 100 * double(mean(active)), ...
    'spike_count_by_neuron_post_warmup', single(counts), ...
    'calculation', struct('context', 'full closed-loop test after warmup', ...
    'rate_units', 'Hz'));
spikeEvents = empty_spike_events();
spikeEvents.status = 'ok';
spikeEvents.recording_backend = 'matlab_gpu_arrayfun';
spikeEvents.baseline_backend = 'matlab_gpu_arrayfun';
spikeEvents.truth_backend = 'Euler true continuation from network warmup state';
spikeEvents.dt = double(cfg.dt);
spikeEvents.n_hidden = nHidden;
spikeEvents.n_output = size(test.prediction{1}, 2);
spikeEvents.closed_loop_warmup_steps = warmupSteps;
spikeEvents.n_initial_conditions = nConditions;
spikeEvents.events_by_ic = entries;
end

function events = empty_spike_events()
events = struct('status', 'not_recorded', 'message', '', 'schema_version', 5, ...
    'recording_backend', '', 'baseline_backend', '', 'truth_backend', '', ...
    'dt', [], 'n_hidden', [], 'n_output', [], ...
    'closed_loop_warmup_steps', [], 'n_initial_conditions', [], ...
    'events_by_ic', repmat(empty_event_entry(), 0, 1), 'calculation', struct());
end

function entry = empty_event_entry()
entry = struct('event_steps', zeros(0, 1, 'uint32'), ...
    'event_neurons', zeros(0, 1, 'uint32'), ...
    'event_rho', zeros(0, 1, 'single'), 'n_steps', uint32(0), ...
    'n_events', uint64(0), 'baseline_norm', zeros(0, 0, 'single'), ...
    'true_norm', zeros(0, 0, 'single'), ...
    'test_x0_norm', zeros(0, 1, 'single'), 'truth_diagnostic', struct());
end

function dimension = input_dimension(result)
if result.config.kind == "dynamics"
    dimension = numel(result.data_information.mean);
else
    dimension = numel(result.data_information.feature_mean);
end
end

function dimension = output_dimension(result)
if result.config.kind == "dynamics"
    dimension = numel(result.data_information.mean);
elseif result.config.kind == "classification"
    dimension = size(result.test.output, 1);
else
    dimension = numel(result.data_information.target_mean);
end
end

function value = field_or(S, name, defaultValue)
if isstruct(S) && isfield(S, name) && ~isempty(S.(name))
    value = S.(name);
else
    value = defaultValue;
end
end
