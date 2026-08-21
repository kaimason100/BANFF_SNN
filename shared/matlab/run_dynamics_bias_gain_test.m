function result = run_dynamics_bias_gain_test(opts)
%RUN_DYNAMICS_BIAS_GAIN_TEST Test learned-bias gains with the standard GPU evaluator.
%   Each condition changes only an in-memory copy of the saved best bias:
%       B_test = bias_gain * B_learned
%   The normal dynamical-system test path then performs the closed-loop
%   network warmup and scored network/true-system continuation.

if nargin < 1 || ~isstruct(opts)
    error('run_dynamics_bias_gain_test:options', 'Provide one options struct.');
end

required = {'task_ids', 'task_labels', 'test_seeds', 'bias_gains'};
for ii = 1:numel(required)
    if ~isfield(opts, required{ii}) || isempty(opts.(required{ii}))
        error('run_dynamics_bias_gain_test:missingOption', ...
            'opts.%s is required and cannot be empty.', required{ii});
    end
end

repo_root = get_option(opts, 'repo_root', project_root());
task_ids = cellstr(string(opts.task_ids));
task_labels = cellstr(string(opts.task_labels));
test_seeds = double(opts.test_seeds(:).');
bias_gains = double(opts.bias_gains(:).');
trained_model_backend = char(get_option(opts, 'trained_model_backend', 'auto'));

if numel(task_ids) ~= numel(task_labels)
    error('run_dynamics_bias_gain_test:taskLabels', ...
        'task_ids and task_labels must contain the same number of entries.');
end
if any(~isfinite(test_seeds)) || any(test_seeds < 0) || any(test_seeds ~= round(test_seeds))
    error('run_dynamics_bias_gain_test:testSeeds', ...
        'test_seeds must contain finite nonnegative integers.');
end
if numel(unique(test_seeds)) ~= numel(test_seeds)
    error('run_dynamics_bias_gain_test:testSeeds', 'test_seeds must not contain duplicates.');
end
if any(~isfinite(bias_gains))
    error('run_dynamics_bias_gain_test:biasGains', 'bias_gains must contain only finite values.');
end
if numel(unique(bias_gains)) ~= numel(bias_gains)
    error('run_dynamics_bias_gain_test:biasGains', 'bias_gains must not contain duplicates.');
end
if gpuDeviceCount() < 1
    error('run_dynamics_bias_gain_test:noGPU', ...
        'This test requires a supported GPU, but MATLAB did not find one.');
end

test_overrides = struct();
test_overrides.closed_loop_test_time = single(get_option(opts, 'closed_loop_test_time', 50));
test_overrides.closed_loop_test_warmup_time = single(get_option(opts, 'closed_loop_test_warmup_time', 5));
test_overrides.closed_loop_test_ics = double(get_option(opts, 'closed_loop_test_ics', 5));
test_overrides.closed_loop_test_ic_seed = double(get_option(opts, 'closed_loop_test_ic_seed', 123));
validate_test_overrides(test_overrides);

gain_template = struct('gain', [], 'scaled_bias_norm', [], 'test', struct(), 'options', struct());
seed_template = struct('init_seed', [], 'model_file', '', 'learned_bias', [], ...
    'learned_bias_norm', [], 'gain_results', repmat(gain_template, 0, 1));
system_template = struct('task_id', '', 'label', '', 'trained_model_backend', '', ...
    'seed_results', repmat(seed_template, 0, 1));
systems = repmat(system_template, numel(task_ids), 1);

for tt = 1:numel(task_ids)
    model_stem = task_ids{tt};
    [model_files, ~, resolved_backend] = snn_resolve_seed_model_files( ...
        repo_root, model_stem, 'gpu', test_seeds, 1, trained_model_backend);

    seed_results = repmat(seed_template, numel(test_seeds), 1);
    for ss = 1:numel(test_seeds)
        fprintf('%s, network seed %d (%d/%d)\n', ...
            task_labels{tt}, test_seeds(ss), ss, numel(test_seeds));
        train_result = load_training_result(model_files{ss});
        learned_bias = best_bias_from_result(train_result);

        gain_results = repmat(gain_template, numel(bias_gains), 1);
        for gg = 1:numel(bias_gains)
            fprintf('  bias gain %.6g (%d/%d)\n', bias_gains(gg), gg, numel(bias_gains));
            gain_train_result = train_result;
            if ~isfield(gain_train_result, 'best') || ~isstruct(gain_train_result.best)
                gain_train_result.best = struct();
            end
            gain_train_result.best.B = cast(bias_gains(gg), 'like', learned_bias) .* learned_bias;

            one_opts = test_overrides;
            one_opts.model_file = model_files{ss};
            one = test_dynamics_from_result("gpu", gain_train_result, one_opts);

            gain_results(gg).gain = bias_gains(gg);
            gain_results(gg).scaled_bias_norm = double(gather_if_needed(norm(gain_train_result.best.B(:))));
            gain_results(gg).test = one.test;
            gain_results(gg).options = one.options;
        end

        seed_results(ss).init_seed = test_seeds(ss);
        seed_results(ss).model_file = model_files{ss};
        seed_results(ss).learned_bias = learned_bias;
        seed_results(ss).learned_bias_norm = double(gather_if_needed(norm(learned_bias(:))));
        seed_results(ss).gain_results = gain_results;
    end

    systems(tt).task_id = model_stem;
    systems(tt).label = task_labels{tt};
    systems(tt).trained_model_backend = resolved_backend;
    systems(tt).seed_results = seed_results;
end

result = struct();
result.analysis_kind = 'dynamical_systems_learned_bias_gain_sweep';
result.created_at = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss Z'));
result.test_backend = 'gpu';
result.test_seeds = test_seeds;
result.bias_gains = bias_gains;
result.test_options = test_overrides;
result.systems = systems;
result.metric_table = make_metric_table(systems);
end

function validate_test_overrides(opts)
if ~isscalar(opts.closed_loop_test_time) || ~isfinite(opts.closed_loop_test_time) || opts.closed_loop_test_time <= 0
    error('run_dynamics_bias_gain_test:testTime', 'closed_loop_test_time must be positive and finite.');
end
if ~isscalar(opts.closed_loop_test_warmup_time) || ~isfinite(opts.closed_loop_test_warmup_time) || opts.closed_loop_test_warmup_time < 0
    error('run_dynamics_bias_gain_test:warmupTime', 'closed_loop_test_warmup_time must be nonnegative and finite.');
end
if ~isscalar(opts.closed_loop_test_ics) || opts.closed_loop_test_ics < 1 || ...
        opts.closed_loop_test_ics ~= round(opts.closed_loop_test_ics)
    error('run_dynamics_bias_gain_test:testICs', 'closed_loop_test_ics must be a positive integer.');
end
if ~isscalar(opts.closed_loop_test_ic_seed) || ~isfinite(opts.closed_loop_test_ic_seed) || ...
        opts.closed_loop_test_ic_seed ~= round(opts.closed_loop_test_ic_seed)
    error('run_dynamics_bias_gain_test:icSeed', 'closed_loop_test_ic_seed must be a finite integer.');
end
end

function value = get_option(opts, name, default_value)
if isfield(opts, name) && ~isempty(opts.(name))
    value = opts.(name);
else
    value = default_value;
end
end

function x = gather_if_needed(x)
if isa(x, 'gpuArray')
    x = gather(x);
end
end

function T = make_metric_table(systems)
task = strings(0, 1);
network_seed = zeros(0, 1);
bias_gain = zeros(0, 1);
mean_phase_portrait_wd = zeros(0, 1);
failed_truth_continuations = zeros(0, 1);
for tt = 1:numel(systems)
    for ss = 1:numel(systems(tt).seed_results)
        seed_result = systems(tt).seed_results(ss);
        for gg = 1:numel(seed_result.gain_results)
            one = seed_result.gain_results(gg);
            task(end + 1, 1) = string(systems(tt).label); %#ok<AGROW>
            network_seed(end + 1, 1) = seed_result.init_seed; %#ok<AGROW>
            bias_gain(end + 1, 1) = one.gain; %#ok<AGROW>
            mean_phase_portrait_wd(end + 1, 1) = double(one.test.wasserstein_distance); %#ok<AGROW>
            if isfield(one.test, 'truth_simulation_failed_by_ic')
                failed_truth_continuations(end + 1, 1) = sum(one.test.truth_simulation_failed_by_ic); %#ok<AGROW>
            else
                failed_truth_continuations(end + 1, 1) = 0; %#ok<AGROW>
            end
        end
    end
end
T = table(task, network_seed, bias_gain, mean_phase_portrait_wd, failed_truth_continuations);
end
