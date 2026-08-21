function comparison = run_architecture_performance_comparison(task_kind, domain, backend, opts, output_file)
%RUN_ARCHITECTURE_PERFORMANCE_COMPARISON Train the three architecture modes.
%   task_kind is "static" or "dynamics". Static tasks use domain
%   "classification" or "regression"; dynamics uses domain "dynamical_systems".
if nargin < 5
    output_file = "";
end
task_kind = lower(string(task_kind));
domain = lower(string(domain));
backend = lower(string(backend));
decoder_distribution_mode = architecture_decoder_distribution_mode(opts);
base_arch = getfield_with_default(opts, 'arch', default_arch_options());
modes = architecture_comparison_modes(decoder_distribution_mode, base_arch);
results = repmat(struct('name', "", 'arch', struct(), 'result', struct(), ...
    'elapsed_seconds', NaN), numel(modes), 1);

for ii = 1:numel(modes)
    one_opts = opts;
    one_opts.arch = modes(ii).arch;
    one_opts.arch_label = char(modes(ii).name);
    t0 = tic;
    switch task_kind
        case "static"
            result = snn_primary_api('train_static', char(domain), char(backend), one_opts);
        case "dynamics"
            result = snn_primary_api('train_dynamics', 'dynamical_systems', char(backend), one_opts);
        otherwise
            error('run_architecture_performance_comparison:taskKind', ...
                'task_kind must be "static" or "dynamics".');
    end
    results(ii).name = modes(ii).name;
    results(ii).arch = modes(ii).arch;
    results(ii).result = result;
    results(ii).elapsed_seconds = toc(t0);
    results(ii).activity = architecture_activity_diagnostics(result, task_kind, domain, one_opts);
end

comparison = struct();
comparison.task_kind = char(task_kind);
comparison.domain = char(domain);
comparison.backend = char(backend);
comparison.options = opts;
comparison.decoder_distribution_mode = decoder_distribution_mode;
comparison.modes = modes;
comparison.results = results;
comparison.summary = architecture_result_summary(results);

if strlength(string(output_file)) > 0
    output_dir = fileparts(char(output_file));
    if ~isempty(output_dir) && exist(output_dir, 'dir') ~= 7
        mkdir(output_dir);
    end
    save(char(output_file), 'comparison', '-v7.3');
end

function mode = architecture_decoder_distribution_mode(opts)
mode = "both";
if isstruct(opts) && isfield(opts, 'architecture_decoder_distributions')
    mode = string(opts.architecture_decoder_distributions);
elseif isstruct(opts) && isfield(opts, 'arch') && isstruct(opts.arch) && ...
        isfield(opts.arch, 'signed_decoder_distribution_comparison')
    mode = string(opts.arch.signed_decoder_distribution_comparison);
end
mode = lower(mode);
if mode == "normal"
    mode = "gaussian";
elseif mode == "all"
    mode = "both";
end
end
end

function value = getfield_with_default(s, key, default_value)
if isstruct(s) && isfield(s, key)
    value = s.(key);
else
    value = default_value;
end
end

function activity = architecture_activity_diagnostics(result, task_kind, domain, opts)
activity = struct();
if ~(isstruct(opts) && isfield(opts, 'architecture_activity_diagnostics') && opts.architecture_activity_diagnostics)
    activity.enabled = false;
    return;
end
activity.enabled = true;
activity.source = 'cpu_diagnostic_replay';
try
    P = result.model;
    P.B = best_bias_from_result(result);
    max_samples = max(1, round(getfield_with_default(opts, 'architecture_activity_max_samples', 4)));
    switch string(task_kind)
        case "static"
            data = load_static_data(domain, opts);
            X = data.X_test(:, 1:min(max_samples, size(data.X_test, 2)));
            [S, ~, ~, Irec, ~] = static_spike_diagnostics_cpu(P, X, opts);
        case "dynamics"
            opts_diag = opts;
            opts_diag.dynamics_split = 'train';
            [x, ~] = make_dynamics_problem(opts_diag);
            if isstruct(x) && isfield(x, 'pool')
                x = x.pool(:, 1:x.steps);
            elseif iscell(x)
                x = x{1};
            end
            lambda = make_lambda_sequence(size(x, 2), opts_diag);
            [S, ~, ~, Irec, ~] = dynamics_spike_diagnostics_cpu(P, x, lambda, opts_diag);
        otherwise
            error('run_architecture_performance_comparison:activityTaskKind', ...
                'Unknown task kind for activity diagnostics.');
    end
    activity = summarise_activity_arrays(S, Irec, opts.dt);
    activity.enabled = true;
    activity.source = 'cpu_diagnostic_replay';
catch ME
    activity.error = string(ME.message);
    activity.mean_firing_rate = NaN;
    activity.silent_neuron_fraction = NaN;
    activity.mean_spike_count = NaN;
    activity.spike_count_std = NaN;
    activity.mean_recurrent_current = NaN;
    activity.std_recurrent_current = NaN;
    activity.mean_abs_recurrent_current = NaN;
end
end

function activity = summarise_activity_arrays(S, Irec, dt)
N = size(S, 1);
S2 = reshape(single(S), N, []);
spike_count = sum(S2, 2);
duration_seconds = max(single(dt), single(eps)) * single(size(S2, 2));
I = single(Irec(:));
activity = struct();
activity.mean_firing_rate = single(mean(spike_count ./ duration_seconds));
activity.silent_neuron_fraction = single(mean(spike_count == 0));
activity.mean_spike_count = single(mean(spike_count));
activity.spike_count_std = single(std(spike_count, 0));
activity.mean_recurrent_current = single(mean(I));
activity.std_recurrent_current = single(std(I, 0));
activity.mean_abs_recurrent_current = single(mean(abs(I)));
end

function summary = architecture_result_summary(results)
name = strings(numel(results), 1);
elapsed_seconds = nan(numel(results), 1);
best_loss = nan(numel(results), 1);
test_metric = nan(numel(results), 1);
test_loss = nan(numel(results), 1);
for ii = 1:numel(results)
    name(ii) = results(ii).name;
    elapsed_seconds(ii) = results(ii).elapsed_seconds;
    R = results(ii).result;
    if isfield(R, 'best') && isfield(R.best, 'loss')
        best_loss(ii) = double(R.best.loss);
    end
    if isfield(R, 'test')
        if isfield(R.test, 'metric'), test_metric(ii) = double(R.test.metric); end
        if isfield(R.test, 'loss'), test_loss(ii) = double(R.test.loss); end
        if isfield(R.test, 'wasserstein_distance')
            test_metric(ii) = double(R.test.wasserstein_distance);
        end
    end
end
summary = table(name, elapsed_seconds, best_loss, test_loss, test_metric, ...
    'VariableNames', {'Architecture','ElapsedSeconds','BestLoss','TestLoss','TestMetric'});
end
