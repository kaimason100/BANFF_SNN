% Package orientation: Shared MATLAB utility used across tasks. Read this with the caller open so input/output structs and saved-result fields are clear.

function snn_plot_test_result(result, output_dir)
%SNN_PLOT_TEST_RESULT Plot held-out test summaries and training losses.
%   SNN_PLOT_TEST_RESULT(RESULT) displays a compact, interpretable summary of
%   held-out task performance. If RESULT points to saved seed model files, the
%   training and validation loss curves stored in those model files are also
%   plotted. Scalar task metrics are reported as text so they are not mixed with
%   losses on a shared axis.

if nargin < 2 || isempty(output_dir)
    output_dir = '';
end
if ~isempty(output_dir) && exist(output_dir, 'dir') ~= 7
    mkdir(output_dir);
end

if isfield(result, 'seed_results')
    plot_seed_result_summary(result, output_dir);
    plot_training_validation_loss_from_model(result, output_dir);
    return;
end

if ~isfield(result, 'test')
    warning('snn_plot_test_result:noTest', 'Result has no test field.');
    return;
end

T = result.test;
fig = figure('Color', 'w');
if isfield(T, 'Z') && ~isempty(T.Z)
    tiledlayout_compat(1, 2);
    nexttile_compat();
    plot_scalar_summary(result, T);
    nexttile_compat();
    Z = double(T.Z);
    if isvector(Z)
        plot(Z(:), 'LineWidth', 1.2);
        xlabel('Sample/time index');
        ylabel('Output');
    else
        plot(Z.', 'LineWidth', 1.0);
        xlabel('Sample/time index');
        ylabel('Output dimension');
    end
    title('Primary Decoder Outputs');
    grid on;
else
    plot_scalar_summary(result, T);
end
save_plot(fig, output_dir, 'test_summary.png');
plot_training_validation_loss_from_model(result, output_dir);
end

function plot_seed_result_summary(result, output_dir)
fig = figure('Color', 'w');
axis off;
lines = {sprintf('Held-out test summary across %d initialisation seeds', numel(result.seed_results))};
print_seed_test_loss_table(result);

for ii = 1:numel(result.seed_results)
    R = result.seed_results(ii);
    seed_value = get_result_field_local(R, 'init_seed', ii);
    if isfield(R, 'test')
        T = R.test;
        if is_regression_result(result) && has_regression_summary(T)
            stats = get_regression_summary(T);
            lines{end+1} = sprintf('Seed %g: loss %s | RMSE %s | Pearson r %s | Pearson p %s', ...
                seed_value, format_scalar(get_field_or_nan(T, 'loss')), ...
                format_scalar(get_field_or_nan(stats, 'rmse')), ...
                format_scalar(get_field_or_nan(stats, 'pearson_r')), ...
                format_scalar(get_field_or_nan(stats, 'pearson_p'))); %#ok<AGROW>
        elseif is_classification_result(result)
            lines{end+1} = sprintf('Seed %g: loss %s | accuracy %s%%', ...
                seed_value, format_scalar(get_field_or_nan(T, 'loss')), ...
                format_scalar(get_task_value(T))); %#ok<AGROW>
        else
            lines{end+1} = sprintf('Seed %g: %s %s | loss %s', seed_value, ...
                test_value_label(result, T), format_scalar(get_task_value(T)), ...
                format_scalar(get_field_or_nan(T, 'loss'))); %#ok<AGROW>
        end
        truth_summary = truth_diagnostic_summary(T);
        if ~isempty(truth_summary)
            lines{end+1} = sprintf('  %s', truth_summary); %#ok<AGROW>
        end
    elseif isfield(R, 'best') && isfield(R.best, 'wd')
        lines{end+1} = sprintf('Seed %g: best closed-loop WD %s', ...
            seed_value, format_scalar(R.best.wd)); %#ok<AGROW>
    end
end

if isfield(result, 'network_seed_audit') && isstruct(result.network_seed_audit)
    lines{end+1} = ''; %#ok<AGROW>
    lines{end+1} = sprintf('Seed network audit: %s', char(result.network_seed_audit.status)); %#ok<AGROW>
end

if isfield(result, 'summary') && isfield(result.summary, 'metric_table') && ...
        istable(result.summary.metric_table) && height(result.summary.metric_table) > 0
    lines{end+1} = ''; %#ok<AGROW>
    lines{end+1} = 'Across-seed mean +/- SD:'; %#ok<AGROW>
    M = result.summary.metric_table;
    preferred = preferred_summary_metrics(result);
    for jj = 1:numel(preferred)
        row = find(strcmp(string(M.Metric), preferred(jj)), 1, 'first');
        if ~isempty(row)
            lines{end+1} = sprintf('%s: %s +/- %s (n=%d)', ...
                display_metric_name(preferred(jj)), format_scalar(M.Mean(row)), ...
                format_scalar(M.Std(row)), round(double(M.N(row)))); %#ok<AGROW>
        end
    end
end

text(0.03, 0.94, strjoin(lines, newline), 'Units', 'normalized', ...
    'VerticalAlignment', 'top', 'FontName', 'Helvetica', 'FontSize', 11);
title('Multi-Seed Test Summary');
save_plot(fig, output_dir, 'test_summary_by_seed.png');
plot_seed_distribution(result, output_dir);
plot_learned_bias_swarm(result, output_dir);
plot_static_seed_loss(result, output_dir);
end

function print_seed_test_loss_table(result)
if ~(is_classification_result(result) || is_regression_result(result))
    return;
end
if ~isfield(result, 'summary') || ~isfield(result.summary, 'seed_table') || ...
        ~istable(result.summary.seed_table) || height(result.summary.seed_table) == 0
    return;
end
T = result.summary.seed_table;
if ~any(strcmp(T.Properties.VariableNames, 'Loss'))
    return;
end
try
    if is_classification_result(result) && any(strcmp(T.Properties.VariableNames, 'AccuracyPercent'))
        disp(T(:, {'Seed', 'Loss', 'AccuracyPercent'}));
    elseif is_regression_result(result) && any(strcmp(T.Properties.VariableNames, 'RMSE'))
        disp(T(:, {'Seed', 'Loss', 'RMSE', 'PearsonR'}));
    else
        disp(T(:, {'Seed', 'Loss'}));
    end
catch
end
end

function plot_static_seed_loss(result, output_dir)
if ~(is_classification_result(result) || is_regression_result(result))
    return;
end
if ~isfield(result, 'summary') || ~isfield(result.summary, 'seed_table') || ...
        ~istable(result.summary.seed_table) || height(result.summary.seed_table) == 0
    return;
end
T = result.summary.seed_table;
if ~any(strcmp(T.Properties.VariableNames, 'Loss'))
    return;
end
values = double(T.Loss);
ok = isfinite(values);
if ~any(ok)
    return;
end
fig = figure('Color', 'w');
plot(double(T.Seed(ok)), values(ok), 'o-', 'LineWidth', 1.4, 'MarkerSize', 7);
grid on;
xlabel('Initialisation seed');
ylabel('Test loss');
if is_classification_result(result)
    title('Classification test loss by seed');
else
    title('Regression test loss by seed');
end
save_plot(fig, output_dir, 'test_loss_by_seed.png');
end

function plot_learned_bias_swarm(result, output_dir)
if ~isfield(result, 'bias_by_seed') || isempty(result.bias_by_seed)
    return;
end
B = result.bias_by_seed;
fig = figure('Color', 'w');
hold on;
colors = lines(max(1, numel(B)));
tick_labels = strings(1, numel(B));
for ii = 1:numel(B)
    if ~isfield(B(ii), 'values') || isempty(B(ii).values)
        continue;
    end
    y = double(B(ii).values(:));
    ok = isfinite(y);
    y = y(ok);
    if isempty(y)
        continue;
    end
    x = repmat(ii, numel(y), 1);
    tick_labels(ii) = string(B(ii).seed);
    if exist('swarmchart', 'file') == 2
        swarmchart(x, y, 6, colors(ii,:), 'filled', ...
            'MarkerFaceAlpha', 0.18, 'MarkerEdgeAlpha', 0.18);
    else
        jitter = deterministic_jitter(numel(y), 0.30);
        scatter(x + jitter, y, 6, colors(ii,:), 'filled', ...
            'MarkerFaceAlpha', 0.18, 'MarkerEdgeAlpha', 0.18);
    end
    med = median(y, 'omitnan');
    plot([ii - 0.28, ii + 0.28], [med, med], 'k-', 'LineWidth', 1.5);
end
hold off;
grid on;
xlim([0.5, numel(B) + 0.5]);
set(gca, 'XTick', 1:numel(B), 'XTickLabel', cellstr(tick_labels));
xlabel('Initialisation seed');
ylabel('Learned hidden bias');
title('Learned bias distributions by seed');
save_plot(fig, output_dir, 'learned_bias_swarm_by_seed.png');
end

function jitter = deterministic_jitter(n, width)
if n <= 0
    jitter = [];
    return;
end
idx = (1:n).';
jitter = width .* (2 .* mod(sin(idx .* 12.9898) .* 43758.5453, 1) - 1);
end

function tf = is_classification_result(result)
tf = isfield(result, 'domain') && strcmpi(char(result.domain), 'classification');
end

function plot_scalar_summary(result, T)
axis off;
lines = {'Held-out test summary'};
if isfield(T, 'loss') && ~isempty(T.loss)
    lines{end+1} = sprintf('Test loss: %s', format_scalar(T.loss)); %#ok<AGROW>
end

if is_regression_result(result) && has_regression_summary(T)
    lines = add_regression_lines(lines, get_regression_summary(T));
else
    has_value = has_task_value(T);
    value = get_task_value(T);
    label = test_value_label(result, T);
    if has_value
        lines{end+1} = sprintf('%s: %s', label, format_scalar(value)); %#ok<AGROW>
    end
end

if isfield(T, 'count') && ~isempty(T.count)
    lines{end+1} = sprintf('Samples/steps evaluated: %s', format_scalar(T.count)); %#ok<AGROW>
end
truth_summary = truth_diagnostic_summary(T);
if ~isempty(truth_summary)
    lines{end+1} = truth_summary; %#ok<AGROW>
end

text(0.03, 0.92, strjoin(lines, newline), 'Units', 'normalized', ...
    'VerticalAlignment', 'top', 'FontName', 'Helvetica', 'FontSize', 12);
title('Held-Out Test Summary');
end

function metrics = preferred_summary_metrics(result)
if isfield(result, 'domain') && strcmpi(char(result.domain), 'classification')
    metrics = ["AccuracyPercent", "Loss"];
elseif isfield(result, 'domain') && strcmpi(char(result.domain), 'regression')
    metrics = ["RMSE", "PearsonR", "PearsonP", "SignedErrorMean", "SignedErrorSD", "Loss"];
elseif isfield(result, 'domain') && any(strcmpi(char(result.domain), {'dynamical_systems','dynamics'}))
    metrics = ["WassersteinDistance", "Loss", "BestValidationWD", "BestValidationLoss"];
else
    metrics = ["PrimaryMetric", "Loss"];
end
end

function label = display_metric_name(metric)
switch char(metric)
    case 'AccuracyPercent'
        label = 'Accuracy [%]';
    case 'RMSE'
        label = 'RMSE';
    case 'PearsonR'
        label = 'Pearson r';
    case 'PearsonP'
        label = 'Pearson p';
    case 'SignedErrorMean'
        label = 'Signed error mean';
    case 'SignedErrorSD'
        label = 'Signed error SD';
    case 'WassersteinDistance'
        label = 'Wasserstein distance';
    case 'BestValidationWD'
        label = 'Best validation WD';
    case 'BestValidationLoss'
        label = 'Best validation loss';
    case 'PrimaryMetric'
        label = 'Primary metric';
    otherwise
        label = char(metric);
end
end

function plot_seed_distribution(result, output_dir)
if ~isfield(result, 'summary') || ~isfield(result.summary, 'seed_table') || ...
        ~istable(result.summary.seed_table) || height(result.summary.seed_table) == 0
    return;
end
T = result.summary.seed_table;
metric_name = pick_seed_distribution_metric(result, T);
if metric_name == "" || ~any(strcmp(T.Properties.VariableNames, char(metric_name)))
    return;
end
values = double(T.(char(metric_name)));
ok = isfinite(values);
if ~any(ok)
    return;
end

fig = figure('Color', 'w');
seed_values = double(T.Seed);
plot(seed_values(ok), values(ok), 'o', 'MarkerSize', 7, 'LineWidth', 1.4);
hold on;
yline(mean(values(ok), 'omitnan'), '-', 'LineWidth', 1.2);
hold off;
grid on;
xlabel('Initialisation seed');
ylabel(display_metric_name(metric_name));
title([display_metric_name(metric_name) ' across seeds']);
save_plot(fig, output_dir, 'test_metric_distribution_by_seed.png');
end

function metric_name = pick_seed_distribution_metric(result, T)
metric_name = "";
preferred = preferred_summary_metrics(result);
for ii = 1:numel(preferred)
    name = preferred(ii);
    if any(strcmp(T.Properties.VariableNames, char(name))) && any(isfinite(double(T.(char(name)))))
        metric_name = name;
        return;
    end
end
end

function tf = is_regression_result(result)
tf = isfield(result, 'domain') && strcmpi(char(result.domain), 'regression');
end

function tf = has_regression_summary(T)
tf = isfield(T, 'regression') || ...
    (isfield(T, 'Z') && isfield(T, 'Y') && ~isempty(T.Z) && ~isempty(T.Y));
end

function R = get_regression_summary(T)
if isfield(T, 'regression')
    R = T.regression;
else
    R = regression_summary_from_arrays(T.Z, T.Y);
end
end

function lines = add_regression_lines(lines, R)
if isfield(R, 'rmse')
    lines{end+1} = sprintf('RMSE: %s', format_scalar(R.rmse)); %#ok<AGROW>
end
if isfield(R, 'pearson_r')
    lines{end+1} = sprintf('Pearson r: %s', format_scalar(R.pearson_r)); %#ok<AGROW>
elseif isfield(R, 'r')
    lines{end+1} = sprintf('Pearson r: %s', format_scalar(R.r)); %#ok<AGROW>
end
if isfield(R, 'pearson_p')
    lines{end+1} = sprintf('Pearson p: %s', format_scalar(R.pearson_p)); %#ok<AGROW>
elseif isfield(R, 'p')
    lines{end+1} = sprintf('Pearson p: %s', format_scalar(R.p)); %#ok<AGROW>
end
if isfield(R, 'signed_error_mean')
    lines{end+1} = sprintf('Signed error mean: %s', format_scalar(R.signed_error_mean)); %#ok<AGROW>
end
if isfield(R, 'signed_error_sd')
    lines{end+1} = sprintf('Signed error SD: %s', format_scalar(R.signed_error_sd)); %#ok<AGROW>
end
if isfield(R, 'n')
    lines{end+1} = sprintf('Finite prediction-target pairs: %s', format_scalar(R.n)); %#ok<AGROW>
end
end

function R = regression_summary_from_arrays(y_pred, y_true)
y_pred = double(y_pred(:));
y_true = double(y_true(:));
valid = isfinite(y_pred) & isfinite(y_true);
y_pred = y_pred(valid);
y_true = y_true(valid);
err = y_pred - y_true;
R = struct('rmse', NaN, 'pearson_r', NaN, 'pearson_p', NaN, ...
    'signed_error_mean', NaN, 'signed_error_sd', NaN, 'n', int32(numel(err)));
if isempty(err)
    return;
end
R.rmse = sqrt(mean(err.^2));
R.signed_error_mean = mean(err);
if numel(err) > 1
    R.signed_error_sd = std(err, 0);
end
if numel(err) >= 3 && std(y_pred) > 0 && std(y_true) > 0
    try
        [C, P] = corrcoef(y_pred, y_true);
        R.pearson_r = C(1,2);
        R.pearson_p = P(1,2);
    catch
        yp = y_pred - mean(y_pred);
        yt = y_true - mean(y_true);
        R.pearson_r = sum(yp .* yt) / (sqrt(sum(yp.^2)) * sqrt(sum(yt.^2)));
    end
end
end

function value = get_task_value(T)
value = NaN;
if isfield(T, 'wasserstein_distance') && ~isempty(T.wasserstein_distance)
    value = double(T.wasserstein_distance);
elseif isfield(T, 'wd') && ~isempty(T.wd)
    value = double(T.wd);
elseif isfield(T, 'metric') && ~isempty(T.metric)
    value = double(T.metric);
end
end

function txt = truth_diagnostic_summary(T)
txt = '';
if ~isfield(T, 'truth_simulation_failed_by_ic') || isempty(T.truth_simulation_failed_by_ic)
    return;
end
failed = logical(T.truth_simulation_failed_by_ic(:));
if ~any(failed)
    return;
end
failed_idx = find(failed).';
seed_text = 'n/a';
if isfield(T, 'closed_loop_test_ic_seed') && ~isempty(T.closed_loop_test_ic_seed)
    seed_text = mat2str(T.closed_loop_test_ic_seed);
elseif isfield(T, 'closed_loop_ic_seed') && ~isempty(T.closed_loop_ic_seed)
    seed_text = mat2str(T.closed_loop_ic_seed);
end
status_text = '';
if isfield(T, 'truth_diagnostic_by_ic') && numel(T.truth_diagnostic_by_ic) >= failed_idx(1)
    statuses = cell(1, numel(failed_idx));
    for kk = 1:numel(failed_idx)
        D = T.truth_diagnostic_by_ic{failed_idx(kk)};
        if isstruct(D) && isfield(D, 'status')
            statuses{kk} = char(D.status);
        else
            statuses{kk} = 'unknown';
        end
    end
    status_text = sprintf(' statuses=%s', strjoin(statuses, ','));
end
txt = sprintf('Truth continuation failed for IC(s) %s; test IC seed=%s.%s', ...
    mat2str(failed_idx), seed_text, status_text);
end

function tf = has_task_value(T)
tf = (isfield(T, 'wasserstein_distance') && ~isempty(T.wasserstein_distance)) || ...
    (isfield(T, 'wd') && ~isempty(T.wd)) || ...
    (isfield(T, 'metric') && ~isempty(T.metric));
end

function label = test_value_label(result, T)
if isfield(T, 'wasserstein_distance') || isfield(T, 'wd')
    label = 'Wasserstein distance';
    return;
end
domain = '';
if isfield(result, 'domain') && ~isempty(result.domain)
    domain = lower(char(result.domain));
end
switch domain
    case 'classification'
        label = 'Accuracy [%]';
    case 'regression'
        label = 'Pearson r';
    case {'dynamics', 'dynamical_systems'}
        label = 'Wasserstein distance';
    otherwise
        label = 'Task score';
end
end

function value = get_field_or_nan(S, name)
if isstruct(S) && isfield(S, name) && ~isempty(S.(name))
    value = S.(name);
else
    value = NaN;
end
end

function value = get_result_field_local(S, field, default_value)
if isstruct(S) && isfield(S, field) && ~isempty(S.(field))
    value = S.(field);
else
    value = default_value;
end
end

function txt = format_scalar(x)
x = double(x);
if isempty(x) || ~isfinite(x(1))
    txt = 'n/a';
elseif abs(x(1)) >= 1e4 || abs(x(1)) < 1e-3
    txt = sprintf('%.6g', x(1));
else
    txt = sprintf('%.6f', x(1));
end
end

function plot_training_validation_loss_from_model(result, output_dir)
if isfield(result, 'model_files') && numel(result.model_files) > 1
    plot_training_validation_loss_from_seed_models(result, output_dir);
    return;
end

train_result = load_linked_training_result(result);
if isempty(train_result) || ~isfield(train_result, 'history')
    return;
end
[train_loss, val_loss, val_label] = extract_loss_curves(train_result);
if isempty(train_loss)
    return;
end

epochs = (1:numel(train_loss)).';
fig = figure('Color', 'w');
plot(epochs, double(train_loss(:)), 'LineWidth', 1.4);
hold on;
legend_entries = {'Training loss'};
if ~isempty(val_loss)
    ok = isfinite(double(val_loss(:)));
    plot(epochs(ok), double(val_loss(ok)), 'LineWidth', 1.4);
    legend_entries{end+1} = val_label; %#ok<AGROW>
end
snn_plot_continuation_boundaries(gca, train_result, numel(epochs));
hold off;
xlabel('Epoch');
ylabel('Loss');
title('Training and validation loss');
grid on;
set_loss_axis_log(gca);
legend(legend_entries, 'Location', 'best');
save_plot(fig, output_dir, 'training_validation_loss.png');
end

function plot_training_validation_loss_from_seed_models(result, output_dir)
files = result.model_files;
if isstring(files)
    files = cellstr(files(:));
elseif ischar(files)
    files = {files};
end

fig = figure('Color', 'w');
hold on;
legend_entries = {};
colors = lines(max(1, numel(files)));
continuation_boundaries = zeros(1, 0);
for ii = 1:numel(files)
    if exist(files{ii}, 'file') ~= 2
        continue;
    end
    train_result = load_training_result_file(files{ii});
    if isempty(train_result) || ~isfield(train_result, 'history')
        continue;
    end
    [train_loss, val_loss, val_label] = extract_loss_curves(train_result);
    if isempty(train_loss)
        continue;
    end
    epochs = (1:numel(train_loss)).';
    seed_value = ii;
    if isfield(result, 'seed_list') && numel(result.seed_list) >= ii
        seed_value = result.seed_list(ii);
    end
    plot(epochs, double(train_loss(:)), '-', 'Color', colors(ii,:), 'LineWidth', 1.2);
    legend_entries{end+1} = sprintf('Seed %g training', double(seed_value)); %#ok<AGROW>
    if ~isempty(val_loss)
        ok = isfinite(double(val_loss(:)));
        plot(epochs(ok), double(val_loss(ok)), '--', 'Color', colors(ii,:), 'LineWidth', 1.2);
        legend_entries{end+1} = sprintf('Seed %g %s', double(seed_value), val_label); %#ok<AGROW>
    end
    continuation_boundaries = union(continuation_boundaries, ...
        snn_plot_continuation_boundaries([], train_result, numel(epochs)), 'stable');
end
if ~isempty(continuation_boundaries)
    % Seeds are overlaid in one axes, so mark the shared phase boundaries once.
    snn_plot_continuation_boundaries(gca, struct('continuation', ...
        struct('boundaries', continuation_boundaries)), inf);
end
hold off;

if isempty(legend_entries)
    close(fig);
    return;
end
xlabel('Epoch');
ylabel('Loss');
title('Training and validation loss across seeds');
grid on;
set_loss_axis_log(gca);
legend(legend_entries, 'Location', 'best');
save_plot(fig, output_dir, 'training_validation_loss_by_seed.png');
end

function train_result = load_linked_training_result(result)
train_result = [];
if ~isfield(result, 'model_file') || isempty(result.model_file)
    return;
end
train_result = load_training_result_file(char(result.model_file));
end

function train_result = load_training_result_file(model_file)
train_result = [];
if exist(model_file, 'file') ~= 2
    warning('snn_plot_test_result:modelFileMissing', ...
        'Could not plot training losses because the saved model file was not found.');
    return;
end
S = load(model_file);
if isfield(S, 'result')
    train_result = S.result;
    return;
end
names = fieldnames(S);
for ii = 1:numel(names)
    candidate = S.(names{ii});
    if isstruct(candidate) && isfield(candidate, 'history')
        train_result = candidate;
        return;
    end
end
end

function [train_loss, val_loss, val_label] = extract_loss_curves(train_result)
train_loss = [];
val_loss = [];
val_label = 'Validation loss';
hist = train_result.history;
if isstruct(hist)
    if isfield(hist, 'train_loss')
        train_loss = double(hist.train_loss(:));
    elseif isfield(hist, 'loss')
        train_loss = double(hist.loss(:));
    end
    if isfield(hist, 'val_loss')
        val_loss = double(hist.val_loss(:));
    end
else
    train_loss = double(hist(:));
end
if ~isempty(val_loss) && ~isempty(train_loss) && numel(val_loss) ~= numel(train_loss)
    n = min(numel(train_loss), numel(val_loss));
    train_loss = train_loss(1:n);
    val_loss = val_loss(1:n);
end
end

function set_loss_axis_log(ax)
vals = findobj(ax, 'Type', 'line');
y = [];
for ii = 1:numel(vals)
    y = [y; vals(ii).YData(:)]; %#ok<AGROW>
end
y = y(isfinite(y) & y > 0);
if ~isempty(y)
    set(ax, 'YScale', 'log');
end
end

function save_plot(fig, output_dir, filename)
drawnow;
if isempty(output_dir)
    return;
end
pathstr = fullfile(output_dir, filename);
try
    if exist('exportgraphics', 'file') == 2
        exportgraphics(fig, pathstr, 'Resolution', 180);
    else
        saveas(fig, pathstr);
    end
catch ME
    warning('Could not save plot "%s": %s', pathstr, ME.message);
end
end

function tiledlayout_compat(m, n)
if exist('tiledlayout', 'file') ~= 0
    tiledlayout(m, n, 'TileSpacing', 'compact', 'Padding', 'compact');
    tile_state('set', true, m, n, 0);
else
    tile_state('set', false, m, n, 0);
end
end

function nexttile_compat()
[use_tiles, m, n, idx] = tile_state('get');
if use_tiles && exist('nexttile', 'file') ~= 0
    nexttile;
else
    idx = idx + 1;
    tile_state('set', false, m, n, idx);
    subplot(m, n, idx);
end
end

function varargout = tile_state(action, use_tiles, m, n, idx)
persistent p_use_tiles p_m p_n p_idx
if isempty(p_use_tiles)
    p_use_tiles = false;
    p_m = 1;
    p_n = 1;
    p_idx = 0;
end
if strcmp(action, 'set')
    p_use_tiles = logical(use_tiles);
    p_m = double(m);
    p_n = double(n);
    p_idx = double(idx);
end
if nargout > 0
    varargout = {p_use_tiles, p_m, p_n, p_idx};
end
end
