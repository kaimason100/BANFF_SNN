% Package orientation: Shared MATLAB utility used across tasks. Read this with the caller open so input/output structs and saved-result fields are clear.

function snn_plot_check_outputs_and_biases(results, max_points)
%SNN_PLOT_CHECK_OUTPUTS_AND_BIASES Visualize CPU/GPU outputs and biases.
%   Plots one figure per seed. Each figure shows CPU and GPU values before
%   training, CPU and GPU values after training, CPU-GPU differences, and
%   the within-backend training changes.

if nargin < 2 || isempty(max_points)
    max_points = 2000;
end
if iscell(results)
    result_cells = results(:);
else
    results = results(:);
    result_cells = cell(numel(results), 1);
    for ii = 1:numel(results)
        result_cells{ii} = results(ii);
    end
end
if isempty(result_cells)
    warning('snn_plot_check_outputs_and_biases:empty', 'No check results to plot.');
    return;
end

for ii = 1:numel(result_cells)
    r = result_cells{ii};
    if ~isfield(r, 'snapshots') || isempty(r.snapshots)
        warning('snn_plot_check_outputs_and_biases:missingSnapshots', ...
            'Result %d does not contain output/bias snapshots. Re-run the check script to generate them.', ii);
        continue;
    end
    plot_one_result(r, ii, max_points);
end
end

function plot_one_result(result, index, max_points)
S = result.snapshots;
seed_value = local_seed(result, index);

figure('Color', 'w');
tiledlayout_compat(4, 2);

nexttile_compat();
plot_cpu_gpu_pair(S.cpu_output_before, S.gpu_output_before, max_points);
title(metric_title('Outputs before training', get_cmp(result, 'initial')));
ylabel('Value');

nexttile_compat();
plot_cpu_gpu_pair(S.cpu_bias_before, S.gpu_bias_before, max_points);
title(metric_title('Biases before training', get_cmp(result, 'bias_before_training')));
ylabel('Bias');

nexttile_compat();
plot_cpu_gpu_pair(S.cpu_output_after, S.gpu_output_after, max_points);
title(metric_title('Outputs after training', get_cmp(result, 'after_training')));
ylabel('Value');

nexttile_compat();
plot_cpu_gpu_pair(S.cpu_bias_after, S.gpu_bias_after, max_points);
title(metric_title('Biases after training', get_cmp(result, 'bias_after_training')));
ylabel('Bias');

nexttile_compat();
plot_before_after_difference(S.cpu_output_before, S.gpu_output_before, ...
    S.cpu_output_after, S.gpu_output_after, max_points);
title('Output CPU-GPU difference');
ylabel('CPU - GPU');

nexttile_compat();
plot_before_after_difference(S.cpu_bias_before, S.gpu_bias_before, ...
    S.cpu_bias_after, S.gpu_bias_after, max_points);
title('Bias CPU-GPU difference');
ylabel('CPU - GPU');

nexttile_compat();
plot_training_change(S.cpu_output_before, S.cpu_output_after, ...
    S.gpu_output_before, S.gpu_output_after, max_points);
title('Output training change');
ylabel('After - before');

nexttile_compat();
plot_training_change(S.cpu_bias_before, S.cpu_bias_after, ...
    S.gpu_bias_before, S.gpu_bias_after, max_points);
title('Bias training change');
ylabel('After - before');

sgtitle_compat(sprintf('CPU/GPU outputs and biases, seed %g', seed_value));
end

function plot_cpu_gpu_pair(cpu_values, gpu_values, max_points)
[x_cpu, y_cpu] = sampled_vector(cpu_values, max_points);
[x_gpu, y_gpu] = sampled_vector(gpu_values, max_points);
plot(x_cpu, y_cpu, '-', 'LineWidth', 1.1); hold on;
plot(x_gpu, y_gpu, '--', 'LineWidth', 1.1); hold off;
grid on;
xlabel('Flattened element index');
legend({'CPU','GPU'}, 'Location', 'best');
end

function plot_before_after_difference(cpu_before, gpu_before, cpu_after, gpu_after, max_points)
[x_before, d_before] = sampled_difference(cpu_before, gpu_before, max_points);
[x_after, d_after] = sampled_difference(cpu_after, gpu_after, max_points);
plot(x_before, d_before, '-', 'LineWidth', 1.1); hold on;
plot(x_after, d_after, '--', 'LineWidth', 1.1); hold off;
grid on;
xlabel('Flattened element index');
legend({'Before training','After training'}, 'Location', 'best');
yline_compat(0);
end

function plot_training_change(cpu_before, cpu_after, gpu_before, gpu_after, max_points)
[x_cpu, d_cpu] = sampled_difference(cpu_after, cpu_before, max_points);
[x_gpu, d_gpu] = sampled_difference(gpu_after, gpu_before, max_points);
plot(x_cpu, d_cpu, '-', 'LineWidth', 1.1); hold on;
plot(x_gpu, d_gpu, '--', 'LineWidth', 1.1); hold off;
grid on;
xlabel('Flattened element index');
legend({'CPU change','GPU change'}, 'Location', 'best');
yline_compat(0);
end

function [x, y] = sampled_difference(a, b, max_points)
a = double(a(:));
b = double(b(:));
n = min(numel(a), numel(b));
if n < 1
    x = nan;
    y = nan;
    return;
end
d = a(1:n) - b(1:n);
idx = sample_indices(n, max_points);
x = idx;
y = d(idx);
end

function [x, y] = sampled_vector(values, max_points)
y = double(values(:));
if isempty(y)
    x = nan;
    y = nan;
    return;
end
idx = sample_indices(numel(y), max_points);
x = idx;
y = y(idx);
end

function idx = sample_indices(n, max_points)
n = double(n);
max_points = max(1, double(max_points));
if n <= max_points
    idx = 1:n;
else
    idx = unique(round(linspace(1, n, max_points)));
end
idx = idx(:).';
end

function ttl = metric_title(prefix, cmp)
ttl = sprintf('%s | max %.3g, rms %.3g, rel %.3g', ...
    prefix, cmp.max_abs, cmp.rms, cmp.max_rel);
end

function cmp = get_cmp(result, field_name)
cmp = struct('max_abs', nan, 'rms', nan, 'max_rel', nan);
if isfield(result, field_name) && isstruct(result.(field_name))
    fields = {'max_abs','rms','max_rel'};
    for ii = 1:numel(fields)
        if isfield(result.(field_name), fields{ii})
            cmp.(fields{ii}) = double(result.(field_name).(fields{ii}));
        end
    end
end
end

function seed_value = local_seed(result, index)
if isfield(result, 'options') && isfield(result.options, 'seed')
    seed_value = double(result.options.seed);
elseif isfield(result, 'seed')
    seed_value = double(result.seed);
else
    seed_value = index;
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

function sgtitle_compat(txt)
if exist('sgtitle', 'file') ~= 0
    sgtitle(txt);
else
    annotation('textbox', [0 0.95 1 0.04], 'String', txt, ...
        'HorizontalAlignment', 'center', 'EdgeColor', 'none');
end
end

function yline_compat(y)
if exist('yline', 'file') ~= 0
    h = yline(y, 'k:');
    set(h, 'HandleVisibility', 'off');
else
    x_limits = xlim;
    hold on;
    plot(x_limits, [y y], 'k:', 'HandleVisibility', 'off');
    xlim(x_limits);
    hold off;
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
