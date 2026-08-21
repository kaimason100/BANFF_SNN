% Package orientation: Shared MATLAB utility used across tasks. Read this with the caller open so input/output structs and saved-result fields are clear.

function snn_plot_check_summary(summary_table)
%SNN_PLOT_CHECK_SUMMARY Plot CPU-vs-GPU absolute and relative differences.
%   The first panel gives the per-seed verdict. The remaining panels show
%   measured differences, with tolerance lines when the summary table has them.

if isempty(summary_table)
    warning('snn_plot_check_summary:empty', 'No check rows to plot.');
    return;
end

labels = string(summary_table.seed);
if any(isnan(summary_table.seed))
    labels = string(1:height(summary_table));
end

figure('Color', 'w');
tiledlayout_compat(3, 2);

nexttile_compat();
plot_verdict_panel(summary_table, labels);

nexttile_compat();
plot_grouped_log_bars(summary_table, labels, ...
    {'initial_abs','after_abs','bias_abs'}, ...
    {'Initial output','Post-training output','Trained bias'}, ...
    'Output and bias absolute differences', 'Maximum absolute CPU-GPU difference', 'tol_abs');

nexttile_compat();
plot_grouped_log_bars(summary_table, labels, ...
    {'initial_rel','after_rel','bias_rel'}, ...
    {'Initial output','Post-training output','Trained bias'}, ...
    'Output and bias relative differences', 'Maximum relative CPU-GPU difference', 'tol_rel');

nexttile_compat();
plot_grouped_log_bars(summary_table, labels, ...
    {'loss_after_abs','loss_delta_abs_diff','metric_after_abs','metric_delta_abs_diff'}, ...
    {'Post-training loss','Loss change','Post-training metric','Metric change'}, ...
    'Performance differences', 'Absolute CPU-GPU difference', 'tol_abs');

nexttile_compat();
plot_grouped_log_bars(summary_table, labels, ...
    {'neural_spike_abs','neural_rate_abs','neural_voltage_abs', ...
     'neural_input_current_abs','neural_recurrent_current_abs','neural_adaptation_abs'}, ...
    neural_labels(), ...
    'Spiking and network absolute differences', 'Maximum absolute CPU-GPU difference', '');

nexttile_compat();
plot_grouped_log_bars(summary_table, labels, ...
    {'main_diff_vs_training_percent','bias_diff_vs_training_percent'}, ...
    {'Output mismatch / training effect','Bias mismatch / training effect'}, ...
    'Mismatch compared with learning effect', 'Percent', '');

sgtitle_compat('CPU-GPU differences');
end

function plot_verdict_panel(T, labels)
scores = verdict_scores(T);
b = bar(scores, 'FaceColor', 'flat');
b.CData = scores;
colormap(gca, [0.2 0.65 0.35; 0.95 0.7 0.25; 0.82 0.25 0.22]);
ylim([0 3.5]);
yticks([1 2 3]);
yticklabels({'OK','Check scale','Review'});
xticks(1:height(T));
xticklabels(labels);
xlabel('Check seed');
title('Read this first');
grid on;
for ii = 1:height(T)
    text(ii, min(3.25, scores(ii) + 0.12), char(T.verdict(ii)), ...
        'HorizontalAlignment', 'center', 'FontSize', 8);
end
end

function scores = verdict_scores(T)
scores = ones(height(T), 1);
if any(strcmp(T.Properties.VariableNames, 'verdict'))
    verdict = string(T.verdict);
    scores(verdict == "CHECK SCALE" | verdict == "OUTPUT SKIPPED") = 2;
    scores(verdict == "REVIEW") = 3;
end
end

function plot_grouped_log_bars(T, labels, fields, legend_labels, ttl, y_label, tolerance_field)
Y = table_fields_to_matrix(T, fields);
if all(isnan(Y(:)))
    axis off;
    title([ttl ' unavailable']);
    return;
end
Yplot = Y;
Yplot(isfinite(Yplot)) = max(Yplot(isfinite(Yplot)), realmin('double'));
bar(Yplot);
set(gca, 'YScale', 'log');
grid on;
plot_tolerance_line(T, tolerance_field);
xticks(1:height(T));
xticklabels(labels);
xlabel('Check seed');
ylabel(y_label);
legend(legend_labels, 'Location', 'best');
title(ttl);
end

function plot_tolerance_line(T, tolerance_field)
if isempty(tolerance_field) || ~any(strcmp(T.Properties.VariableNames, tolerance_field))
    return;
end
tol = double(T.(tolerance_field));
tol = tol(isfinite(tol) & tol > 0);
if isempty(tol)
    return;
end
tol_value = max(tol);
hold on;
x_limits = xlim;
plot(x_limits, [tol_value tol_value], 'k--', 'LineWidth', 1.0, 'HandleVisibility', 'off');
xlim(x_limits);
hold off;
end

function Y = table_fields_to_matrix(T, fields)
Y = nan(height(T), numel(fields));
for jj = 1:numel(fields)
    if any(strcmp(T.Properties.VariableNames, fields{jj}))
        Y(:,jj) = double(T.(fields{jj}));
    end
end
Y(~isfinite(Y)) = nan;
end

function labels = neural_labels()
labels = {'Spike raster','Rate','Voltage','Input current','Recurrent current','Adaptation'};
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
