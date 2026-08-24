% Package orientation: Publication renderer for the Lorenz, Yacht, and BC neuron sweeps.

function figures = plot_lorenz_neuron_sweep_publication(lorenz_sweep, opts)
%PLOT_LORENZ_NEURON_SWEEP_PUBLICATION Render all neuron-sweep publication panels.
% opts.classification_sweep and opts.regression_sweep are the corresponding
% saved breast-cancer and Yacht sweep summaries.

if nargin < 2 || isempty(opts), opts = struct(); end
bc_sweep = required_opt(opts, 'classification_sweep');
yacht_sweep = required_opt(opts, 'regression_sweep');
prepared = prepare_phase_conditions(lorenz_sweep, opts);

figures = struct();
figures.phase_portraits = plot_phase_figure(lorenz_sweep, prepared, opts);
figures.wasserstein_swarm = plot_wd_figure(lorenz_sweep, opts);
figures.classification_accuracy = plot_accuracy_figure(bc_sweep, opts);
figures.regression_performance = plot_regression_figure(yacht_sweep, opts);
figures.combined = plot_combined_figure(lorenz_sweep, prepared, bc_sweep, yacht_sweep, opts);
end

function prepared = prepare_phase_conditions(sweep, opts)
prepared = repmat(struct('pred', [], 'truth', []), 1, numel(sweep.conditions));
for ii = 1:numel(sweep.conditions)
    pred = double(sweep.conditions(ii).pred_norm);
    truth = double(sweep.conditions(ii).true_norm);
    n = min(size(pred, 1), size(truth, 1));
    if n < 2 || size(pred, 2) < 2 || size(truth, 2) < 2
        error('plot_lorenz_neuron_sweep_publication:badPhaseData', ...
            'Lorenz N_hidden=%d has no finite two-dimensional test trajectory.', sweep.conditions(ii).n_hidden);
    end
    tfrac = max(eps, min(1, get_opt(opts, 'phase_tfrac', 1)));
    stride = max(1, round(get_opt(opts, 'phase_stride', 1)));
    last = max(2, 1 + floor(tfrac * (n - 1)));
    idx = 1:stride:last;
    prepared(ii).pred = pred(idx, 1:2);
    prepared(ii).truth = truth(idx, 1:2);
end
end

function fig = plot_phase_figure(sweep, prepared, opts)
n = numel(sweep.conditions); fig = figure('Color', 'w');
pos = axes_grid_positions(2, n, get_opt(opts, 'left_margin', .06), get_opt(opts, 'right_margin', .03), ...
    get_opt(opts, 'bottom_margin', .08), get_opt(opts, 'top_margin', .10), get_opt(opts, 'hgap', .025), get_opt(opts, 'vgap', .08), true);
add_figure_title(fig, get_opt(opts, 'phase_figure_title', 'Lorenz phase portraits'), get_opt(opts, 'figure_title_font_size', 17), pos, 1);
limits = phase_limits(prepared);
for ii = 1:n
    ax = axes(fig, 'Units', 'normalized', 'Position', pos{1, ii});
    draw_phase(ax, prepared(ii), limits, opts);
    title(ax, condition_label(sweep, ii, opts), 'FontSize', get_opt(opts, 'panel_title_font_size', 14), 'Interpreter', 'none');
end
legend_pos = get_opt(opts, 'separate_network_true_legend_position', []);
if isempty(legend_pos), legend_pos = pos{2, 1}; end
add_network_true_legend(fig, legend_pos, opts);
end

function fig = plot_wd_figure(sweep, opts)
fig = figure('Color', 'w'); ax = axes(fig, 'Position', get_opt(opts, 'single_metric_axes_position', [.12 .16 .80 .72]));
plot_wd_swarm(ax, sweep, opts);
title(ax, get_opt(opts, 'wd_figure_title', 'Lorenz phase-portrait Wasserstein distance'), 'FontSize', get_opt(opts, 'figure_title_font_size', 17));
end

function fig = plot_accuracy_figure(sweep, opts)
fig = figure('Color', 'w'); ax = axes(fig, 'Position', get_opt(opts, 'single_metric_axes_position', [.12 .16 .80 .72]));
plot_metric_swarm(ax, sweep, 'accuracy', opts);
title(ax, get_opt(opts, 'accuracy_figure_title', 'Breast-cancer classification accuracy'), 'FontSize', get_opt(opts, 'figure_title_font_size', 17));
end

function fig = plot_regression_figure(sweep, opts)
fig = figure('Color', 'w');
pos = get_opt(opts, 'regression_separate_axes_positions', {[.11 .16 .36 .72], [.58 .16 .36 .72]});
ax_r = axes(fig, 'Position', pos{1}); plot_metric_swarm(ax_r, sweep, 'pearson_r', opts);
title(ax_r, get_opt(opts, 'pearson_figure_title', 'Yacht Pearson r'), 'FontSize', get_opt(opts, 'figure_title_font_size', 17));
ax_rmse = axes(fig, 'Position', pos{2}); plot_metric_swarm(ax_rmse, sweep, 'rmse', opts);
title(ax_rmse, get_opt(opts, 'rmse_figure_title', 'Yacht RMSE'), 'FontSize', get_opt(opts, 'figure_title_font_size', 17));
end

function fig = plot_combined_figure(lorenz, prepared, bc, yacht, opts)
fig = figure('Color', 'w'); n = numel(lorenz.conditions);
left = get_opt(opts, 'combined_left_margin', .07); right = get_opt(opts, 'combined_right_margin', .03);
hgap = get_opt(opts, 'combined_hgap', .02); width = (1-left-right-(n-1)*hgap)/n;
phase_y = get_opt(opts, 'combined_phase_y', .62); phase_h = get_opt(opts, 'combined_phase_height', .27);
limits = phase_limits(prepared);
for ii = 1:n
    ax = axes(fig, 'Units', 'normalized', 'Position', [left+(ii-1)*(width+hgap), phase_y, width, phase_h]);
    draw_phase(ax, prepared(ii), limits, opts);
    title(ax, condition_label(lorenz, ii, opts), 'FontSize', get_opt(opts, 'panel_title_font_size', 14), 'Interpreter', 'none');
end
phase_title = get_opt(opts, 'combined_phase_title', 'Lorenz phase portraits');
if ~isempty(strtrim(char(phase_title)))
    annotation(fig, 'textbox', [.30 phase_y+phase_h+.045 .40 .035], 'String', phase_title, ...
        'Interpreter', 'none', 'FontSize', get_opt(opts, 'combined_phase_title_font_size', 12), ...
        'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'EdgeColor', 'none');
end
add_figure_title(fig, get_opt(opts, 'combined_figure_title', 'Neuron-count scaling across tasks'), get_opt(opts, 'combined_figure_title_font_size', 18), ...
    {[left phase_y width phase_h], [left+(n-1)*(width+hgap) phase_y width phase_h]}, 1);

metric_pos = get_opt(opts, 'combined_metric_axes_positions', {[.09 .34 .36 .20], [.57 .34 .36 .20], [.09 .08 .36 .20], [.57 .08 .36 .20]});
ax_wd = axes(fig, 'Position', metric_pos{1}); plot_wd_swarm(ax_wd, lorenz, opts); title(ax_wd, get_opt(opts, 'combined_wd_title', 'Lorenz WD'), 'FontSize', get_opt(opts, 'panel_title_font_size', 14));
ax_acc = axes(fig, 'Position', metric_pos{2}); plot_metric_swarm(ax_acc, bc, 'accuracy', opts); title(ax_acc, get_opt(opts, 'combined_accuracy_title', 'Breast-cancer accuracy'), 'FontSize', get_opt(opts, 'panel_title_font_size', 14));
ax_r = axes(fig, 'Position', metric_pos{3}); plot_metric_swarm(ax_r, yacht, 'pearson_r', opts); title(ax_r, get_opt(opts, 'combined_pearson_title', 'Yacht Pearson r'), 'FontSize', get_opt(opts, 'panel_title_font_size', 14));
ax_rmse = axes(fig, 'Position', metric_pos{4}); plot_metric_swarm(ax_rmse, yacht, 'rmse', opts); title(ax_rmse, get_opt(opts, 'combined_rmse_title', 'Yacht RMSE'), 'FontSize', get_opt(opts, 'panel_title_font_size', 14));
legend_pos = get_opt(opts, 'combined_network_true_legend_position', [0.02 .54 .20 .05]);
if logical(get_opt(opts, 'combined_network_true_legend_enable', true))
    add_network_true_legend(fig, legend_pos, opts);
end
if logical(get_opt(opts, 'combined_panel_letters_enable', true))
    add_panel_letter(fig, 'A', [.005 min(.94, phase_y+phase_h+.025) .035 .035], opts);
    add_panel_letter(fig, 'B', [metric_pos{1}(1)-.095 metric_pos{1}(2)+metric_pos{1}(4)+.012 .035 .035], opts);
    add_panel_letter(fig, 'C', [metric_pos{2}(1)-.08 metric_pos{2}(2)+metric_pos{2}(4)+.012 .035 .035], opts);
    add_panel_letter(fig, 'D', [metric_pos{3}(1)-.095 metric_pos{3}(2)+metric_pos{3}(4)+.012 .035 .035], opts);
    add_panel_letter(fig, 'E', [metric_pos{4}(1)-.08 metric_pos{4}(2)+metric_pos{4}(4)+.012 .035 .035], opts);
end
end

function draw_phase(ax, P, limits, opts)
hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on'); set_fixed_outer(ax); ax.FontSize = get_opt(opts, 'fs_ticks', 12);
plot(ax, P.pred(:,1), P.pred(:,2), '-', 'Color', get_opt(opts, 'network_color', [0 0 0]), 'LineWidth', get_opt(opts, 'phase_line_width', .7));
xlabel(ax, get_opt(opts, 'phase_x_label', 'x_1'), 'FontSize', get_opt(opts, 'fs_labels', 13));
ylabel(ax, get_opt(opts, 'phase_y_label', 'x_2'), 'FontSize', get_opt(opts, 'fs_labels', 13)); axis(ax, 'square');
xlim(ax, limits(1:2)); ylim(ax, limits(3:4));
if logical(get_opt(opts, 'phase_true_inset_enable', true)), add_true_phase_inset(ax, P.truth, limits, opts); end
end

function plot_wd_swarm(ax, sweep, opts)
values = arrayfun(@(x) x.wd(:), sweep.conditions, 'UniformOutput', false);
plot_values(ax, values, condition_labels(sweep, opts), get_opt(opts, 'wd_x_label', 'Number of neurons'), get_opt(opts, 'wd_y_label', 'Phase-portrait WD'), get_opt(opts, 'wd_y_scale', 'log'), opts, true, true, true);
end

function plot_metric_swarm(ax, sweep, field, opts)
values = arrayfun(@(x) x.(field)(:), sweep.conditions, 'UniformOutput', false);
switch field
    case 'accuracy'
        ylabel_text = get_opt(opts, 'accuracy_y_label', 'Accuracy (%)'); yscale = get_opt(opts, 'accuracy_y_scale', 'linear'); ylim_value = get_opt(opts, 'accuracy_y_limits', [0 100]);
    case 'pearson_r'
        ylabel_text = get_opt(opts, 'pearson_y_label', 'Pearson r'); yscale = get_opt(opts, 'pearson_y_scale', 'linear'); ylim_value = get_opt(opts, 'pearson_y_limits', [-1 1]);
    case 'rmse'
        ylabel_text = get_opt(opts, 'rmse_y_label', 'RMSE'); yscale = get_opt(opts, 'rmse_y_scale', 'linear'); ylim_value = get_opt(opts, 'rmse_y_limits', []);
end
plot_values(ax, values, condition_labels(sweep, opts), get_opt(opts, 'metric_x_label', 'Number of neurons'), ylabel_text, yscale, opts, false, false, false);
if ~isempty(ylim_value), ylim(ax, ylim_value); end
end

function plot_values(ax, values_by_condition, labels, xlabel_text, ylabel_text, yscale, opts, add_legend, show_errorbars, show_points)
hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on'); set_fixed_outer(ax); ax.FontSize = get_opt(opts, 'fs_ticks', 12);
point_handle = []; summary_handle = [];
all_values = vertcat(values_by_condition{:}); all_values = double(all_values(isfinite(all_values)));
if strcmpi(yscale, 'log') && ~isempty(all_values)
    bar_base = max(eps, min(all_values) * get_opt(opts, 'log_bar_base_fraction', .8));
else
    bar_base = 0;
end
for ii = 1:numel(values_by_condition)
    values = double(values_by_condition{ii}); values = values(isfinite(values));
    jitter = deterministic_jitter(numel(values), get_opt(opts, 'swarm_width', .22));
    if show_points
        h = scatter(ax, ii+jitter, values, get_opt(opts, 'bottom_panel_marker_size', get_opt(opts, 'swarm_marker_size', 34)), 'filled', 'MarkerFaceColor', get_opt(opts, 'swarm_point_color', [0 0 0]), 'MarkerFaceAlpha', get_opt(opts, 'swarm_point_alpha', .7), 'MarkerEdgeColor', 'none');
        if isempty(point_handle), point_handle = h; end
    end
    if ~isempty(values)
        mu = mean(values); sigma = std(values, 0);
        hbar = bar(ax, ii, mu, get_opt(opts, 'bottom_panel_bar_width', .60), ...
            'FaceColor', get_opt(opts, 'bottom_panel_bar_color', [.55 .64 .75]), 'EdgeColor', 'none');
        hbar.BaseValue = bar_base;
        if isprop(hbar, 'FaceAlpha'), hbar.FaceAlpha = get_opt(opts, 'bottom_panel_bar_alpha', .78); end
        uistack(hbar, 'bottom');
        if numel(values) > 1 && show_errorbars
            summary_handle = errorbar(ax, ii, mu, sigma, 'Color', get_opt(opts, 'swarm_mean_color', [0 .447 .741]), 'LineWidth', get_opt(opts, 'swarm_errorbar_line_width', 1.5), 'CapSize', get_opt(opts, 'swarm_errorbar_cap_size', 7));
        else
            summary_handle = hbar;
        end
    end
end
xlim(ax, [.5 numel(values_by_condition)+.5]); xticks(ax, 1:numel(values_by_condition)); xticklabels(ax, labels);
xlabel(ax, xlabel_text, 'FontSize', get_opt(opts, 'fs_labels', 13)); ylabel(ax, ylabel_text, 'FontSize', get_opt(opts, 'fs_labels', 13)); set(ax, 'YScale', yscale);
if add_legend && logical(get_opt(opts, 'wd_legend_enable', true)) && ~isempty(point_handle) && ~isempty(summary_handle)
    h = legend(ax, [point_handle summary_handle], {get_opt(opts, 'wd_ic_legend_label', 'Test IC'), get_opt(opts, 'wd_mean_legend_label', 'Mean \pm SD')}, 'Box', 'on', 'Location', get_opt(opts, 'wd_legend_location', 'northeast'), 'FontSize', get_opt(opts, 'wd_legend_font_size', 10));
    h.Color = get_opt(opts, 'wd_legend_background_color', [.90 .95 1]); h.EdgeColor = get_opt(opts, 'wd_legend_edge_color', [0 .447 .741]);
end
end

function limits = phase_limits(prepared)
values = vertcat(prepared.pred); values = values(all(isfinite(values), 2), :);
if isempty(values), limits = [-1 1 -1 1]; return; end
dx = max(range(values(:,1)), eps); dy = max(range(values(:,2)), eps);
limits = [min(values(:,1))-.05*dx max(values(:,1))+.05*dx min(values(:,2))-.05*dy max(values(:,2))+.05*dy];
end

function add_true_phase_inset(main_ax, truth, limits, opts)
fig = ancestor(main_ax, 'figure'); p = main_ax.Position; w = p(3)*get_opt(opts, 'phase_true_inset_width_fraction', .32); h = p(4)*get_opt(opts, 'phase_true_inset_height_fraction', .32);
inset_x = p(1) + p(3) * get_opt(opts, 'phase_true_inset_left_offset_fraction', .05);
ax = axes(fig, 'Units', 'normalized', 'Position', [inset_x p(2)+p(4)-h w h]);
plot(ax, truth(:,1), truth(:,2), '-', 'Color', get_opt(opts, 'true_color', [.866 .329 0]), 'LineWidth', get_opt(opts, 'phase_true_inset_line_width', .6));
box(ax, 'on'); axis(ax, 'square'); xlim(ax, limits(1:2)); ylim(ax, limits(3:4)); set(ax, 'XTick', [], 'YTick', [], 'Color', 'w'); set_fixed_outer(ax); ax.Position = [inset_x p(2)+p(4)-h w h];
end

function add_network_true_legend(fig, pos, opts)
ax = axes(fig, 'Units', 'normalized', 'Position', pos, 'Visible', 'off'); hold(ax, 'on');
h1 = plot(ax, NaN, NaN, '-', 'Color', get_opt(opts, 'network_color', [0 0 0]), 'LineWidth', 1.1);
h2 = plot(ax, NaN, NaN, '-', 'Color', get_opt(opts, 'true_color', [.866 .329 0]), 'LineWidth', 1.1);
legend(ax, [h1 h2], {get_opt(opts, 'network_legend_label', 'Network output'), get_opt(opts, 'true_legend_label', 'True system')}, 'Box', 'off', 'Location', get_opt(opts, 'legend_location', 'southwest'), 'FontSize', get_opt(opts, 'legend_font_size', 13));
end

function add_figure_title(fig, text, font_size, pos, row)
if isempty(text), return; end
left = pos{row,1}(1); last = pos{row,end}; right = last(1)+last(3); top = max(cellfun(@(p) p(2)+p(4), pos(row,:)));
annotation(fig, 'textbox', [(left+right)/2-.25 min(.95,top+.012) .5 .04], 'String', text, 'Interpreter', 'none', 'FontSize', font_size, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'EdgeColor', 'none');
end

function add_panel_letter(fig, text, position, opts)
annotation(fig, 'textbox', position, 'String', text, 'FontSize', get_opt(opts, 'combined_panel_letter_font_size', 20), 'FontWeight', 'bold', 'EdgeColor', 'none', 'HorizontalAlignment', 'left');
end

function pos = axes_grid_positions(rows, cols, left, right, bottom, top, hgap, vgap, force_square)
pos = cell(rows, cols); total_w = 1-left-right-(cols-1)*hgap; total_h = 1-top-bottom-(rows-1)*vgap; w = total_w/cols; h = total_h/rows;
if force_square, side = min(w,h); w = side; h = side; start = left+max(0,(total_w-(cols*w+(cols-1)*hgap))/2); else, start = left; end
for cc=1:cols, for rr=1:rows, pos{rr,cc} = [start+(cc-1)*(w+hgap), 1-top-rr*h-(rr-1)*vgap, w, h]; end, end
end

function labels = condition_labels(sweep, opts)
labels = get_opt(opts, 'condition_labels', {}); if numel(labels) ~= numel(sweep.conditions), labels = {sweep.conditions.label}; end
end

function label = condition_label(sweep, index, opts), labels = condition_labels(sweep, opts); label = labels{index}; end
function set_fixed_outer(ax), if isprop(ax,'ActivePositionProperty'), ax.ActivePositionProperty='Position'; end, if isprop(ax,'PositionConstraint'), ax.PositionConstraint='innerposition'; end, end
function jitter = deterministic_jitter(n,width), index=(1:n).'; jitter=width*(2*mod(sin(index*12.9898)*43758.5453,1)-1); end
function value = get_opt(S,name,default_value), if isstruct(S)&&isfield(S,name)&&~isempty(S.(name)), value=S.(name); else, value=default_value; end, end
function value = required_opt(S,name), if ~isstruct(S)||~isfield(S,name)||isempty(S.(name)), error('plot_lorenz_neuron_sweep_publication:missingSweep','opts.%s is required.',name); end, value=S.(name); end
