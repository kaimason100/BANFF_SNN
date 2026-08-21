function figures = plot_dynamics_bias_gain_test(result, opts)
%PLOT_DYNAMICS_BIAS_GAIN_TEST Plot gain-wise trajectories for every seed.

if nargin < 2 || isempty(opts), opts = struct(); end
opts = fill_plot_defaults(opts);

validate_result(result);
validate_plot_options(opts);
figures = struct('phase_portraits', {{}}, 'time_series', {{}});

for tt = 1:numel(result.systems)
    system_result = result.systems(tt);
    for ss = 1:numel(system_result.seed_results)
        seed_result = system_result.seed_results(ss);
        n_available_ic = available_ic_count(seed_result);
        plot_ics = unique(double(opts.plot_ic_indices(:).'), 'stable');
        if any(plot_ics < 1 | plot_ics > n_available_ic | plot_ics ~= round(plot_ics))
            error('plot_dynamics_bias_gain_test:plotICs', ...
                '%s seed %d has %d saved test ICs; plot_ic_indices must be integers in that range.', ...
                system_result.label, seed_result.init_seed, n_available_ic);
        end

        for ic = plot_ics
            phase_fig = plot_phase_figure(system_result, seed_result, ic, opts);
            time_fig = plot_time_figure(system_result, seed_result, ic, opts);
            figures.phase_portraits{end + 1, 1} = phase_fig;
            figures.time_series{end + 1, 1} = time_fig;
        end
    end
end

function opts = fill_plot_defaults(opts)
if ~isstruct(opts)
    error('plot_dynamics_bias_gain_test:options', 'Plot options must be a struct.');
end
opts = set_default(opts, 'plot_ic_indices', 1);
opts = set_default(opts, 'phase_subsample', 5);
opts = set_default(opts, 'time_subsample', 5);
opts = set_default(opts, 'time_window', [0 Inf]);
opts = set_default(opts, 'true_reference_gain', 1);
opts = set_default(opts, 'network_color', [0 0 0]);
opts = set_default(opts, 'true_color', [0.90 0.30 0.00]);
opts = set_default(opts, 'network_line_width', 0.75);
opts = set_default(opts, 'true_line_width', 1.0);
opts = set_default(opts, 'font_name', 'Arial');
opts = set_default(opts, 'font_size', 10);
opts = set_default(opts, 'legend_location', 'best');
opts = set_default(opts, 'show_legend', true);
opts = set_default(opts, 'phase_figure_position', [0.02 0.08 0.96 0.82]);
opts = set_default(opts, 'time_figure_position', [0.02 0.08 0.96 0.82]);
end

function opts = set_default(opts, name, value)
if ~isfield(opts, name) || isempty(opts.(name))
    opts.(name) = value;
end
end
end

function fig = plot_phase_figure(system_result, seed_result, ic, opts)
gain_results = seed_result.gain_results;
[pred0, truth0] = trajectories_for_ic(gain_results(1).test, ic);
n_state = min(size(pred0, 2), size(truth0, 2));
pairs = phase_portrait_pairs(n_state);
n_pairs = size(pairs, 1);
n_gains = numel(gain_results);
true_reference_index = reference_gain_index(gain_results, opts.true_reference_gain);
limits = phase_limits_by_pair(gain_results, ic, pairs, true_reference_index);

fig = figure('Color', 'w', 'Units', 'normalized', 'Position', opts.phase_figure_position);
layout = tiledlayout(n_pairs, n_gains + 1, 'TileSpacing', 'compact', 'Padding', 'compact');
for pp = 1:n_pairs
    for gg = 1:n_gains
        [pred, truth] = trajectories_for_ic(gain_results(gg).test, ic);
        dims = pairs(pp, :);
        pred_idx = 1:opts.phase_subsample:size(pred, 1);

        network_ax = nexttile(layout, (pp - 1) * (n_gains + 1) + gg);
        network_line = plot(network_ax, pred(pred_idx, dims(1)), pred(pred_idx, dims(2)), ...
            '-', 'Color', opts.network_color, 'LineWidth', opts.network_line_width);
        format_phase_axes(network_ax, dims, limits(pp, :), opts);
        if pp == 1
            title(network_ax, sprintf('Gain = %.4g: Network output', gain_results(gg).gain));
        end

        if opts.show_legend && pp == 1 && gg == n_gains
            legend(network_ax, network_line, {'Network output'}, ...
                'Location', opts.legend_location, 'Box', 'on');
        end
    end

    [~, truth] = trajectories_for_ic(gain_results(true_reference_index).test, ic);
    dims = pairs(pp, :);
    truth_idx = 1:opts.phase_subsample:size(truth, 1);
    true_ax = nexttile(layout, pp * (n_gains + 1));
    true_line = plot(true_ax, truth(truth_idx, dims(1)), truth(truth_idx, dims(2)), ...
            '-', 'Color', opts.true_color, 'LineWidth', opts.true_line_width);
    format_phase_axes(true_ax, dims, limits(pp, :), opts);
    if pp == 1
        title(true_ax, sprintf('True system: Gain = %.4g', gain_results(true_reference_index).gain));
    end
    if opts.show_legend && pp == 1
        legend(true_ax, true_line, {'True system'}, ...
            'Location', opts.legend_location, 'Box', 'on');
    end
end
title(layout, sprintf('%s phase portraits, network seed %d, test IC %d', ...
    system_result.label, seed_result.init_seed, ic), ...
    'FontName', opts.font_name, 'FontSize', opts.font_size + 2, 'FontWeight', 'bold');
end

function fig = plot_time_figure(system_result, seed_result, ic, opts)
gain_results = seed_result.gain_results;
[pred0, truth0] = trajectories_for_ic(gain_results(1).test, ic);
n_state = min(size(pred0, 2), size(truth0, 2));
n_gains = numel(gain_results);
true_reference_index = reference_gain_index(gain_results, opts.true_reference_gain);
limits = time_limits_by_state(gain_results, ic, n_state, opts.time_window, true_reference_index);

fig = figure('Color', 'w', 'Units', 'normalized', 'Position', opts.time_figure_position);
layout = tiledlayout(n_state, n_gains + 1, 'TileSpacing', 'compact', 'Padding', 'compact');
for dd = 1:n_state
    for gg = 1:n_gains
        [pred, truth] = trajectories_for_ic(gain_results(gg).test, ic);
        dt = double(gain_results(gg).options.dt);
        n = min(size(pred, 1), size(truth, 1));
        t = (0:n-1).' * dt;
        keep = time_selection(t, opts.time_window);
        keep = keep(1:opts.time_subsample:end);

        network_ax = nexttile(layout, (dd - 1) * (n_gains + 1) + gg);
        network_line = plot(network_ax, t(keep), pred(keep, dd), '-', ...
            'Color', opts.network_color, 'LineWidth', opts.network_line_width);
        format_time_axes(network_ax, dd, t, opts.time_window, limits(dd, :), opts, dd == n_state);
        if dd == 1
            title(network_ax, sprintf('Gain = %.4g: Network output', gain_results(gg).gain));
        end

        if opts.show_legend && dd == 1 && gg == n_gains
            legend(network_ax, network_line, {'Network output'}, ...
                'Location', opts.legend_location, 'Box', 'on');
        end
    end

    [pred, truth] = trajectories_for_ic(gain_results(true_reference_index).test, ic);
    dt = double(gain_results(true_reference_index).options.dt);
    n = min(size(pred, 1), size(truth, 1));
    t = (0:n-1).' * dt;
    keep = time_selection(t, opts.time_window);
    keep = keep(1:opts.time_subsample:end);
    true_ax = nexttile(layout, dd * (n_gains + 1));
    true_line = plot(true_ax, t(keep), truth(keep, dd), '-', ...
            'Color', opts.true_color, 'LineWidth', opts.true_line_width);
    format_time_axes(true_ax, dd, t, opts.time_window, limits(dd, :), opts, dd == n_state);
    if dd == 1
        title(true_ax, sprintf('True system: Gain = %.4g', gain_results(true_reference_index).gain));
    end
    if opts.show_legend && dd == 1
        legend(true_ax, true_line, {'True system'}, ...
            'Location', opts.legend_location, 'Box', 'on');
    end
end

title(layout, sprintf('%s time series, network seed %d, test IC %d', ...
    system_result.label, seed_result.init_seed, ic), ...
    'FontName', opts.font_name, 'FontSize', opts.font_size + 2, 'FontWeight', 'bold');
end

function format_phase_axes(ax, dims, limits, opts)
grid(ax, 'on'); box(ax, 'on'); axis(ax, 'equal');
xlim(ax, limits(1:2)); ylim(ax, limits(3:4));
xlabel(ax, sprintf('x_%d', dims(1)));
ylabel(ax, sprintf('x_%d', dims(2)));
style_axes(ax, opts);
end

function format_time_axes(ax, state_index, t, time_window, limits, opts, show_xlabel)
grid(ax, 'on'); box(ax, 'on');
xlim(ax, finite_time_limits(t, time_window));
ylim(ax, limits);
ylabel(ax, sprintf('x_%d', state_index));
if show_xlabel
    xlabel(ax, 'Time');
end
style_axes(ax, opts);
end

function index = reference_gain_index(gain_results, requested_gain)
gains = [gain_results.gain];
if ~isscalar(requested_gain) || ~isfinite(requested_gain)
    error('plot_dynamics_bias_gain_test:trueReferenceGain', ...
        'true_reference_gain must be one finite scalar matching a tested gain.');
end
tolerance = 10 * eps(max(1, max(abs(gains))));
index = find(abs(gains - requested_gain) <= tolerance, 1, 'first');
if isempty(index)
    error('plot_dynamics_bias_gain_test:trueReferenceGain', ...
        'true_reference_gain %.6g is not in bias_gains.', requested_gain);
end
end

function limits = phase_limits_by_pair(gain_results, ic, pairs, true_reference_index)
limits = nan(size(pairs, 1), 4);
for pp = 1:size(pairs, 1)
    values = zeros(0, 2);
    for gg = 1:numel(gain_results)
        [pred, ~] = trajectories_for_ic(gain_results(gg).test, ic);
        values = [values; pred(:, pairs(pp, :))]; %#ok<AGROW>
    end
    [~, truth] = trajectories_for_ic(gain_results(true_reference_index).test, ic);
    values = [values; truth(:, pairs(pp, :))]; %#ok<AGROW>
    limits(pp, :) = padded_xy_limits(values);
end
end

function limits = time_limits_by_state(gain_results, ic, n_state, time_window, true_reference_index)
limits = nan(n_state, 2);
for dd = 1:n_state
    values = zeros(0, 1);
    for gg = 1:numel(gain_results)
        [pred, ~] = trajectories_for_ic(gain_results(gg).test, ic);
        dt = double(gain_results(gg).options.dt);
        n = size(pred, 1);
        t = (0:n-1).' * dt;
        keep = time_selection(t, time_window);
        values = [values; pred(keep, dd)]; %#ok<AGROW>
    end
    [pred, truth] = trajectories_for_ic(gain_results(true_reference_index).test, ic);
    dt = double(gain_results(true_reference_index).options.dt);
    n = min(size(pred, 1), size(truth, 1));
    t = (0:n-1).' * dt;
    keep = time_selection(t, time_window);
    values = [values; truth(keep, dd)]; %#ok<AGROW>
    limits(dd, :) = padded_limits(values);
end
end

function [pred, truth] = trajectories_for_ic(test, ic)
if ~isfield(test, 'pred_norm_by_ic') || ~isfield(test, 'true_norm_by_ic')
    error('plot_dynamics_bias_gain_test:missingTrajectories', ...
        'A gain result does not contain pred_norm_by_ic and true_norm_by_ic.');
end
pred = double(test.pred_norm_by_ic{ic});
truth = double(test.true_norm_by_ic{ic});
n = min(size(pred, 1), size(truth, 1));
pred = pred(1:n, :);
truth = truth(1:n, :);
end

function n = available_ic_count(seed_result)
if isempty(seed_result.gain_results)
    n = 0;
    return;
end
test = seed_result.gain_results(1).test;
n = min(numel(test.pred_norm_by_ic), numel(test.true_norm_by_ic));
end

function keep = time_selection(t, time_window)
keep = find(t >= time_window(1) & t <= time_window(2));
if isempty(keep)
    error('plot_dynamics_bias_gain_test:timeWindow', ...
        'time_window does not overlap the saved scored trajectory.');
end
end

function limits = finite_time_limits(t, time_window)
limits = [max(t(1), time_window(1)), min(t(end), time_window(2))];
if diff(limits) <= 0
    limits = [t(1), t(end)];
end
end

function limits = padded_xy_limits(values)
limits = [padded_limits(values(:, 1)), padded_limits(values(:, 2))];
end

function limits = padded_limits(values)
values = values(isfinite(values));
if isempty(values)
    limits = [-1 1];
    return;
end
lo = min(values); hi = max(values);
span = hi - lo;
if span == 0
    span = max(1, abs(lo));
end
pad = 0.04 * span;
limits = [lo - pad, hi + pad];
end

function style_axes(ax, opts)
set(ax, 'FontName', opts.font_name, 'FontSize', opts.font_size, ...
    'LineWidth', 0.75, 'Layer', 'top');
end

function validate_result(result)
if ~isfield(result, 'systems') || isempty(result.systems)
    error('plot_dynamics_bias_gain_test:result', ...
        'result.systems is required and cannot be empty.');
end
end

function validate_plot_options(opts)
if isempty(opts.plot_ic_indices) || ~isnumeric(opts.plot_ic_indices) || any(~isfinite(opts.plot_ic_indices))
    error('plot_dynamics_bias_gain_test:plotICs', ...
        'plot_ic_indices must contain at least one finite numeric index.');
end
if ~isscalar(opts.phase_subsample) || ~isfinite(opts.phase_subsample) || ...
        opts.phase_subsample < 1 || opts.phase_subsample ~= round(opts.phase_subsample)
    error('plot_dynamics_bias_gain_test:phaseSubsample', ...
        'phase_subsample must be a positive integer.');
end
if ~isscalar(opts.time_subsample) || ~isfinite(opts.time_subsample) || ...
        opts.time_subsample < 1 || opts.time_subsample ~= round(opts.time_subsample)
    error('plot_dynamics_bias_gain_test:timeSubsample', ...
        'time_subsample must be a positive integer.');
end
if numel(opts.time_window) ~= 2 || any(isnan(opts.time_window)) || ...
        opts.time_window(1) < 0 || opts.time_window(2) <= opts.time_window(1)
    error('plot_dynamics_bias_gain_test:timeWindow', ...
        'time_window must be [start end], with 0 <= start < end; end may be Inf.');
end
if numel(opts.network_color) ~= 3 || any(~isfinite(opts.network_color)) || ...
        any(opts.network_color < 0 | opts.network_color > 1)
    error('plot_dynamics_bias_gain_test:networkColor', 'network_color must be an RGB triplet in [0,1].');
end
if numel(opts.true_color) ~= 3 || any(~isfinite(opts.true_color)) || ...
        any(opts.true_color < 0 | opts.true_color > 1)
    error('plot_dynamics_bias_gain_test:trueColor', 'true_color must be an RGB triplet in [0,1].');
end
if ~isscalar(opts.network_line_width) || ~isfinite(opts.network_line_width) || opts.network_line_width <= 0 || ...
        ~isscalar(opts.true_line_width) || ~isfinite(opts.true_line_width) || opts.true_line_width <= 0
    error('plot_dynamics_bias_gain_test:lineWidth', 'Line widths must be positive finite scalars.');
end
if ~isscalar(opts.font_size) || ~isfinite(opts.font_size) || opts.font_size <= 0
    error('plot_dynamics_bias_gain_test:fontSize', 'font_size must be a positive finite scalar.');
end
if ~isscalar(opts.show_legend) || ~(islogical(opts.show_legend) || isnumeric(opts.show_legend))
    error('plot_dynamics_bias_gain_test:showLegend', 'show_legend must be a scalar logical value.');
end
validate_figure_position(opts.phase_figure_position, 'phase_figure_position');
validate_figure_position(opts.time_figure_position, 'time_figure_position');
end

function validate_figure_position(position, name)
if ~isnumeric(position) || numel(position) ~= 4 || any(~isfinite(position)) || ...
        position(3) <= 0 || position(4) <= 0
    error('plot_dynamics_bias_gain_test:figurePosition', ...
        '%s must be a finite [left bottom width height] vector with positive size.', name);
end
end
