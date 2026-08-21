% Package orientation: Shared visualisation for rate-preserving dynamics shuffle analysis.

function figures = snn_plot_dynamics_rate_coding(result, opts)
%SNN_PLOT_DYNAMICS_RATE_CODING Plot examples and seed-level shuffle metrics.

if nargin < 2 || isempty(opts), opts = struct(); end
if ~isfield(result, 'seed_results') || isempty(result.seed_results)
    error('snn_plot_dynamics_rate_coding:noResults', 'No rate-coding analysis results were supplied.');
end
seed_results = result.seed_results; n_seed = numel(seed_results); n_cond = numel(seed_results(1).conditions);
assert_rate_preservation(seed_results);
for ii = 1:n_seed
    if isfield(seed_results(ii), 'valid_truth_ic') && any(~seed_results(ii).valid_truth_ic)
        warning('snn_plot_dynamics_rate_coding:failedTruthIcs', ...
            'Network seed %g has failed true continuations for IC(s) [%s]; corresponding WD values are infinite.', ...
            seed_results(ii).init_seed, num2str(find(~seed_results(ii).valid_truth_ic).'));
    end
end
example_seed = get_opt(opts, 'example_network_seed', seed_results(1).init_seed);
example_index = find(arrayfun(@(R) double(R.init_seed) == double(example_seed), seed_results), 1);
if isempty(example_index)
    error('snn_plot_dynamics_rate_coding:exampleSeedMissing', ...
        'Requested example_network_seed %g is not among the analysed network seeds.', example_seed);
end
example_results = seed_results(example_index);
figures = struct();
figures.phase_portraits = plot_examples(example_results, opts, 'phase');
figures.time_series = plot_examples(example_results, opts, 'time');
figures.wasserstein_swarm = plot_metric_swarm(seed_results, opts, 'wd', 'Phase-portrait WD');
figures.mse_swarm = plot_metric_swarm(seed_results, opts, 'mse', 'Decoder MSE');
fprintf(['Rate-preservation check passed: every neuron has identical spike counts within every shuffle window. ', ...
    'Displayed %d seeds x %d shuffle conditions.%s'], n_seed, n_cond, newline);
end

function fig = plot_examples(seed_results, opts, kind)
n_seed = 1; n_cond = numel(seed_results.conditions);
n_cols = n_cond + 2; fig = figure('Color', 'w');
Pos = grid_positions(n_seed + 1, n_cols, .06, .02, .07, .10, .018, .035, true);
system_name = get_opt(opts, 'system_name', 'Dynamics');
title_text = sprintf('%s Rate-Preserving Shuffle Phase Portraits', system_name);
if strcmp(kind, 'time'), title_text = sprintf('%s Rate-Preserving Shuffle Time Series', system_name); end
publication_title(fig, title_text, get_opt(opts, 'fs_titles', 14) + 3, Pos);
limits = example_limits(seed_results, kind);
for rr = 1:n_seed
    E = seed_results.example;
    draw_example(axes('Units', 'normalized', 'Position', Pos{rr, 1}), E.truth, E.dt, kind, limits, opts, 'true');
    draw_example(axes('Units', 'normalized', 'Position', Pos{rr, 2}), E.baseline, E.dt, kind, limits, opts, 'network');
    for cc = 1:n_cond
        draw_example(axes('Units', 'normalized', 'Position', Pos{rr, cc + 2}), E.shuffled{cc}, E.dt, kind, limits, opts, 'shuffle');
    end
    if rr == 1
        title_at(Pos{rr, 1}, 'True system', get_opt(opts, 'fs_titles', 14));
        title_at(Pos{rr, 2}, 'Unperturbed network', get_opt(opts, 'fs_titles', 14));
        for cc = 1:n_cond
            title_at(Pos{rr, cc + 2}, shuffle_label(seed_results.conditions(cc)), get_opt(opts, 'fs_titles', 14));
        end
    end
    annotation(fig, 'textbox', [.002 Pos{rr, 1}(2) .05 Pos{rr, 1}(4)], 'String', ...
        sprintf('Seed %g', seed_results.init_seed), 'EdgeColor', 'none', ...
        'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', 'FontSize', get_opt(opts, 'fs_ticks', 12));
end
legend_axes = axes('Units', 'normalized', 'Position', Pos{n_seed + 1, 1}, 'Visible', 'off'); hold(legend_axes, 'on');
h1 = plot(legend_axes, NaN, NaN, '-', 'Color', get_opt(opts, 'network_color', [0 0 0]), 'LineWidth', 1.1);
h2 = plot(legend_axes, NaN, NaN, '-', 'Color', get_opt(opts, 'true_color', [.866 .329 0]), 'LineWidth', 1.1);
legend(legend_axes, [h1 h2], {'Network output', 'True system'}, 'Location', 'southwest', 'Box', 'off');
end

function fig = plot_metric_swarm(seed_results, opts, field, y_label)
n_cond = numel(seed_results(1).conditions); fig = figure('Color', 'w'); ax = axes('Position', [.12 .16 .80 .72]);
hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on'); set_fixed_outer(ax); ax.FontSize = get_opt(opts, 'fs_ticks', 12);
for cc = 1:n_cond
    values = arrayfun(@(R) double(R.conditions(cc).(field)), seed_results); values = values(isfinite(values));
    scatter(ax, cc + deterministic_jitter(numel(values), .20), values, get_opt(opts, 'swarm_marker_size', 38), 'filled', ...
        'MarkerFaceColor', [0 0 0], 'MarkerEdgeColor', 'none');
    errorbar(ax, cc, mean(values), std(values, 0), 'Color', get_opt(opts, 'true_color', [.866 .329 0]), 'LineWidth', 1.5, 'CapSize', 9);
    scatter(ax, cc, mean(values), 42, 'filled', 'MarkerFaceColor', get_opt(opts, 'true_color', [.866 .329 0]), 'MarkerEdgeColor', 'none');
end
xticks(ax, 1:n_cond); xticklabels(ax, arrayfun(@(C) shuffle_label(C), seed_results(1).conditions, 'UniformOutput', false));
xlabel(ax, 'Within-window timing shuffle', 'FontSize', get_opt(opts, 'fs_labels', 13)); ylabel(ax, y_label, 'FontSize', get_opt(opts, 'fs_labels', 13));
title(ax, sprintf('%s: %s Across Network Seeds', get_opt(opts, 'system_name', 'Dynamics'), y_label), ...
    'FontSize', get_opt(opts, 'fs_titles', 14) + 3);
end

function draw_example(ax, X, dt, kind, limits, opts, mode)
network_color = get_opt(opts, 'network_color', [0 0 0]); true_color = get_opt(opts, 'true_color', [.866 .329 0]);
hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on'); set_fixed_outer(ax); ax.FontSize = get_opt(opts, 'fs_ticks', 12);
color = network_color; if strcmp(mode, 'true'), color = true_color; end
if strcmp(kind, 'phase')
    plot(ax, X(:, 1), X(:, 2), '-', 'Color', color, 'LineWidth', .85); xlabel(ax, 'x_1'); ylabel(ax, 'x_2'); axis(ax, 'square');
else
    t = (0:size(X, 1)-1).' * dt; max_time = min(get_opt(opts, 'time_window_s', 10), t(end)); keep = t <= max_time;
    plot(ax, t(keep), X(keep, 1), '-', 'Color', color, 'LineWidth', 1.0); xlabel(ax, 't (s)'); ylabel(ax, 'x_1');
end
if all(isfinite(limits)), xlim(ax, limits(1:2)); ylim(ax, limits(3:4)); end
end

function limits = example_limits(seed_results, kind)
V = [];
for rr = 1
    E = seed_results.example;
    if strcmp(kind, 'phase')
        V = [V; E.truth(:, 1:2); E.baseline(:, 1:2)]; %#ok<AGROW>
        for cc = 1:numel(E.shuffled), V = [V; E.shuffled{cc}(:, 1:2)]; end %#ok<AGROW>
    else
        t = (0:size(E.truth, 1)-1).' * E.dt;
        V = [V; [t E.truth(:, 1)]; [t E.baseline(:, 1)]]; %#ok<AGROW>
        for cc = 1:numel(E.shuffled), V = [V; [t E.shuffled{cc}(:, 1)]]; end %#ok<AGROW>
    end
end
x = V(:, 1); y = V(:, 2); dx = max(max(x)-min(x), eps); dy = max(max(y)-min(y), eps);
limits = [min(x)-.05*dx max(x)+.05*dx min(y)-.05*dy max(y)+.05*dy];
end

function label = shuffle_label(condition)
if condition.window_steps <= 1, label = 'No shuffle'; else, label = sprintf('W = %.0f ms', 1e3 * condition.window_s); end
end

function title_at(position, text, font_size)
annotation('textbox', [position(1), position(2) + position(4) + .003, position(3), .025], ...
    'String', text, 'FontSize', font_size, 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', 'EdgeColor', 'none', 'FitBoxToText', 'off');
end

function publication_title(fig, text, font_size, Pos)
left = Pos{1, 1}(1); right_pos = Pos{1, end}; right = right_pos(1)+right_pos(3); width = min(.8, max(.3, right-left));
annotation(fig, 'textbox', [(left+right)/2-width/2 .95 width .04], 'String', text, 'FontSize', font_size, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'EdgeColor', 'none', 'FitBoxToText', 'off');
end

function Pos = grid_positions(rows, cols, left, right, bottom, top, hgap, vgap, square)
Pos = cell(rows, cols); total_w = 1-left-right-(cols-1)*hgap; total_h = 1-top-bottom-(rows-1)*vgap; w = total_w/cols; h = total_h/rows;
if square, side = min(w,h); w = side; h = side; left = left + max(0, (total_w-(cols*w+(cols-1)*hgap))/2); end
for cc = 1:cols, for rr = 1:rows, Pos{rr,cc} = [left+(cc-1)*(w+hgap) 1-top-rr*h-(rr-1)*vgap w h]; end, end
end

function assert_rate_preservation(seed_results)
for rr = 1:numel(seed_results)
    errors = [seed_results(rr).conditions.max_count_error];
    if any(errors ~= 0), error('snn_plot_dynamics_rate_coding:countMismatch', 'Rate-preserving shuffle count check failed.'); end
end
end

function set_fixed_outer(ax)
if isprop(ax, 'ActivePositionProperty'), ax.ActivePositionProperty = 'Position'; end
if isprop(ax, 'PositionConstraint'), ax.PositionConstraint = 'innerposition'; end
end

function jitter = deterministic_jitter(n, width)
index = (1:n).'; jitter = width * (2 * mod(sin(index*12.9898)*43758.5453, 1) - 1);
end

function value = get_opt(S, name, default_value)
if isstruct(S) && isfield(S, name) && ~isempty(S.(name)), value = S.(name); else, value = default_value; end
end
