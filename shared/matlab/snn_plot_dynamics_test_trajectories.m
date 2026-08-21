% Package orientation: Shared MATLAB utility used across tasks. Read this with the caller open so input/output structs and saved-result fields are clear.

function snn_plot_dynamics_test_trajectories(result, output_dir)
%SNN_PLOT_DYNAMICS_TEST_TRAJECTORIES Plot closed-loop DS test trajectories.
%   SNN_PLOT_DYNAMICS_TEST_TRAJECTORIES(RESULT) displays the true and network
%   closed-loop trajectories returned by snn_primary_api('test_dynamics', ...).
%   Each seed and initial condition gets a time-series figure and a
%   phase-portrait figure containing every 2-D state pair. Trajectories are
%   shown in the normalized coordinates used for training and closed-loop
%   network prediction.

if nargin < 2 || isempty(output_dir)
    output_dir = '';
end
if ~isempty(output_dir) && exist(output_dir, 'dir') ~= 7
    mkdir(output_dir);
end

[plot_cases] = dynamics_test_plot_cases(result);
if isempty(plot_cases)
    warning('snn_plot_dynamics_test_trajectories:noTest', ...
        'No dynamical-system test result was available for trajectory plotting.');
    return;
end

plotted_any = false;
for cc = 1:numel(plot_cases)
    test_result = plot_cases(cc).test;
    opts = plot_cases(cc).options;
    [pred_by_ic, true_by_ic] = dynamics_prediction_cells(test_result);
    if isempty(pred_by_ic) || isempty(true_by_ic)
        continue;
    end

    dt = double(get_opt_local(opts, 'dt', 1));
    system_name = char(get_opt_local(opts, 'system_name', 'dynamical system'));
    units_label = 'normalized state';
    n_ic = min(numel(pred_by_ic), numel(true_by_ic));

    for ic = 1:n_ic
        pred = double(pred_by_ic{ic});
        truth = double(true_by_ic{ic});
        n = min(size(pred, 1), size(truth, 1));
        if n == 0
            continue;
        end
        pred = double(pred(1:n, :));
        truth = double(truth(1:n, :));
        n_state = min(size(pred, 2), size(truth, 2));
        if n_state == 0
            continue;
        end
        pred = pred(:, 1:n_state);
        truth = truth(:, 1:n_state);
        t = (0:n-1).' * dt;
        plot_dynamics_time_series(t, truth, pred, system_name, units_label, ...
            ic, output_dir, plot_cases(cc).title_label, plot_cases(cc).file_suffix);
        plot_dynamics_phase_portraits(truth, pred, system_name, units_label, ...
            ic, output_dir, plot_cases(cc).title_label, plot_cases(cc).file_suffix);
        plotted_any = true;
    end
end

if ~plotted_any
    warning('snn_plot_dynamics_test_trajectories:noTrajectories', ...
        'The test result does not contain pred_norm_by_ic/true_norm_by_ic trajectories.');
end
end

function plot_cases = dynamics_test_plot_cases(result)
plot_cases = struct('test', {}, 'options', {}, 'title_label', {}, 'file_suffix', {});
if isstruct(result) && isfield(result, 'seed_results') && ~isempty(result.seed_results)
    seed_results = result.seed_results;
    for ii = 1:numel(seed_results)
        R = seed_results(ii);
        if ~isfield(R, 'test') || ~isstruct(R.test)
            continue;
        end
        opts = struct();
        if isfield(R, 'options') && isstruct(R.options)
            opts = R.options;
        end
        seed_value = seed_value_for_case(result, R, ii);
        plot_cases(end + 1) = struct( ... %#ok<AGROW>
            'test', R.test, ...
            'options', opts, ...
            'title_label', seed_title_label(seed_value, ii), ...
            'file_suffix', seed_file_suffix(seed_value, ii));
    end
    return;
end
if isstruct(result) && isfield(result, 'test') && isstruct(result.test)
    opts = struct();
    if isfield(result, 'options') && isstruct(result.options)
        opts = result.options;
    end
    plot_cases = struct('test', result.test, 'options', opts, ...
        'title_label', '', 'file_suffix', '');
end
end

function [pred_by_ic, true_by_ic] = dynamics_prediction_cells(test_result)
pred_by_ic = {};
true_by_ic = {};
if isfield(test_result, 'pred_norm_by_ic') && isfield(test_result, 'true_norm_by_ic')
    pred_by_ic = force_cell_column(test_result.pred_norm_by_ic);
    true_by_ic = force_cell_column(test_result.true_norm_by_ic);
elseif isfield(test_result, 'pred_norm') && isfield(test_result, 'true_norm')
    pred_by_ic = {test_result.pred_norm};
    true_by_ic = {test_result.true_norm};
end
end

function C = force_cell_column(x)
if iscell(x)
    C = x(:);
else
    C = {x};
end
end

function plot_dynamics_time_series(t, truth, pred, system_name, units_label, ic, output_dir, seed_label, file_suffix)
n_state = size(truth, 2);
fig = figure('Color', 'w');
tiledlayout_compat(n_state, 1);
for dd = 1:n_state
    nexttile_compat();
    plot(t, pred(:,dd), 'r--', 'LineWidth', 1.2);
    hold on;
    plot(t, truth(:,dd), 'k-', 'LineWidth', 1.3);
    hold off;
    grid on;
    ylabel(sprintf('x_%d', dd));
    if dd == 1
        title(sprintf('%s closed-loop test time series%s, IC %d (%s)', ...
            system_name, seed_label, ic, units_label));
        legend({'Network', 'True system'}, 'Location', 'best');
    end
    if dd == n_state
        xlabel('Time');
    end
end
save_plot(fig, output_dir, sprintf('dynamics_time_series%s_ic%02d.png', file_suffix, ic));
end

function plot_dynamics_phase_portraits(truth, pred, system_name, units_label, ic, output_dir, seed_label, file_suffix)
n_state = size(truth, 2);
if n_state < 2
    return;
end
pairs = nchoosek(1:n_state, 2);
n_pairs = size(pairs, 1);
fig = figure('Color', 'w');
tiledlayout_compat(n_pairs, 2);
for pp = 1:n_pairs
    a = pairs(pp, 1);
    b = pairs(pp, 2);
    lims = phase_axis_limits(pred(:, [a b]), truth(:, [a b]));

    nexttile_compat();
    plot(pred(:,a), pred(:,b), 'r--', 'LineWidth', 1.1);
    axis equal;
    apply_phase_limits(lims);
    grid on;
    xlabel(sprintf('x_%d', a));
    ylabel(sprintf('x_%d', b));
    title(sprintf('Network: x_%d vs x_%d', a, b));

    nexttile_compat();
    plot(truth(:,a), truth(:,b), 'k-', 'LineWidth', 1.2);
    axis equal;
    apply_phase_limits(lims);
    grid on;
    xlabel(sprintf('x_%d', a));
    ylabel(sprintf('x_%d', b));
    title(sprintf('True system: x_%d vs x_%d', a, b));
end
sgtitle_compat(sprintf('%s closed-loop phase portraits%s, IC %d (%s)', ...
    system_name, seed_label, ic, units_label));
save_plot(fig, output_dir, sprintf('dynamics_phase_portraits%s_ic%02d.png', file_suffix, ic));
end

function lims = phase_axis_limits(pred_pair, truth_pair)
X = [pred_pair; truth_pair];
lims = nan(1, 4);
for dd = 1:2
    vals = X(:, dd);
    vals = vals(isfinite(vals));
    if isempty(vals)
        lims(2 * dd - 1:2 * dd) = [-1 1];
    else
        lo = min(vals);
        hi = max(vals);
        if lo == hi
            pad = max(1, abs(lo)) * 0.05;
        else
            pad = (hi - lo) * 0.05;
        end
        lims(2 * dd - 1:2 * dd) = [lo - pad, hi + pad];
    end
end
end

function apply_phase_limits(lims)
if all(isfinite(lims))
    xlim(lims(1:2));
    ylim(lims(3:4));
end
end

function value = get_opt_local(opts, name, default_value)
if isstruct(opts) && isfield(opts, name) && ~isempty(opts.(name))
    value = opts.(name);
else
    value = default_value;
end
end

function seed_value = seed_value_for_case(result, R, index)
seed_value = NaN;
if isfield(R, 'init_seed') && ~isempty(R.init_seed)
    seed_value = double(R.init_seed);
elseif isfield(R, 'options') && isstruct(R.options) && isfield(R.options, 'init_seed') && ~isempty(R.options.init_seed)
    seed_value = double(R.options.init_seed);
elseif isfield(R, 'options') && isstruct(R.options) && isfield(R.options, 'seed') && ~isempty(R.options.seed)
    seed_value = double(R.options.seed);
elseif isfield(result, 'seed_list') && numel(result.seed_list) >= index
    seed_value = double(result.seed_list(index));
end
end

function label = seed_title_label(seed_value, index)
if isfinite(seed_value)
    label = sprintf(', seed %g', seed_value);
else
    label = sprintf(', seed index %d', index);
end
end

function suffix = seed_file_suffix(seed_value, index)
if isfinite(seed_value)
    if abs(seed_value - round(seed_value)) < eps(max(1, abs(seed_value)))
        suffix = sprintf('_seed%03d', round(seed_value));
    else
        suffix = ['_seed' regexprep(sprintf('%.12g', seed_value), '[^A-Za-z0-9]+', '_')];
    end
else
    suffix = sprintf('_seed_index%02d', index);
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
