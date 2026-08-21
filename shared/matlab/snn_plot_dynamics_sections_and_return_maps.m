% Package orientation: Shared MATLAB utility used across tasks. Read this with the caller open so input/output structs and saved-result fields are clear.

function snn_plot_dynamics_sections_and_return_maps(result, output_dir, min_peak_prominence)
%SNN_PLOT_DYNAMICS_SECTIONS_AND_RETURN_MAPS Plot DS sections and peak maps.
%   SNN_PLOT_DYNAMICS_SECTIONS_AND_RETURN_MAPS(RESULT) displays Poincare
%   sections and scalar peak return maps for every tested seed, comparing the
%   true closed-loop dynamical system and the trained network output. The plots
%   use the same saved test trajectories as SNN_PLOT_DYNAMICS_TEST_TRAJECTORIES
%   and keep them in the normalized coordinates used for training and
%   closed-loop prediction.
%
%   Return maps require MATLAB's findpeaks with MinPeakProminence fixed at 1.
%   The routine errors rather than substituting a non-equivalent detector.

if nargin < 2 || isempty(output_dir)
    output_dir = '';
end
if nargin < 3 || isempty(min_peak_prominence)
    min_peak_prominence = 1;
end
if ~isscalar(min_peak_prominence) || ~isfinite(min_peak_prominence) || min_peak_prominence ~= 1
    error('snn_plot_dynamics_sections_and_return_maps:fixedPeakProminence', ...
        'Return maps require min_peak_prominence = 1.');
end
if exist('findpeaks', 'file') ~= 2
    error('snn_plot_dynamics_sections_and_return_maps:findpeaksRequired', ...
        ['Return maps require MATLAB findpeaks with MinPeakProminence = 1. ', ...
         'Install or enable the Signal Processing Toolbox before plotting.']);
end
if ~isempty(output_dir) && exist(output_dir, 'dir') ~= 7
    mkdir(output_dir);
end

[plot_cases] = dynamics_test_plot_cases_local(result);
if isempty(plot_cases)
    warning('snn_plot_dynamics_sections_and_return_maps:noTest', ...
        'No dynamical-system test result was available for Poincare/return-map plotting.');
    return;
end

plotted_any = false;
for cc = 1:numel(plot_cases)
    test_result = plot_cases(cc).test;
    opts = plot_cases(cc).options;
    [pred_by_ic, true_by_ic] = dynamics_prediction_cells_local(test_result);
    if isempty(pred_by_ic) || isempty(true_by_ic)
        continue;
    end

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

        plot_poincare_sections_all_2d(truth, pred, system_name, units_label, ...
            ic, output_dir, plot_cases(cc).title_label, plot_cases(cc).file_suffix);
        plot_peak_return_maps(truth, pred, system_name, units_label, ...
            ic, output_dir, min_peak_prominence, plot_cases(cc).title_label, plot_cases(cc).file_suffix);
        plotted_any = true;
    end
end

if ~plotted_any
    warning('snn_plot_dynamics_sections_and_return_maps:noTrajectories', ...
        'The test result does not contain pred_norm_by_ic/true_norm_by_ic trajectories.');
end
end

function plot_cases = dynamics_test_plot_cases_local(result)
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
        seed_value = seed_value_for_case_local(result, R, ii);
        plot_cases(end + 1) = struct( ... %#ok<AGROW>
            'test', R.test, ...
            'options', opts, ...
            'title_label', seed_title_label_local(seed_value, ii), ...
            'file_suffix', seed_file_suffix_local(seed_value, ii));
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

function [pred_by_ic, true_by_ic] = dynamics_prediction_cells_local(test_result)
pred_by_ic = {};
true_by_ic = {};
if isfield(test_result, 'pred_norm_by_ic') && isfield(test_result, 'true_norm_by_ic')
    pred_by_ic = force_cell_column_local(test_result.pred_norm_by_ic);
    true_by_ic = force_cell_column_local(test_result.true_norm_by_ic);
elseif isfield(test_result, 'pred_norm') && isfield(test_result, 'true_norm')
    pred_by_ic = {test_result.pred_norm};
    true_by_ic = {test_result.true_norm};
end
end

function C = force_cell_column_local(x)
if iscell(x)
    C = x(:);
else
    C = {x};
end
end

function plot_poincare_sections_all_2d(truth, pred, system_name, units_label, ic, output_dir, seed_label, file_suffix)
n_state = size(truth, 2);
if n_state < 2
    return;
end

pairs = nchoosek(1:n_state, 2);
n_pairs = size(pairs, 1);
n_cols = min(3, n_pairs);
n_rows = ceil(n_pairs / n_cols);

for section_dim = 1:n_state
    section_value = finite_median_local(truth(:, section_dim));
    true_pts = poincare_points_local(truth, section_dim, section_value);
    pred_pts = poincare_points_local(pred, section_dim, section_value);
    if isempty(true_pts) && isempty(pred_pts)
        continue;
    end

    fig = figure('Color', 'w');
    tiledlayout_compat_local(n_rows, n_cols);
    for pp = 1:n_pairs
        a = pairs(pp, 1);
        b = pairs(pp, 2);
        nexttile_compat_local();
        legend_handles = {};
        legend_labels = {};
        if ~isempty(pred_pts)
            h_pred = plot(pred_pts(:, a), pred_pts(:, b), 'rx', ...
                'MarkerSize', 5.0, 'LineWidth', 1.1);
            hold on;
            legend_handles{end + 1} = h_pred; %#ok<AGROW>
            legend_labels{end + 1} = 'Network'; %#ok<AGROW>
        end
        if ~isempty(true_pts)
            h_true = plot(true_pts(:, a), true_pts(:, b), 'ko', ...
                'MarkerSize', 4.5, 'LineWidth', 1.0);
            hold on;
            legend_handles{end + 1} = h_true; %#ok<AGROW>
            legend_labels{end + 1} = 'True system'; %#ok<AGROW>
        end
        hold off;
        grid on;
        xlabel(sprintf('x_%d', a));
        ylabel(sprintf('x_%d', b));
        title(sprintf('x_%d vs x_%d', a, b));
        if pp == 1 && ~isempty(legend_handles)
            legend([legend_handles{:}], legend_labels, 'Location', 'best');
        end
    end
    sgtitle_compat_local(sprintf('%s Poincare section x_%d = %.3g%s, IC %d (%s)', ...
        system_name, section_dim, section_value, seed_label, ic, units_label));
    save_plot_local(fig, output_dir, ...
        sprintf('dynamics_poincare%s_x%d_ic%02d.png', file_suffix, section_dim, ic));
end
end

function pts = poincare_points_local(X, section_dim, section_value)
if isempty(X) || size(X, 1) < 2 || section_dim > size(X, 2) || ~isfinite(section_value)
    pts = zeros(0, size(X, 2));
    return;
end

s = X(:, section_dim);
cross_idx = find(s(1:end-1) < section_value & s(2:end) >= section_value);
pts = zeros(numel(cross_idx), size(X, 2));
for ii = 1:numel(cross_idx)
    k = cross_idx(ii);
    denom = s(k + 1) - s(k);
    if denom == 0 || ~isfinite(denom)
        frac = 0;
    else
        frac = (section_value - s(k)) / denom;
    end
    frac = min(max(frac, 0), 1);
    pts(ii, :) = X(k, :) + frac .* (X(k + 1, :) - X(k, :));
end
end

function value = finite_median_local(x)
x = double(x(:));
x = x(isfinite(x));
if isempty(x)
    value = NaN;
else
    value = median(x);
end
end

function plot_peak_return_maps(truth, pred, system_name, units_label, ic, output_dir, min_peak_prominence, seed_label, file_suffix)
n_state = size(truth, 2);
if n_state < 1
    return;
end

n_cols = min(3, n_state);
n_rows = ceil(n_state / n_cols);
fig = figure('Color', 'w');
tiledlayout_compat_local(n_rows, n_cols);
has_any_peak_map = false;

for dd = 1:n_state
    true_peaks = peaks_with_min_prominence_local(truth(:, dd), min_peak_prominence);
    pred_peaks = peaks_with_min_prominence_local(pred(:, dd), min_peak_prominence);

    nexttile_compat_local();
    legend_handles = {};
    legend_labels = {};
    if numel(pred_peaks) >= 2
        h_pred = plot(pred_peaks(1:end-1), pred_peaks(2:end), 'rx', ...
            'MarkerSize', 5.0, 'LineWidth', 1.1);
        hold on;
        legend_handles{end + 1} = h_pred; %#ok<AGROW>
        legend_labels{end + 1} = 'Network'; %#ok<AGROW>
        has_any_peak_map = true;
    end
    if numel(true_peaks) >= 2
        h_true = plot(true_peaks(1:end-1), true_peaks(2:end), 'ko', ...
            'MarkerSize', 4.5, 'LineWidth', 1.0);
        hold on;
        legend_handles{end + 1} = h_true; %#ok<AGROW>
        legend_labels{end + 1} = 'True system'; %#ok<AGROW>
        has_any_peak_map = true;
    end
    hold off;
    grid on;
    axis square;
    xlabel(sprintf('x_%d peak_n', dd));
    ylabel(sprintf('x_%d peak_{n+1}', dd));
    title(sprintf('x_%d return map', dd));
    if dd == 1 && ~isempty(legend_handles)
        legend([legend_handles{:}], legend_labels, 'Location', 'best');
    end
end

sgtitle_compat_local(sprintf('%s peak return maps%s, IC %d (%s, min prominence %.3g)', ...
    system_name, seed_label, ic, units_label, min_peak_prominence));
if has_any_peak_map
    save_plot_local(fig, output_dir, sprintf('dynamics_peak_return_maps%s_ic%02d.png', file_suffix, ic));
else
    drawnow;
end
end

function peaks = peaks_with_min_prominence_local(x, min_peak_prominence)
x = double(x(:));
x = x(isfinite(x));
if numel(x) < 3
    peaks = zeros(0, 1);
    return;
end

try
    peaks = findpeaks(x, 'MinPeakProminence', 1);
catch ME
    error('snn_plot_dynamics_sections_and_return_maps:findpeaksFailed', ...
        'Return maps require findpeaks(x, ''MinPeakProminence'', 1): %s', ME.message);
end
peaks = peaks(:);
end

function value = get_opt_local(opts, name, default_value)
if isstruct(opts) && isfield(opts, name) && ~isempty(opts.(name))
    value = opts.(name);
else
    value = default_value;
end
end

function seed_value = seed_value_for_case_local(result, R, index)
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

function label = seed_title_label_local(seed_value, index)
if isfinite(seed_value)
    label = sprintf(', seed %g', seed_value);
else
    label = sprintf(', seed index %d', index);
end
end

function suffix = seed_file_suffix_local(seed_value, index)
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

function save_plot_local(fig, output_dir, filename)
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

function tiledlayout_compat_local(m, n)
if exist('tiledlayout', 'file') ~= 0
    tiledlayout(m, n, 'TileSpacing', 'compact', 'Padding', 'compact');
    tile_state_local('set', true, m, n, 0);
else
    tile_state_local('set', false, m, n, 0);
end
end

function nexttile_compat_local()
[use_tiles, m, n, idx] = tile_state_local('get');
if use_tiles && exist('nexttile', 'file') ~= 0
    nexttile;
else
    idx = idx + 1;
    tile_state_local('set', false, m, n, idx);
    subplot(m, n, idx);
end
end

function sgtitle_compat_local(txt)
if exist('sgtitle', 'file') ~= 0
    sgtitle(txt);
else
    annotation('textbox', [0 0.95 1 0.04], 'String', txt, ...
        'HorizontalAlignment', 'center', 'EdgeColor', 'none');
end
end

function varargout = tile_state_local(action, use_tiles, m, n, idx)
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
