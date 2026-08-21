function plot_closed_loop_validation_trajectories(closed, opts, epoch)
%PLOT_CLOSED_LOOP_VALIDATION_TRAJECTORIES Live DS validation trajectory plots.
%   This helper is called only after the normal closed-loop validation has
%   already been computed. When opts.closed_loop_validation_plot.enable is false,
%   the function returns immediately and performs no trajectory conversion,
%   figure updates, or other avoidable work. Validation trajectories are plotted
%   in the normalized coordinates used by dynamics training and prediction.
%
%   Expected option fields:
%     enable                 Toggle trajectory/phase-portrait validation plots.
%     every                  Plot every N closed-loop validation events.
%     max_initial_conditions Maximum validation ICs to display.
%     max_points             Maximum time points drawn per trace after decimation.

cfg = local_field(opts, 'closed_loop_validation_plot', struct('enable', false));
if ~isstruct(cfg) || ~local_field(cfg, 'enable', false)
    return;
end

plot_every = max(1, round(double(local_field(cfg, 'every', 1))));
validation_event_index = closed_loop_validation_event_index(epoch, opts);
if mod(validation_event_index - 1, plot_every) ~= 0
    return;
end

[pred_by_ic, true_by_ic] = trajectory_cells(closed);
if isempty(pred_by_ic) || isempty(true_by_ic)
    warning('plot_closed_loop_validation_trajectories:noTrajectories', ...
        'Closed-loop validation plotting is enabled, but validation trajectories were not available.');
    return;
end

max_ic = local_field(cfg, 'max_initial_conditions', numel(pred_by_ic));
if isinf(max_ic)
    n_ic = min(numel(pred_by_ic), numel(true_by_ic));
else
    n_ic = min([numel(pred_by_ic), numel(true_by_ic), max(1, round(double(max_ic)))]);
end
max_points = max(10, round(double(local_field(cfg, 'max_points', 2000))));
dt = double(local_field(opts, 'dt', 1));
system_name = char(local_field(opts, 'system_name', 'dynamical system'));
units_label = 'normalized state';

for ic = 1:n_ic
    pred = double(pred_by_ic{ic});
    truth = double(true_by_ic{ic});
    [t, truth, pred] = align_and_decimate_trajectories(truth, pred, dt, max_points);
    if isempty(t)
        continue;
    end
    plot_validation_time_series(t, truth, pred, system_name, units_label, epoch, ic);
    plot_validation_phase_portraits(truth, pred, system_name, units_label, epoch, ic);
end
drawnow limitrate;
end

function [pred_by_ic, true_by_ic] = trajectory_cells(closed)
pred_by_ic = {};
true_by_ic = {};
if isstruct(closed) && isfield(closed, 'pred_norm_by_ic') && isfield(closed, 'true_norm_by_ic')
    pred_by_ic = force_cell(closed.pred_norm_by_ic);
    true_by_ic = force_cell(closed.true_norm_by_ic);
elseif isstruct(closed) && isfield(closed, 'pred_norm') && isfield(closed, 'true_norm')
    pred_by_ic = {closed.pred_norm};
    true_by_ic = {closed.true_norm};
end
end

function C = force_cell(x)
if iscell(x)
    C = x(:);
else
    C = {x};
end
end

function [t, truth, pred] = align_and_decimate_trajectories(truth, pred, dt, max_points)
n = min(size(truth, 1), size(pred, 1));
n_state = min(size(truth, 2), size(pred, 2));
if n == 0 || n_state == 0
    t = [];
    truth = [];
    pred = [];
    return;
end
stride = max(1, ceil(n / max_points));
idx = 1:stride:n;
truth = double(truth(idx, 1:n_state));
pred = double(pred(idx, 1:n_state));
t = (double(idx(:)) - 1) * dt;
end

function plot_validation_time_series(t, truth, pred, system_name, units_label, epoch, ic)
n_state = size(truth, 2);
fig = validation_figure(sprintf('time_%d', ic));
clf(fig);
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
        title(sprintf('%s closed-loop validation time series, epoch %d, IC %d (%s)', ...
            system_name, epoch, ic, units_label));
        legend({'Network', 'True system'}, 'Location', 'best');
    end
    if dd == n_state
        xlabel('Time');
    end
end
end

function plot_validation_phase_portraits(truth, pred, system_name, units_label, epoch, ic)
n_state = size(truth, 2);
if n_state < 2
    return;
end
pairs = nchoosek(1:n_state, 2);
n_pairs = size(pairs, 1);
fig = validation_figure(sprintf('phase_%d', ic));
clf(fig);
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
sgtitle_compat(sprintf('%s closed-loop validation phase portraits, epoch %d, IC %d (%s)', ...
    system_name, epoch, ic, units_label));
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

function fig = validation_figure(key)
persistent figures
if isempty(figures)
    figures = containers.Map('KeyType', 'char', 'ValueType', 'any');
end
if isKey(figures, key) && isvalid(figures(key))
    fig = figures(key);
else
    fig = figure('Color', 'w');
    figures(key) = fig;
end
figure(fig);
end

function value = local_field(s, name, default_value)
if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    value = s.(name);
else
    value = default_value;
end
end

function idx = closed_loop_validation_event_index(epoch, opts)
validate_every = double(local_field(opts, 'closed_loop_validate_every', inf));
if ~isfinite(validate_every) || validate_every <= 0
    idx = max(1, round(double(epoch)));
elseif double(epoch) <= 1
    idx = 1;
else
    idx = 1 + floor(double(epoch) / validate_every);
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
