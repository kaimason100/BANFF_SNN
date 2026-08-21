function snn_plot_training_result(result, output_dir)
%SNN_PLOT_TRAINING_RESULT Plot training and validation curves.
%   The function accepts result structs returned by snn_primary_api training
%   actions. Figures are displayed and also saved when OUTPUT_DIR is given.

if nargin < 2 || isempty(output_dir), output_dir = ''; end
if ~isempty(output_dir) && exist(output_dir, 'dir') ~= 7, mkdir(output_dir); end

if isfield(result, 'history') && isstruct(result.history)
    plot_static_history(result, output_dir);
elseif isfield(result, 'history')
    plot_dynamics_history(result, output_dir);
else
    warning('snn_plot_training_result:noHistory', 'Result has no history field to plot.');
end
end

function plot_static_history(result, output_dir)
H = result.history;
ep = (1:numel(H.train_loss)).';
fig = figure('Color', 'w');
tiledlayout_compat(1, 2);
nexttile_compat();
plot(ep, H.train_loss, 'LineWidth', 1.4); hold on;
plot(ep, H.val_loss, 'LineWidth', 1.4);
xlabel('Epoch'); ylabel('Loss'); title('Training and Validation Loss'); grid on;
set_loss_axis_log(gca);
legend({'Train','Validation'}, 'Location', 'best');
nexttile_compat();
plot(ep, H.train_metric, 'LineWidth', 1.4); hold on;
plot(ep, H.val_metric, 'LineWidth', 1.4);
metric_name = metric_label(result);
xlabel('Epoch'); ylabel(metric_name); title(metric_name); grid on;
legend({'Train','Validation'}, 'Location', 'best');
save_plot(fig, output_dir, 'training_curves.png');
end

function plot_dynamics_history(result, output_dir)
loss = result.history(:);
ep = (1:numel(loss)).';
fig = figure('Color', 'w');
if isfield(result, 'closed_loop_validation')
    tiledlayout_compat(1, 2);
    nexttile_compat();
    plot(ep, loss, 'LineWidth', 1.4);
    xlabel('Epoch'); ylabel('Mean trajectory loss'); title('Training Loss'); grid on;
    set_loss_axis_log(gca);
    nexttile_compat();
    plot_finite_history(ep, result.closed_loop_validation.wd, 1.4);
    xlabel('Epoch'); ylabel('Closed-loop WD'); title('Closed-Loop Validation WD'); grid on;
    set_loss_axis_log(gca);
else
    plot(ep, loss, 'LineWidth', 1.4);
    xlabel('Epoch'); ylabel('Mean trajectory loss'); title('Dynamical-System Training Loss'); grid on;
    set_loss_axis_log(gca);
end
save_plot(fig, output_dir, 'training_loss.png');
end

function label = metric_label(result)
domain = "";
if isfield(result, 'domain')
    domain = lower(string(result.domain));
end
if domain == "classification"
    label = 'Accuracy [%]';
elseif domain == "regression"
    label = 'Pearson r';
else
    label = 'Training metric';
end
end

function save_plot(fig, output_dir, filename)
drawnow;
if isempty(output_dir), return; end
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

function plot_finite_history(epochs, values, line_width)
mask = isfinite(values);
if any(mask)
    plot(epochs(mask), double(values(mask)), 'LineWidth', line_width);
end
end

function set_loss_axis_log(ax)
%SET_LOSS_AXIS_LOG Use a log y-axis when the plotted loss is positive.
%   Non-positive or entirely missing histories are left linear because MATLAB
%   cannot display those values on a logarithmic axis.
ys = [];
kids = get(ax, 'Children');
for ii = 1:numel(kids)
    if isprop(kids(ii), 'YData')
        ys = [ys; double(kids(ii).YData(:))]; %#ok<AGROW>
    end
end
ys = ys(isfinite(ys) & ys > 0);
if ~isempty(ys)
    set(ax, 'YScale', 'log');
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
if isempty(p_use_tiles), p_use_tiles = false; p_m = 1; p_n = 1; p_idx = 0; end
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
