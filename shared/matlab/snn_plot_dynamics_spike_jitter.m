% Package orientation: Optional visualisation for the separate post-hoc
% Gaussian-jitter diagnostic; publication timing figures use snn_plot_dynamics_rate_coding.

function snn_plot_dynamics_spike_jitter(result)
%SNN_PLOT_DYNAMICS_SPIKE_JITTER Plot baseline and jittered decoder trajectories.

if ~isfield(result, 'seed_results') || isempty(result.seed_results)
    error('snn_plot_dynamics_spike_jitter:noResults', 'No jitter analysis results were supplied.');
end
for ss = 1:numel(result.seed_results)
    R = result.seed_results(ss);
    dt = double(get_opt_local(R.options, 'dt', 1));
    system_name = char(get_opt_local(R.options, 'system_name', 'dynamical system'));
    for ic = 1:numel(R.closed_loop_cases)
        C = R.closed_loop_cases(ic);
        if isempty(C.baseline_pred_norm) || isempty(C.true_norm)
            continue;
        end
        n = min(size(C.baseline_pred_norm, 1), size(C.true_norm, 1));
        baseline = double(C.baseline_pred_norm(1:n, :));
        truth = double(C.true_norm(1:n, :));
        t = (0:n - 1).' * dt;
        plot_time_series(t, truth, baseline, C.jitter, system_name, R.init_seed, ic);
        if size(truth, 2) >= 2
            plot_phase_portraits(truth, baseline, C.jitter, system_name, R.init_seed, ic);
        end
    end
end
end

function plot_time_series(t, truth, baseline, jitter, system_name, seed, ic)
n_rows = size(truth, 2);
fig = figure('Color', 'w');
tiledlayout(n_rows, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
colors = lines(max(1, numel(jitter)));
for dd = 1:n_rows
    nexttile;
    plot(t, baseline(:, dd), 'k-', 'LineWidth', 1.0); hold on;
    plot(t, truth(:, dd), 'Color', [0.85 0.33 0], 'LineWidth', 1.1);
    labels = {'Baseline network', 'True system'};
    for jj = 1:numel(jitter)
        X = double(jitter(jj).pred_norm);
        if ~isempty(X)
            plot(t(1:min(numel(t), size(X, 1))), X(1:min(numel(t), size(X, 1)), dd), ...
                '-', 'Color', colors(jj, :), 'LineWidth', 0.8);
            labels{end + 1} = sprintf('Jitter %.3g ms', 1e3 * jitter(jj).std_s); %#ok<AGROW>
        end
    end
    grid on; ylabel(sprintf('x_%d', dd));
    if dd == 1
        title(sprintf('%s post-hoc spike-time jitter, seed %g, IC %d', system_name, seed, ic));
        legend(labels, 'Location', 'best');
    end
    if dd == n_rows, xlabel('Time (s)'); end
end
drawnow;
end

function plot_phase_portraits(truth, baseline, jitter, system_name, seed, ic)
n_cols = 2 + numel(jitter);
fig = figure('Color', 'w');
tiledlayout(1, n_cols, 'TileSpacing', 'compact', 'Padding', 'compact');
plot_phase(nexttile, truth, [0.85 0.33 0], 'True system');
plot_phase(nexttile, baseline, [0 0 0], 'Baseline network');
colors = lines(max(1, numel(jitter)));
for jj = 1:numel(jitter)
    X = double(jitter(jj).pred_norm);
    if isempty(X), X = nan(1, size(truth, 2)); end
    plot_phase(nexttile, X, colors(jj, :), sprintf('Jitter %.3g ms', 1e3 * jitter(jj).std_s));
end
sgtitle(sprintf('%s post-hoc spike-time jitter phase portraits, seed %g, IC %d', system_name, seed, ic));
drawnow;
end

function plot_phase(ax, X, color, title_text)
plot(ax, X(:, 1), X(:, 2), '-', 'Color', color, 'LineWidth', 0.9);
grid(ax, 'on'); axis(ax, 'equal'); xlabel(ax, 'x_1'); ylabel(ax, 'x_2'); title(ax, title_text);
end

function value = get_opt_local(S, name, default_value)
if isstruct(S) && isfield(S, name) && ~isempty(S.(name))
    value = S.(name);
else
    value = default_value;
end
end
