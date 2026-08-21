% Package orientation: Shared MATLAB utility used across tasks. Read this with the caller open so input/output structs and saved-result fields are clear.

function DIAG = save_spiking_diagnostics_from_raster(S, dt, task_tag, output_dir, varargin)
%SAVE_SPIKING_DIAGNOSTICS_FROM_RASTER Summarise and plot spike behaviour.
% S may be [neurons x steps] or [neurons x steps x samples]. The function
% computes exact firing-rate summaries and inverse-ISI rate diagnostics over
% the supplied rasters. Inverse-ISI rates are calculated from every
% within-sample inter-spike interval in the test simulation.

opts = parse_spike_diag_options(varargin{:});
if nargin < 4 || isempty(output_dir), output_dir = pwd; end
if nargin < 3 || isempty(task_tag), task_tag = 'spiking_test'; end
if opts.save_files && exist(output_dir, 'dir') ~= 7, mkdir(output_dir); end

S = logical(S);
dt = double(dt);
if ndims(S) < 3
    S = reshape(S, size(S,1), size(S,2), 1);
end

N = size(S, 1);
T = size(S, 2);
M = size(S, 3);
state = init_state(N, T, M, dt, opts);
for ii = 1:M
    state = update_state(state, S(:,:,ii), [], [], [], [], ii);
end
DIAG = finalise_and_save(state, task_tag, output_dir, opts);
end

function opts = parse_spike_diag_options(varargin)
opts = struct();
opts.context = '';
opts.save_files = false;
opts.max_examples = 3;
opts.max_raster_neurons = 300;
opts.max_voltage_neurons = 8;
opts.isi_edges = [0:0.001:0.050, 0.055:0.005:0.250, 0.275:0.025:1.000, inf];
opts.isi_rate_edges_hz = [0:1:50, 55:5:150, 160:10:500, 550:50:1000, inf];
opts.u_buffer = [];
opts.i_in = [];
opts.i_rec = [];
opts.w = [];
if mod(numel(varargin), 2) ~= 0
    error('Options must be name/value pairs.');
end
for ii = 1:2:numel(varargin)
    name = lower(char(varargin{ii}));
    opts.(name) = varargin{ii+1};
end
end

function state = init_state(N, T, M, dt, opts)
state = struct();
state.N = double(N);
state.T = double(T);
state.M = double(M);
state.dt = double(dt);
state.duration_s = double(T) * double(dt);
state.spike_counts = zeros(N, 1);
state.sample_population_rate_hz = nan(M, 1);
state.sample_active_fraction = nan(M, 1);
state.isi_edges = opts.isi_edges(:).';
state.isi_hist = zeros(1, numel(state.isi_edges)-1);
state.isi_rate_edges_hz = opts.isi_rate_edges_hz(:).';
state.isi_rate_hist = zeros(1, numel(state.isi_rate_edges_hz)-1);
state.isi_sum_by_neuron = zeros(N, 1);
state.isi_rate_sum_by_neuron = zeros(N, 1);
state.isi_count_by_neuron = zeros(N, 1);
state.isi_min_by_neuron = inf(N, 1);
state.isi_max_by_neuron = zeros(N, 1);
state.examples = struct('sample_index', {}, 'S', {}, 'u_buffer', {}, 'I_in', {}, 'I_rec', {}, 'w', {});
state.max_examples = double(opts.max_examples);
state.max_raster_neurons = double(opts.max_raster_neurons);
state.max_voltage_neurons = double(opts.max_voltage_neurons);
end

function state = update_state(state, S_one, u_buffer, I_in, I_rec, w, sample_index)
S_one = logical(S_one);
spike_counts_one = full(sum(S_one, 2));
state.spike_counts = state.spike_counts + double(spike_counts_one);
state.sample_population_rate_hz(sample_index) = double(sum(spike_counts_one)) / ...
    max(realmin, state.N * state.duration_s);
state.sample_active_fraction(sample_index) = mean(spike_counts_one > 0);

for nn = find(spike_counts_one(:).' > 1)
    t_sp = find(S_one(nn, :));
    isi = diff(t_sp) * state.dt;
    if isempty(isi), continue; end
    isi_rate_hz = 1 ./ isi;
    isi_rate_hz = isi_rate_hz(isfinite(isi_rate_hz) & isi_rate_hz > 0);
    state.isi_hist = state.isi_hist + histcounts_compat(isi, state.isi_edges);
    state.isi_rate_hist = state.isi_rate_hist + histcounts_compat(isi_rate_hz, state.isi_rate_edges_hz);
    state.isi_sum_by_neuron(nn) = state.isi_sum_by_neuron(nn) + sum(isi);
    state.isi_rate_sum_by_neuron(nn) = state.isi_rate_sum_by_neuron(nn) + sum(isi_rate_hz);
    state.isi_count_by_neuron(nn) = state.isi_count_by_neuron(nn) + numel(isi);
    state.isi_min_by_neuron(nn) = min(state.isi_min_by_neuron(nn), min(isi));
    state.isi_max_by_neuron(nn) = max(state.isi_max_by_neuron(nn), max(isi));
end

if numel(state.examples) < state.max_examples
    ex = struct();
    ex.sample_index = sample_index;
    ex.S = S_one;
    ex.u_buffer = u_buffer;
    ex.I_in = I_in;
    ex.I_rec = I_rec;
    ex.w = w;
    state.examples(end+1) = ex; %#ok<AGROW>
end
end

function DIAG = finalise_and_save(state, task_tag, output_dir, opts)
task_tag = safe_tag(task_tag);
if ~isempty(opts.context)
    task_tag = [task_tag '_' safe_tag(opts.context)];
end
prefix = fullfile(output_dir, ['spiking_diagnostics_' task_tag]);

rate_hz = state.spike_counts ./ max(realmin, state.M * state.duration_s);
mean_isi = state.isi_sum_by_neuron ./ max(1, state.isi_count_by_neuron);
mean_isi(state.isi_count_by_neuron == 0) = NaN;
mean_isi_rate_hz = state.isi_rate_sum_by_neuron ./ max(1, state.isi_count_by_neuron);
mean_isi_rate_hz(state.isi_count_by_neuron == 0) = NaN;
state.isi_min_by_neuron(isinf(state.isi_min_by_neuron)) = NaN;

DIAG = struct();
DIAG.task_tag = task_tag;
DIAG.n_neurons = state.N;
DIAG.n_samples = state.M;
DIAG.steps = state.T;
DIAG.dt = state.dt;
DIAG.duration_per_sample_s = state.duration_s;
DIAG.mean_rate_hz_by_neuron = rate_hz;
DIAG.population_rate_hz_by_sample = state.sample_population_rate_hz;
DIAG.active_fraction_by_sample = state.sample_active_fraction;
DIAG.active_percent_by_sample = 100 * state.sample_active_fraction;
DIAG.silent_percent_by_sample = 100 * (1 - state.sample_active_fraction);
DIAG.active_neuron_count = sum(rate_hz > 0);
DIAG.silent_neuron_count = sum(rate_hz == 0);
DIAG.active_neuron_percent = 100 * mean(rate_hz > 0);
DIAG.silent_neuron_percent = 100 * mean(rate_hz == 0);
DIAG.isi_edges_s = state.isi_edges;
DIAG.isi_hist_counts = state.isi_hist;
DIAG.isi_rate_edges_hz = state.isi_rate_edges_hz;
DIAG.isi_rate_hist_counts = state.isi_rate_hist;
DIAG.mean_isi_s_by_neuron = mean_isi;
DIAG.mean_isi_rate_hz_by_neuron = mean_isi_rate_hz;
DIAG.isi_count_by_neuron = state.isi_count_by_neuron;
DIAG.min_isi_s_by_neuron = state.isi_min_by_neuron;
DIAG.max_isi_s_by_neuron = state.isi_max_by_neuron;
DIAG.examples = state.examples;
if ~isempty(DIAG.examples)
    if ~isempty(opts.u_buffer), DIAG.examples(1).u_buffer = opts.u_buffer; end
    if ~isempty(opts.i_in), DIAG.examples(1).I_in = opts.i_in; end
    if ~isempty(opts.i_rec), DIAG.examples(1).I_rec = opts.i_rec; end
    if ~isempty(opts.w), DIAG.examples(1).w = opts.w; end
end

print_summary(DIAG);
if opts.save_files
    save([prefix '.mat'], 'DIAG', '-v7');
    write_summary([prefix '_summary.txt'], DIAG);
    write_rate_csv([prefix '_neuron_rates.csv'], DIAG);
end
plot_rate_summary([prefix '_rates.png'], DIAG, opts.save_files);
plot_isi_rate_summary([prefix '_isi_rate.png'], DIAG, opts.save_files);
plot_rate_isi_scatter([prefix '_rate_vs_isi_rate.png'], DIAG, opts.save_files);
plot_examples(prefix, DIAG, state.max_raster_neurons, state.max_voltage_neurons, opts.save_files);

if opts.save_files
    fprintf('[SPIKING] Saved diagnostics: %s_*.png and %s.mat\n', prefix, prefix);
else
    fprintf('[SPIKING] Displayed spiking diagnostics without saving files. To save files, pass ''save_files'', true.\n');
end
end

function counts = histcounts_compat(x, edges)
if exist('histcounts', 'file') == 2
    counts = histcounts(x, edges);
else
    counts = zeros(1, numel(edges)-1);
    for ii = 1:numel(counts)
        if ii == numel(counts)
            counts(ii) = sum(x >= edges(ii) & x <= edges(ii+1));
        else
            counts(ii) = sum(x >= edges(ii) & x < edges(ii+1));
        end
    end
end
end

function print_summary(D)
rate = D.mean_rate_hz_by_neuron;
active = rate > 0;
fprintf('\n=== Spiking Diagnostics: %s ===\n', D.task_tag);
fprintf('Samples: %d | Neurons: %d | Steps/sample: %d | Seconds/sample: %.6g\n', D.n_samples, D.n_neurons, D.steps, D.duration_per_sample_s);
fprintf('Mean neuron firing rate: %.6g Hz | median %.6g Hz | 95th percentile %.6g Hz\n', mean(rate), median(rate), percentile_compat(rate, 95));
fprintf('Active neurons: %d / %d (%.2f%%)\n', sum(active), numel(active), 100*mean(active));
fprintf('Silent neurons: %d / %d (%.2f%%)\n', sum(~active), numel(active), 100*mean(~active));
fprintf('Mean population rate by sample: %.6g Hz/neuron\n', mean(D.population_rate_hz_by_sample));
fprintf('Mean active neurons by sample: %.2f%% | silent %.2f%%\n', mean(D.active_percent_by_sample), mean(D.silent_percent_by_sample));
fprintf('Total within-sample ISIs counted: %d\n', sum(D.isi_hist_counts));
fprintf('Median per-neuron mean inverse-ISI rate: %.6g Hz\n', median(D.mean_isi_rate_hz_by_neuron(isfinite(D.mean_isi_rate_hz_by_neuron))));
end

function write_summary(pathstr, D)
fid = fopen(pathstr, 'w');
if fid < 0, warning('Could not write spiking summary: %s', pathstr); return; end
cleanup = onCleanup(@() fclose(fid));
rate = D.mean_rate_hz_by_neuron;
active = rate > 0;
fprintf(fid, 'Spiking diagnostics: %s\n', D.task_tag);
fprintf(fid, 'Samples: %d\nNeurons: %d\nSteps/sample: %d\nSeconds/sample: %.6g\n', ...
    D.n_samples, D.n_neurons, D.steps, D.duration_per_sample_s);
fprintf(fid, 'Mean neuron firing rate: %.6g Hz\n', mean(rate));
fprintf(fid, 'Median neuron firing rate: %.6g Hz\n', median(rate));
fprintf(fid, '95th percentile neuron firing rate: %.6g Hz\n', percentile_compat(rate, 95));
fprintf(fid, 'Active neurons: %d / %d (%.2f%%)\n', sum(active), numel(active), 100*mean(active));
fprintf(fid, 'Silent neurons: %d / %d (%.2f%%)\n', sum(~active), numel(active), 100*mean(~active));
fprintf(fid, 'Mean population rate by sample: %.6g Hz/neuron\n', mean(D.population_rate_hz_by_sample));
fprintf(fid, 'Mean active neurons by sample: %.2f%%\n', mean(D.active_percent_by_sample));
fprintf(fid, 'Mean silent neurons by sample: %.2f%%\n', mean(D.silent_percent_by_sample));
fprintf(fid, 'Total within-sample ISIs counted: %d\n', sum(D.isi_hist_counts));
fprintf(fid, 'Median per-neuron mean inverse-ISI rate: %.6g Hz\n', median(D.mean_isi_rate_hz_by_neuron(isfinite(D.mean_isi_rate_hz_by_neuron))));
end

function write_rate_csv(pathstr, D)
fid = fopen(pathstr, 'w');
if fid < 0, warning('Could not write rate CSV: %s', pathstr); return; end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'neuron,mean_rate_hz,mean_inverse_isi_rate_hz,mean_isi_s,isi_count,min_isi_s,max_isi_s\n');
for nn = 1:D.n_neurons
    fprintf(fid, '%d,%.9g,%.9g,%.9g,%d,%.9g,%.9g\n', nn, D.mean_rate_hz_by_neuron(nn), ...
        D.mean_isi_rate_hz_by_neuron(nn), D.mean_isi_s_by_neuron(nn), D.isi_count_by_neuron(nn), ...
        D.min_isi_s_by_neuron(nn), D.max_isi_s_by_neuron(nn));
end
end

function plot_rate_summary(pathstr, D, save_files)
fig = figure('Color', 'w');
tiledlayout_compat(2, 3);
nexttile_compat(); histogram(D.mean_rate_hz_by_neuron, 80); xlabel('Mean firing rate [Hz]'); ylabel('# neurons'); title('Neuron Rates');
nexttile_compat(); plot(sort(D.mean_rate_hz_by_neuron, 'descend'), 'LineWidth', 1.5); xlabel('Neuron rank'); ylabel('Mean rate [Hz]'); title('Ranked Rates'); grid on;
nexttile_compat(); histogram(D.population_rate_hz_by_sample, 50); xlabel('Population rate [Hz/neuron]'); ylabel('# samples'); title('Sample Population Rates');
nexttile_compat(); histogram(D.active_percent_by_sample, 50); xlabel('Active neurons [%]'); ylabel('# samples'); title('Active Neurons by Sample');
nexttile_compat(); histogram(D.silent_percent_by_sample, 50); xlabel('Silent neurons [%]'); ylabel('# samples'); title('Silent Neurons by Sample');
nexttile_compat();
bar([D.active_neuron_percent, D.silent_neuron_percent]);
set(gca, 'XTickLabel', {'Active','Silent'});
ylabel('Neurons [%]'); ylim([0 100]); title('Test-Set Neuron Participation'); grid on;
save_figure(fig, pathstr, save_files);
end

function plot_isi_rate_summary(pathstr, D, save_files)
edges = D.isi_rate_edges_hz;
centers = edges(1:end-1) + diff(edges)/2;
centers(~isfinite(centers)) = edges(end-1);
fig = figure('Color', 'w');
tiledlayout_compat(1, 2);
nexttile_compat(); bar(centers, D.isi_rate_hist_counts, 1); set(gca, 'YScale', 'log'); autoscale_nonzero_hist_x(centers, D.isi_rate_hist_counts); xlabel('Instantaneous rate 1/ISI [Hz]'); ylabel('# intervals'); title('Pooled Inverse-ISI Rate Histogram');
nexttile_compat(); histogram(D.mean_isi_rate_hz_by_neuron(isfinite(D.mean_isi_rate_hz_by_neuron)), 80); xlabel('Per-neuron mean 1/ISI rate [Hz]'); ylabel('# neurons'); title('Mean Inverse-ISI Rate by Neuron');
save_figure(fig, pathstr, save_files);
end

function autoscale_nonzero_hist_x(centers, counts)
%AUTOSCALE_NONZERO_HIST_X Ignore empty high-rate tail bins when setting x-limits.
finite_nonzero = isfinite(centers) & isfinite(counts) & counts > 0;
if ~any(finite_nonzero)
    return;
end
hi = max(centers(finite_nonzero));
if ~(isfinite(hi) && hi > 0)
    return;
end
xlim([0, hi * 1.05]);
end

function plot_rate_isi_scatter(pathstr, D, save_files)
ok = isfinite(D.mean_isi_rate_hz_by_neuron) & D.mean_rate_hz_by_neuron > 0;
fig = figure('Color', 'w');
scatter(D.mean_rate_hz_by_neuron(ok), D.mean_isi_rate_hz_by_neuron(ok), 8, 'filled');
xlabel('Mean firing rate [Hz]'); ylabel('Mean inverse-ISI rate [Hz]'); title('Neuron Rate vs Inverse-ISI Rate'); grid on;
save_figure(fig, pathstr, save_files);
end

function plot_examples(prefix, D, max_raster_neurons, max_voltage_neurons, save_files)
for ee = 1:numel(D.examples)
    ex = D.examples(ee);
    S = ex.S;
    spike_counts = sum(S, 2);
    [~, ord] = sort(spike_counts, 'descend');
    keep = ord(1:min(max_raster_neurons, numel(ord)));
    [rr, cc] = find(S(keep, :));
    fig = figure('Color', 'w');
    if isempty(rr)
        plot(0, 0); xlim([0 max(1, D.steps)]); ylim([0 1]); text(0.05, 0.5, 'No spikes in example sample', 'Units', 'normalized');
    else
        scatter(cc * D.dt, rr, 4, 'k', 'filled');
        ylim([0 numel(keep)+1]);
    end
    xlabel('Time [s]'); ylabel('Neuron rank (most active subset)'); title(sprintf('Example Raster %d', ex.sample_index));
    save_figure(fig, sprintf('%s_example_%02d_raster.png', prefix, ee), save_files);

    if ~isempty(ex.u_buffer)
        U = double(ex.u_buffer);
        filled = find(any(isfinite(U), 1), 1, 'last');
        if ~isempty(filled) && filled > 0
            fig = figure('Color', 'w');
            nplot = min(max_voltage_neurons, size(U, 1));
            t = (0:filled-1) * D.dt;
            plot(t, U(1:nplot, 1:filled).', 'LineWidth', 1.2);
            xlabel('Time [s]'); ylabel('u [mV]'); title(sprintf('Example Voltage Traces %d', ex.sample_index));
            save_figure(fig, sprintf('%s_example_%02d_voltage.png', prefix, ee), save_files);
        end
    end

    if ~isempty(ex.I_in) || ~isempty(ex.I_rec) || ~isempty(ex.w)
        fig = figure('Color', 'w');
        tiledlayout_compat(1, 3);
        nexttile_compat(); plot_matrix_summary(ex.I_in, D.dt, 'Input drive [mV]');
        nexttile_compat(); plot_matrix_summary(ex.I_rec, D.dt, 'Recurrent drive [mV]');
        nexttile_compat(); plot_matrix_summary(ex.w, D.dt, 'Adaptation w [mV]');
        save_figure(fig, sprintf('%s_example_%02d_currents.png', prefix, ee), save_files);
    end
end
end

function plot_matrix_summary(M, dt, ttl)
if isempty(M)
    text(0.05, 0.5, 'not logged', 'Units', 'normalized'); axis off; title(ttl); return;
end
M = double(M);
t = (0:size(M,2)-1) * dt;
plot(t, mean(M, 1), 'LineWidth', 1.2); hold on;
plot(t, percentile_compat(M, 10, 1), '--');
plot(t, percentile_compat(M, 90, 1), '--'); hold off;
xlabel('Time [s]'); ylabel(ttl); title(ttl); grid on;
end

function tag = safe_tag(tag)
if ischar(tag)
    tag = tag;
elseif isa(tag, 'string')
    tag = char(tag);
elseif isnumeric(tag) && isscalar(tag)
    tag = sprintf('%g', tag);
else
    tag = char(tag);
end
tag = regexprep(tag, '[^A-Za-z0-9_\\-]+', '_');
if isempty(tag), tag = 'spiking_test'; end
end

function p = percentile_compat(x, q, dim)
if nargin < 3, dim = []; end
x = double(x);
if exist('prctile', 'file') == 2
    if isempty(dim), p = prctile(x, q); else, p = prctile(x, q, dim); end
    return;
end
if isempty(dim)
    x = sort(x(:));
    if isempty(x), p = NaN; return; end
    idx = max(1, min(numel(x), round((q/100) * numel(x))));
    p = x(idx);
else
    p = median(x, dim);
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

function save_figure(fig, pathstr, save_files)
if nargin < 3, save_files = true; end
if ~save_files
    drawnow;
    return;
end
try
    if exist('exportgraphics', 'file') == 2
        exportgraphics(fig, pathstr, 'Resolution', 180);
    else
        saveas(fig, pathstr);
    end
catch ME
    warning('Could not save figure "%s": %s', pathstr, ME.message);
end
try, close(fig); catch, end
end
