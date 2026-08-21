% Package orientation: Shared MATLAB utility used across tasks. Read this with the caller open so input/output structs and saved-result fields are clear.

function snn_plot_check_neural_comparison(results)
%SNN_PLOT_CHECK_NEURAL_COMPARISON Plot CPU-vs-GPU neural replay differences.
%   Each panel shows absolute and relative differences between the final
%   CPU-trained and GPU-trained models replayed on the same inputs.

if iscell(results)
    result_cells = results(:);
else
    results = results(:);
    result_cells = cell(numel(results),1);
    for ii = 1:numel(results)
        result_cells{ii} = results(ii);
    end
end
if isempty(result_cells)
    warning('snn_plot_check_neural_comparison:empty', 'No check results to plot.');
    return;
end

for ii = 1:numel(result_cells)
    r = result_cells{ii};
    if ~isfield(r, 'neural') || isempty(r.neural)
        continue;
    end
    plot_one_neural_check(r, ii);
end
end

function plot_one_neural_check(result, index)
N = result.neural;
seed_value = local_seed(result, index);
figure('Color', 'w');
tiledlayout_compat(3, 2);

nexttile_compat();
plot_spike_difference(N.cpu_spikes, N.gpu_spikes, get_cmp(N, 'spike_raster_difference'));

nexttile_compat();
plot_rate_difference(N.cpu_rate_hz_by_neuron, N.gpu_rate_hz_by_neuron, get_cmp(N, 'rate_difference'));

nexttile_compat();
plot_trace_difference(N.cpu_voltage, N.gpu_voltage, 'Voltage', get_cmp(N, 'voltage_difference'));

nexttile_compat();
plot_trace_difference(N.cpu_adaptation, N.gpu_adaptation, 'Adaptation', get_cmp(N, 'adaptation_difference'));

nexttile_compat();
plot_trace_difference(N.cpu_input_current, N.gpu_input_current, 'Input current', ...
    get_cmp(N, 'input_current_difference'));

nexttile_compat();
plot_trace_difference(N.cpu_recurrent_current, N.gpu_recurrent_current, 'Recurrent current', ...
    get_cmp(N, 'recurrent_current_difference'));

sgtitle_compat(sprintf('CPU-GPU spiking and network differences, seed %g', seed_value));
if isfield(N, 'replay_note') && ~isempty(N.replay_note)
    fprintf('[CHECK neural] seed %g: %s\n', seed_value, N.replay_note);
end
end

function plot_spike_difference(S_cpu, S_gpu, cmp)
S_cpu = flatten_spikes(S_cpu);
S_gpu = flatten_spikes(S_gpu);
n = min([size(S_cpu,1), size(S_gpu,1), 80]);
steps = min(size(S_cpu,2), size(S_gpu,2));
if n < 1 || steps < 1
    axis off;
    title('Spike raster difference unavailable');
    return;
end
D = xor(S_cpu(1:n,1:steps), S_gpu(1:n,1:steps));
imagesc(D);
colormap(gca, [1 1 1; 0.1 0.1 0.1]);
grid on;
xlabel('Step');
ylabel('Neuron');
title(sprintf('Spike raster abs diff | max %.3g, rel %.3g', cmp.max_abs, cmp.max_rel));
end

function plot_rate_difference(rates_cpu, rates_gpu, cmp)
rates_cpu = double(rates_cpu(:));
rates_gpu = double(rates_gpu(:));
n = min(numel(rates_cpu), numel(rates_gpu));
if n < 1
    axis off;
    title('Rate difference unavailable');
    return;
end
Dabs = abs(rates_cpu(1:n) - rates_gpu(1:n));
Drel = relative_difference(Dabs, rates_cpu(1:n));
semilogy(1:n, clamp_for_log(Dabs), '-', 'LineWidth', 1.0); hold on;
semilogy(1:n, clamp_for_log(Drel), '--', 'LineWidth', 1.0); hold off;
grid on;
xlabel('Neuron');
ylabel('Difference');
legend({'Absolute','Relative'}, 'Location', 'best');
title(sprintf('Rate diff | max %.3g Hz, rel %.3g', cmp.max_abs, cmp.max_rel));
end

function plot_trace_difference(A_cpu, A_gpu, ttl, cmp)
A_cpu = double(A_cpu);
A_gpu = double(A_gpu);
if isempty(A_cpu) || isempty(A_gpu)
    axis off;
    title([ttl ' difference unavailable']);
    return;
end
if ndims(A_cpu) > 2
    A_cpu = A_cpu(:,:,1);
end
if ndims(A_gpu) > 2
    A_gpu = A_gpu(:,:,1);
end
n = min(size(A_cpu,1), size(A_gpu,1));
steps = min(size(A_cpu,2), size(A_gpu,2));
if n < 1 || steps < 1
    axis off;
    title([ttl ' difference unavailable']);
    return;
end
A_cpu = A_cpu(1:n,1:steps);
A_gpu = A_gpu(1:n,1:steps);
Dabs = abs(A_cpu - A_gpu);
Drel = relative_difference(Dabs, A_cpu);
semilogy(1:steps, clamp_for_log(max(Dabs, [], 1)), '-', 'LineWidth', 1.0); hold on;
semilogy(1:steps, clamp_for_log(max(Drel, [], 1)), '--', 'LineWidth', 1.0); hold off;
grid on;
xlabel('Step');
ylabel('Max across neurons');
legend({'Absolute','Relative'}, 'Location', 'best');
title(sprintf('%s diff | max %.3g, rel %.3g', ttl, cmp.max_abs, cmp.max_rel));
end

function Drel = relative_difference(Dabs, reference)
Drel = Dabs ./ max(1, abs(double(reference)));
end

function Y = clamp_for_log(Y)
Y = double(Y);
Y(~isfinite(Y)) = nan;
Y(isfinite(Y)) = max(Y(isfinite(Y)), realmin('double'));
end

function cmp = get_cmp(s, field_name)
cmp = struct('max_abs', nan, 'max_rel', nan);
if isfield(s, field_name) && isstruct(s.(field_name))
    if isfield(s.(field_name), 'max_abs')
        cmp.max_abs = double(s.(field_name).max_abs);
    end
    if isfield(s.(field_name), 'max_rel')
        cmp.max_rel = double(s.(field_name).max_rel);
    end
end
end

function S = flatten_spikes(S)
S = logical(S);
if ndims(S) >= 3
    S = reshape(S, size(S,1), size(S,2) * size(S,3));
end
end

function seed_value = local_seed(result, index)
if isfield(result, 'options') && isfield(result.options, 'seed')
    seed_value = double(result.options.seed);
elseif isfield(result, 'seed')
    seed_value = double(result.seed);
else
    seed_value = index;
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
