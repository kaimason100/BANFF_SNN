% Package orientation: Shared implementation helper for publication analysis exports.

function neural = publication_static_rate_summary(P, data, opts, batch_size)
%PUBLICATION_STATIC_RATE_SUMMARY Compute full-test static firing summaries.
%   Rates are in Hz. rate_matrix is N_hidden x N_test_samples and stores the
%   mean firing rate of each neuron during each held-out sample presentation.

if nargin < 4 || isempty(batch_size)
    batch_size = 32;
end
batch_size = max(1, round(double(batch_size)));
n_samples = size(data.X_test, 2);
rate_matrix = zeros(P.N_hidden, n_samples, 'single');

for start_idx = 1:batch_size:n_samples
    idx = start_idx:min(n_samples, start_idx + batch_size - 1);
    [S, ~, ~, ~, ~] = static_spike_diagnostics_cpu(P, data.X_test(:, idx), opts);
    sample_rates = squeeze(sum(S, 2)) ./ max(double(opts.steps_present) * double(opts.dt), eps);
    if isvector(sample_rates)
        sample_rates = sample_rates(:);
    end
    rate_matrix(:, idx) = single(sample_rates);
end

rate_by_neuron = mean(rate_matrix, 2, 'omitnan');
active_mask = rate_by_neuron > 0;

neural = struct();
neural.mean_firing_rate_by_neuron_hz = single(rate_by_neuron(:));
neural.firing_rate_matrix_hz = rate_matrix;
neural.active_neuron_mask = active_mask(:);
neural.active_fraction = double(mean(active_mask(:)));
neural.active_fraction_percent = 100 * neural.active_fraction;
neural.calculation = struct();
neural.calculation.context = 'static_full_held_out_test';
neural.calculation.rate_units = 'Hz';
neural.calculation.rate_matrix_shape = 'N_hidden x N_test_samples';
neural.calculation.sample_rate_formula = 'spike_count_per_neuron_per_sample / (steps_present * dt)';
neural.calculation.mean_rate_formula = 'mean(sample_rate_hz across held-out test samples)';
neural.calculation.active_neuron_rule = 'mean_firing_rate_by_neuron_hz > 0';
neural.calculation.batch_size = batch_size;
neural.calculation.n_test_samples = n_samples;
neural.calculation.steps_present = double(opts.steps_present);
neural.calculation.dt = double(opts.dt);
end
