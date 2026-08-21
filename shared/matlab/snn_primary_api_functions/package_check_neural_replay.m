function neural = package_check_neural_replay(S_cpu, S_gpu, U_cpu, U_gpu, Iin_cpu, Iin_gpu, Irec_cpu, Irec_gpu, W_cpu, W_gpu, dt, domain)
%PACKAGE_CHECK_NEURAL_REPLAY Summaries for CPU/GPU-trained neural replay.
%   This stores compact neural diagnostics directly in the check result so
%   plots can compare spike rasters, rates, voltages and currents.

S_cpu = logical(S_cpu);
S_gpu = logical(S_gpu);
neural = struct();
neural.domain = char(domain);
neural.dt = single(dt);
neural.cpu_spikes = S_cpu;
neural.gpu_spikes = S_gpu;
neural.cpu_voltage = single(U_cpu);
neural.gpu_voltage = single(U_gpu);
neural.cpu_input_current = single(Iin_cpu);
neural.gpu_input_current = single(Iin_gpu);
neural.cpu_recurrent_current = single(Irec_cpu);
neural.gpu_recurrent_current = single(Irec_gpu);
neural.cpu_adaptation = single(W_cpu);
neural.gpu_adaptation = single(W_gpu);
neural.spike_raster_difference = compare_arrays(single(S_cpu), single(S_gpu));
neural.voltage_difference = compare_arrays(single(U_cpu), single(U_gpu));
neural.input_current_difference = compare_arrays(single(Iin_cpu), single(Iin_gpu));
neural.recurrent_current_difference = compare_arrays(single(Irec_cpu), single(Irec_gpu));
neural.adaptation_difference = compare_arrays(single(W_cpu), single(W_gpu));

cpu_counts = spike_counts_by_neuron(S_cpu);
gpu_counts = spike_counts_by_neuron(S_gpu);
duration_s = double(size(S_cpu, 2)) * double(dt);
if ndims(S_cpu) >= 3
    n_repeats = size(S_cpu, 3);
else
    n_repeats = 1;
end
denom = max(realmin, duration_s * double(n_repeats));
neural.cpu_rate_hz_by_neuron = single(double(cpu_counts) ./ denom);
neural.gpu_rate_hz_by_neuron = single(double(gpu_counts) ./ denom);
neural.rate_difference = compare_arrays(neural.cpu_rate_hz_by_neuron, neural.gpu_rate_hz_by_neuron);
neural.cpu_active_percent = single(100 * mean(cpu_counts > 0));
neural.gpu_active_percent = single(100 * mean(gpu_counts > 0));
neural.active_percent_abs_diff = abs(double(neural.cpu_active_percent) - double(neural.gpu_active_percent));
neural.total_spikes_cpu = double(sum(cpu_counts));
neural.total_spikes_gpu = double(sum(gpu_counts));
neural.total_spike_abs_diff = abs(neural.total_spikes_cpu - neural.total_spikes_gpu);
end

function counts = spike_counts_by_neuron(S)
if ndims(S) >= 3
    counts = squeeze(sum(sum(S, 2), 3));
else
    counts = sum(S, 2);
end
counts = double(counts(:));
end
