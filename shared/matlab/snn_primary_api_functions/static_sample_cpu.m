% static_sample_cpu.m
function [z_avg, ebar_sum] = static_sample_cpu(P, x, opts)
%STATIC_SAMPLE_CPU Run one static input sample through the SNN.
%   The input is held constant for opts.steps_present timesteps. The task
%   output is the average primary readout over the final averaging window.
%   The returned ebar_sum is the summed e-prop eligibility over the same
%   window, so it aligns with the averaged readout used in the loss.

% Initial hidden state at rest: membrane at leak potential, no adaptation,
% no synaptic current, and no readout-filter activity.
u = single(zeros(P.N_hidden,1) + P.E_L);
w = zeros(P.N_hidden,1,'single');
x_syn = zeros(P.N_hidden,1,'single');
r = zeros(P.N_hidden,1,'single');
elig = zero_elig(P.N_hidden);
z_sum = zeros(P.N_out,1,'single');
ebar_sum = zeros(P.N_hidden,1,'single');
% Fixed input current for this static sample. P.INPUT_SCALE supplies the
% single global 1/sqrt(N_in) factor for all tasks.
I_in = P.W_in * (P.INPUT_SCALE * single(x));
for k = 1:opts.steps_present
    % Advance hidden dynamics and local e-prop eligibility.
    [u, w, rho, spike, surr, x_syn, r] = primary_step(P, I_in, u, w, x_syn, r);
    elig = elig_step(P, elig, rho, spike, surr);
    if k >= opts.k_avg_start
        % Primary readout from filtered spikes. Static tasks use a temporal
        % average of this output over the final part of the stimulus window.
        z_sum = z_sum + P.W_out * r;
        ebar_sum = ebar_sum + elig.Ebar_f;
    end
end
z_avg = z_sum ./ single(opts.steps_avg);
end
