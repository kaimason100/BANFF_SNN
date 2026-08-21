% elig_step.m
function elig = elig_step(P, elig, rho, spike, surr)
%ELIG_STEP E-prop eligibility update for the trainable hidden bias B.
%   The bias affects membrane current directly. The code tracks how a small
%   change in each neuron's bias would affect its voltage/adaptation, then
%   converts that local effect through the surrogate spike derivative and the
%   synaptic filters to obtain Ebar_f, the eligibility of the readout state r.

% Fractional pre/post-spike decays matching advance_u_w and cascade_advance.
log_alpha = log(max(P.alpha, realmin('single')));
log_beta = log(max(P.beta, realmin('single')));
log_gsr = log(max(P.gamma_sr, realmin('single')));
log_gsd = log(max(P.gamma_sd, realmin('single')));
a_pre = exp(rho .* log_alpha);
b_pre = exp(rho .* log_beta);
a_post = exp((single(1)-rho) .* log_alpha);
b_post = exp((single(1)-rho) .* log_beta);

% Bias-to-voltage eligibility before reset. eps_v_noa is the voltage
% eligibility excluding adaptation feedback; eps_a tracks adaptation
% eligibility. ev_full_pre combines them into membrane eligibility.
elig.eps_v_noa = a_pre .* elig.eps_v_noa + (single(1)-a_pre);
ev_full_pre = elig.eps_v_noa - (single(1)-a_pre) .* elig.eps_a;
% Convert voltage eligibility to spike eligibility using the surrogate
% derivative at the threshold crossing.
e_raw = surr .* ev_full_pre;
% The general API includes subthreshold adaptation sensitivity through a_eff.
% Active publication configurations set it to zero, leaving exponential decay
% between the spike-triggered eligibility jumps below.
elig.eps_a = b_pre .* elig.eps_a + (single(1)-b_pre) .* (P.a_eff .* ev_full_pre);
if any(spike)
    elig.eps_v_noa(spike) = 0;
    elig.eps_a(spike) = elig.eps_a(spike) + P.b_param .* e_raw(spike);
end
% Finish voltage/adaptation eligibility through the post-spike fraction.
elig.eps_v_noa = a_post .* elig.eps_v_noa + (single(1)-a_post);
ev_full = elig.eps_v_noa - (single(1)-a_post) .* elig.eps_a;
elig.eps_a = b_post .* elig.eps_a + (single(1)-b_post) .* (P.a_eff .* ev_full);

% Propagate spike eligibility through the same rise/decay synaptic cascade
% used by the forward state. Ebar_f is the eligibility of r and is multiplied
% by W_out' * dL/dz in the learning rule.
gsr_pre = exp(rho .* log_gsr);
gsd_pre = exp(rho .* log_gsd);
gsr_post = exp((single(1)-rho) .* log_gsr);
gsd_post = exp((single(1)-rho) .* log_gsd);
Ex_pre = gsr_pre .* elig.Ebar_x;
elig.Ebar_f = gsd_pre .* elig.Ebar_f + (single(1)-gsd_pre) .* Ex_pre;
if any(spike), Ex_pre(spike) = Ex_pre(spike) + P.spike_jump_sr .* e_raw(spike); end
elig.Ebar_x = gsr_post .* Ex_pre;
elig.Ebar_f = gsd_post .* elig.Ebar_f + (single(1)-gsd_post) .* elig.Ebar_x;
end
