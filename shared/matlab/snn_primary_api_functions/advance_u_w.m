% advance_u_w.m
function [u, w, rho, spike, surr] = advance_u_w(P, u0, w0, I_tot)
%ADVANCE_U_W Exact timestep update for adaptive LIF voltage/adaptation.
%   The timestep is split at the estimated threshold crossing time rho. This
%   preserves event timing within the discrete step and keeps the surrogate
%   derivative aligned with the spike crossing rather than the endpoint.

% Candidate voltage at the end of the full timestep without a reset event.
u_hat = P.E_L + P.alpha .* (u0 - P.E_L) + P.oneMinusAlpha .* (I_tot - w0);
spike = u_hat >= P.V_th;
% Fraction of the timestep at which voltage crosses threshold under a linear
% interpolation between u0 and u_hat. Non-spiking neurons keep rho=0.
u_diff = u_hat - u0;
rho = zeros(size(u0), 'single');
idx = spike & (u_diff > 0);
rho(idx) = (P.V_th - u0(idx)) ./ max(u_diff(idx), realmin('single'));
rho = min(max(rho, single(0)), single(1));
rho(~isfinite(rho)) = single(0);

% Convert full-step decay constants into pre-spike and post-spike fractional
% decays. This is equivalent to exp(-rho*dt/tau) and exp(-(1-rho)*dt/tau).
log_alpha = log(max(P.alpha, realmin('single')));
log_beta = log(max(P.beta, realmin('single')));
alpha_pre = exp(rho .* log_alpha);
beta_pre = exp(rho .* log_beta);
alpha_post = exp((single(1)-rho) .* log_alpha);
beta_post = exp((single(1)-rho) .* log_beta);

% Advance membrane/adaptation to the event time. The general implementation
% supports subthreshold voltage-to-adaptation coupling through a_eff, but all
% active publication configurations set a_eff to zero; adaptation is therefore
% purely spike triggered and otherwise decays with beta.
u_star = P.E_L + alpha_pre .* (u0 - P.E_L) + (single(1)-alpha_pre) .* (I_tot - w0);
w1 = beta_pre .* w0 + (single(1)-beta_pre) .* (P.a_eff .* (u_star - P.E_L));
% Surrogate derivative for e-prop, evaluated near the threshold crossing.
u_lin = u0 + rho .* (u_hat - u0);
surr = P.phi_u .* max(single(0), single(1) - abs((u_lin - P.V_th) ./ max(P.delta_u, realmin('single'))));
if any(spike)
    % Reset membrane and add spike-triggered adaptation jump.
    u_star(spike) = P.V_reset;
    w1(spike) = w1(spike) + P.b_param;
end
% Complete the remaining fraction of the timestep after reset/jump.
u = P.E_L + alpha_post .* (u_star - P.E_L) + (single(1)-alpha_post) .* (I_tot - w1);
w = beta_post .* w1 + (single(1)-beta_post) .* (P.a_eff .* (u - P.E_L));
end
