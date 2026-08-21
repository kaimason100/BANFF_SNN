% primary_step.m
function [u, w, rho, spike, surr, x_syn, r] = primary_step(P, I_in, u, w, x_syn, r)
%PRIMARY_STEP Advance the recurrent spiking network by one timestep.
%   The recurrent current is dispatched by P.recurrent_mode, then added to
%   external input plus the trainable bias current B.

I_rec = recurrent_current(P, r);
% Total current driving the adaptive LIF membrane. P.B is the only learned
% parameter in this release.
I_tot = I_in + I_rec + P.B;
% Exact membrane/adaptation update with fractional spike timing and surrogate
% derivative evaluated at the threshold crossing.
[u, w, rho, spike, surr] = advance_u_w(P, u, w, I_tot);
% Advance synaptic filters to the spike time, inject spike jump, then advance
% through the remaining fraction of the timestep.
[x_syn, r] = cascade_advance(P, rho, x_syn, r);
x_syn(spike) = x_syn(spike) + P.spike_jump_sr;
[x_syn, r] = cascade_advance(P, single(1)-rho, x_syn, r);
end
