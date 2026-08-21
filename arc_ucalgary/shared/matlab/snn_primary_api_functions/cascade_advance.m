% cascade_advance.m
function [x_syn, r] = cascade_advance(P, f, x_syn, r)
%CASCADE_ADVANCE Advance the two-stage synaptic filter by a timestep fraction.
%   x_syn is the fast rise-filter state. r is the slower decay/readout state.
%   The fraction f is either rho or 1-rho when a timestep is split around a
%   spike event.
gsr_f = exp(single(f) .* log(max(P.gamma_sr, realmin('single'))));
gsd_f = exp(single(f) .* log(max(P.gamma_sd, realmin('single'))));
% Rise state decays first; decay/readout state integrates the rise state.
x_syn = gsr_f .* x_syn;
r = gsd_f .* r + (single(1)-gsd_f) .* x_syn;
end
