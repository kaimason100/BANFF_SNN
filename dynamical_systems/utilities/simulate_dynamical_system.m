% Package orientation: Dynamical-system utility. It defines or simulates true systems that become training/evaluation trajectories for the SNN dynamics tasks.

function X = simulate_dynamical_system(f, Tspan, x0, dt, params, method, rate)
%SIMULATE_DYNAMICAL_SYSTEM Integrate dx/dt = f(t, x, params) with fixed dt.
%   X = simulate_dynamical_system(f, [t0 t1], x0, dt, params, method)
% Inputs:
%   f      : function handle @(t, x, params) -> dxdt  (vector of size D)
%   Tspan  : [t0 t1] single/double
%   x0     : [D x 1] single/double initial condition
%   dt     : scalar step (single/double). Output grid includes t0.
%   params : arbitrary struct or [] passed into f
%   method : 'euler' (default). Other method names error explicitly.
%   rate   : optional scalar multiplier on dx/dt. rate=1 gives the raw
%            physical dynamics; larger values advance the same vector field
%            faster on the training/sample clock.
% Output:
%   X      : [D x N] single, N = round((t1 - t0)/dt)+1
%
% Notes:
%   - Returns SINGLE precision (for speed and consistency with training)
%   - f may output single or double; we cast to single
%   - No stiffness handling; for stiff systems, reduce dt.

if nargin < 6 || isempty(method), method = 'euler'; end
if nargin < 7 || isempty(rate), rate = 1; end
rate = single(rate);

t0 = Tspan(1); t1 = Tspan(2);
N  = round((t1 - t0)/dt) + 1;
D  = numel(x0);

X  = zeros(D, N, 'single');
x  = single(x0);
X(:,1) = x;

t = t0;
switch lower(method)
    case 'euler'
        for k = 2:N
            dx = rate .* single(f(t, x, params));
            x  = x + dt * dx;
            X(:,k) = x;
            t = t + dt;
        end
    otherwise
        error('simulate_dynamical_system:unsupportedIntegrator', ...
            'Unknown method: %s. This release uses fixed-step Euler only.', method);
end
end
