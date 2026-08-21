% make_dynamics_system_for_api.m
function sys = make_dynamics_system_for_api(system_name)
%MAKE_DYNAMICS_SYSTEM_FOR_API Resolve named dynamical systems explicitly.
%   Release scripts use make_dynamical_system.m. A missing system helper is
%   an installation/path error and must not silently become a toy oscillator.
name = lower(strtrim(char(system_name)));
name = strrep(name, '-', '_');
if any(strcmp(name, {'toy_sincos','sincos','toy'}))
    sys = struct();
    sys.name = 'toy_sincos';
    sys.dim = 2;
    sys.params = struct('omega', single(4*pi));
    sys.x0_default = single([0; 1]);
    sys.f = @(t, x, p) single([p.omega .* x(2); -p.omega .* x(1)]);
    return;
end
if strcmp(name, 'sprott_s')
    name = 'sprotts';
end
ensure_dynamics_utility_on_path('make_dynamical_system');
try
    sys = make_dynamical_system(name);
catch ME
    error('snn_primary_api:unknownDynamicsSystem', ...
        'Could not create dynamical system "%s": %s', name, ME.message);
end
if ~isstruct(sys) || ~isfield(sys, 'f') || ~isa(sys.f, 'function_handle') || ...
        ~isfield(sys, 'x0_default') || ~isfield(sys, 'params')
    error('snn_primary_api:badDynamicsSystem', ...
        'make_dynamical_system("%s") did not return the required system fields.', name);
end
end
