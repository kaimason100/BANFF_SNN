% simulate_long_dynamics_trajectory.m
function [t, x_raw] = simulate_long_dynamics_trajectory(sys, opts, sim_time)
steps = max(2, round(double(sim_time) / double(opts.dt)) + 1);
x0 = single(get_opt(opts, 'x0_override', sys.x0_default));
ensure_dynamics_utility_on_path('simulate_dynamical_system');
integrator = char(get_opt(opts, 'integrator', 'euler'));
raw = simulate_dynamical_system(sys.f, [0 single(sim_time)], x0, opts.dt, sys.params, integrator, opts.dyn_sys_rate);
x_raw = single(raw(:,1:min(steps,size(raw,2))));
t = single((0:size(x_raw,2)-1) .* double(opts.dt));
if any(~isfinite(x_raw(:)))
    error('snn_primary_api:nonfiniteLongSimulation', '%s long trajectory contains non-finite values.', sys.name);
end
end
