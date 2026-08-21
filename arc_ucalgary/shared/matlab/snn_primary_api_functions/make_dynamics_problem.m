% make_dynamics_problem.m
function [x, P] = make_dynamics_problem(opts)
%MAKE_DYNAMICS_PROBLEM Build normalized trajectory data and matching SNN.
%   For training, a deterministic long trajectory is simulated once, burn-in
%   is discarded, normalization statistics are fit on the remaining pool, and
%   short contiguous snippets are sampled from that pool during training.

rng(get_opt(opts, 'split_seed', get_opt(opts, 'seed', 42)), 'twister');
% Dynamics trajectories include both endpoints, so T_sim=1 and dt=0.001
% gives num_samples=1001 and num_transitions=1000 one-step predictions.
num_samples = max(2, round(double(opts.T_sim) / double(opts.dt)) + 1);
sys = make_dynamics_system_for_api(opts.system_name);
split = lower(string(get_opt(opts, 'dynamics_split', 'train')));
if split == "closed_loop" && isfield(opts, 'dynamics_mu') && isfield(opts, 'dynamics_sigma')
    % Closed-loop evaluation/testing reuses saved training normalization so
    % predictions and true trajectories are compared in the same coordinates.
    mu = single(opts.dynamics_mu);
    sigma = single(opts.dynamics_sigma);
    [~, raw_test] = simulate_long_dynamics_trajectory(sys, opts, opts.T_sim);
    x = single((raw_test(:,1:min(num_samples,size(raw_test,2))) - mu) ./ sigma);
    P = make_primary_model(numel(mu), numel(mu), opts);
    return;
end
[~, raw] = simulate_long_dynamics_trajectory(sys, opts, get_opt(opts, 'long_sim_time', opts.T_sim));
% Endpoint-inclusive indexing retains raw(:,burn_in_k), so a nominal 10 s
% burn-in at dt=1 ms discards samples at 0--9.998 s and retains t=9.999 s.
% This exact historical convention is documented in the manuscript.
burn_in_k = max(1, round(double(get_opt(opts, 'burn_in_time', 0)) / double(opts.dt)));
burn_in_k = min(burn_in_k, size(raw,2));
pool_raw = raw(:,burn_in_k:end);
% Normalize each state dimension using training-pool statistics.
mu = mean(pool_raw,2);
sigma = std(pool_raw,0,2); sigma(sigma==0) = 1;
pool_norm = single((pool_raw - mu) ./ sigma);
if split == "closed_loop"
    [~, raw_test] = simulate_long_dynamics_trajectory(sys, opts, opts.T_sim);
    x = single((raw_test(:,1:min(num_samples,size(raw_test,2))) - mu) ./ sigma);
else
    if size(pool_norm,2) < num_samples
        error('snn_primary_api:dynamicsPoolTooShort', ...
            ['Long trajectory pool has %d samples after burn-in, but T_sim requires %d samples. ', ...
            'Increase opts.long_sim_time or reduce opts.T_sim/burn_in_time.'], size(pool_norm,2), num_samples);
    end
    % Store the long normalized pool and snippet metadata. The training loop
    % samples contiguous windows using max_start_idx.
    x = struct('pool', pool_norm, 'steps', num_samples, ...
        'num_samples', num_samples, 'num_transitions', num_samples - 1, ...
        'sample_convention', 'endpoint_inclusive', ...
        'max_start_idx', size(pool_norm,2)-num_samples+1, ...
        'train_blocks', max(1, round(get_opt(opts, 'train_blocks', 1))), ...
        'mu', mu, 'sigma', sigma);
end
P = make_primary_model(size(pool_norm,1), size(pool_norm,1), opts);
end
