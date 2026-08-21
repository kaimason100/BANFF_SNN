% Package orientation: Shared implementation helper for publication analysis exports.

function neural = publication_dynamics_rate_summary(P, opts)
%PUBLICATION_DYNAMICS_RATE_SUMMARY Compute full closed-loop test firing rates.
%   Rates are computed after the configured closed-loop warmup.  When the
%   centralized publication export enables it, sparse full-rollout events
%   are stored separately in seed_entry.spike_events for offline timing tests.

eval_set = make_closed_loop_eval_set(opts);
n_ic = numel(eval_set.x_true);
rate_by_ic = zeros(P.N_hidden, n_ic, 'single');
spike_steps_by_ic = zeros(1, n_ic);
spike_count_total = zeros(P.N_hidden, 1);
total_steps = 0;

for ic = 1:n_ic
    [spike_count, spike_steps] = dynamics_spike_count_after_warmup( ...
        P, eval_set.x_true{ic}, eval_set.lambda{ic}, eval_set.opts, eval_set.warmup_steps);
    spike_steps_by_ic(ic) = spike_steps;
    spike_count_total = spike_count_total + spike_count;
    total_steps = total_steps + spike_steps;
    rate_by_ic(:, ic) = single(spike_count ./ max(spike_steps * double(eval_set.opts.dt), eps));
end

rate_by_neuron = spike_count_total ./ max(total_steps * double(eval_set.opts.dt), eps);
active_mask = rate_by_neuron > 0;

neural = struct();
neural.mean_firing_rate_by_neuron_hz = single(rate_by_neuron(:));
neural.mean_firing_rate_by_neuron_by_ic_hz = rate_by_ic;
neural.active_neuron_mask = active_mask(:);
neural.active_fraction = double(mean(active_mask(:)));
neural.active_fraction_percent = 100 * neural.active_fraction;
neural.spike_count_by_neuron_post_warmup = single(spike_count_total(:));
neural.spike_steps_by_ic = spike_steps_by_ic;
neural.calculation = struct();
neural.calculation.context = 'dynamics_full_closed_loop_test_after_warmup';
neural.calculation.rate_units = 'Hz';
neural.calculation.rate_formula = 'post_warmup_spike_count_per_neuron / (post_warmup_steps * dt)';
neural.calculation.active_neuron_rule = 'mean_firing_rate_by_neuron_hz > 0';
neural.calculation.closed_loop_warmup_time = double(eval_set.warmup_time);
neural.calculation.closed_loop_test_time = double(eval_set.test_time);
neural.calculation.closed_loop_test_ics = n_ic;
neural.calculation.warmup_steps = double(eval_set.warmup_steps);
neural.calculation.total_post_warmup_steps = double(total_steps);
neural.calculation.dt = double(eval_set.opts.dt);
end

function [spike_count, spike_steps] = dynamics_spike_count_after_warmup(P, x, lambda, opts, warmup_steps)
steps = size(x, 2);
u = single(zeros(P.N_hidden, 1) + P.E_L);
w = zeros(P.N_hidden, 1, 'single');
x_syn = zeros(P.N_hidden, 1, 'single');
r = zeros(P.N_hidden, 1, 'single');
Zprev = zeros(P.N_out, 1, 'single');
spike_count = zeros(P.N_hidden, 1);
spike_steps = 0;

for k = 1:steps-1
    if k == 1 || lambda(k)
        x_in = x(:, k);
    else
        x_in = Zprev;
    end
    I_in = P.W_in * (P.INPUT_SCALE * single(x_in));
    I_rec = recurrent_current(P, r);
    [u, w, rho, spike, ~] = advance_u_w(P, u, w, I_in + I_rec + P.B);
    [x_syn, r] = cascade_advance(P, rho, x_syn, r);
    x_syn(spike) = x_syn(spike) + P.spike_jump_sr;
    [x_syn, r] = cascade_advance(P, single(1) - rho, x_syn, r);
    Zprev = P.W_out * r;
    if k > warmup_steps
        spike_count = spike_count + double(spike(:));
        spike_steps = spike_steps + 1;
    end
end

if spike_steps == 0
    spike_steps = max(1, steps - 1);
end
end
