% train_dynamics_cpu.m
function result = train_dynamics_cpu(x, lambda, P, opts)
P_initial = P;
hist = zeros(opts.epochs,1,'single');
closed_hist = init_closed_loop_history(opts.epochs);
best = struct('loss', inf, 'wd', inf, 'B', P.B, 'epoch', 0);
n_steps = supervised_step_count(x);
opts_closed = opts;
opts_closed.T_sim = single(get_opt(opts, 'closed_loop_validation_time', opts.T_sim));
closed_eval_set = make_closed_loop_eval_set(opts_closed);
for ep = 1:opts.epochs
    lr = single(get_lr(ep, opts.epochs, opts.SCHED));
    if isstruct(x) && isfield(x, 'pool')
        loss_sum = single(0);
        block_steps = max(1, x.steps - 1);
        for bb = 1:x.train_blocks
            start_idx = randi(x.max_start_idx, 1, 'uint32');
            xb = x.pool(:, double(start_idx):double(start_idx)+x.steps-1);
            [block_loss, gB] = dynamics_epoch_cpu(xb, lambda, P, opts, true);
            P = adam_bias_update(P, gB, lr, block_steps);
            loss_sum = loss_sum + block_loss;
        end
    elseif iscell(x)
        loss_sum = single(0);
        for bb = 1:numel(x)
            [block_loss, gB] = dynamics_epoch_cpu(x{bb}, lambda{bb}, P, opts, true);
            P = adam_bias_update(P, gB, lr, max(1, size(x{bb},2)-1));
            loss_sum = loss_sum + block_loss;
        end
    else
        [loss_sum, gB] = dynamics_epoch_cpu(x, lambda, P, opts, true);
        P = adam_bias_update(P, gB, lr, n_steps);
    end
    hist(ep) = single(loss_sum / max(1,n_steps));
    if should_closed_loop_validate(ep, opts)
        closed = dynamics_closed_loop_evaluation_cpu(P, opts, closed_eval_set);
        closed_hist.wd(ep) = single(closed.wasserstein_distance);
        plot_closed_loop_validation_trajectories(closed, opts, ep);
        if closed.wasserstein_distance < best.wd
            best = struct('loss', hist(ep), 'wd', closed.wasserstein_distance, 'B', P.B, 'epoch', ep);
        end
    elseif hist(ep) < best.loss && ~isfinite(best.wd)
        best = struct('loss', hist(ep), 'wd', inf, 'B', P.B, 'epoch', ep);
    end
    live_hist = pack_dynamics_live_history(hist, closed_hist);
    update_live_training_plot("dynamics", live_hist, ep, opts);
    update_arc_training_progress("dynamics", live_hist, ep, opts, best);
end
if best.epoch == 0
    best = struct('loss', hist(end), 'wd', inf, 'B', P.B, 'epoch', opts.epochs);
end
Pbest = P;
Pbest.B = best.B;
assert_bias_only_update(P_initial, P, 'train_dynamics_cpu');
meta = primary_bias_training_metadata();
meta.mex = mex_runtime_metadata('cpu', 'dynamical_systems');
result = struct('backend', 'cpu', 'history', hist, 'best', best, ...
    'closed_loop_validation', closed_hist, 'final_B', P.B, 'model', Pbest, ...
    'options', opts, 'training_metadata', meta);
result.model.recurrent_mode = Pbest.recurrent_mode;
result.model.decoder_mode = Pbest.decoder_mode;
result.dynamics = dynamics_training_metadata(x, opts);
result.training = struct('trainable_parameters', 'hidden_bias_only');
result = attach_architecture_metadata(result, Pbest, opts);
end
