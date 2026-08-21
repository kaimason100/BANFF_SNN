% train_dynamics_gpu.m
function result = train_dynamics_gpu(x, lambda, P, opts)
P_initial = P;
validate_gpu_dynamics_cell_lengths(x);
Pg = init_dynamics_gpu(P, opts, max_sequence_steps(x));
cleanup = onCleanup(@() clear_dynamics_gpu()); %#ok<NASGU>
hist = zeros(opts.epochs,1,'single');
closed_hist = init_closed_loop_history(opts.epochs);
best = struct('loss', inf, 'wd', inf, 'B', Pg.B, 'epoch', 0);
n_steps = supervised_step_count(x);
opts_closed = opts;
opts_closed.T_sim = single(get_opt(opts, 'closed_loop_validation_time', opts.T_sim));
closed_eval_set = make_closed_loop_eval_set(opts_closed);
[checkpoint, resumed] = arc_load_training_checkpoint(opts);
start_epoch = 1;
if resumed
    arc_validate_checkpoint_kind(checkpoint, 'dynamics', 'dynamical_systems', 'gpu');
    Pg = checkpoint.model;
    hist = checkpoint.history;
    closed_hist = checkpoint.closed_loop_validation;
    best = checkpoint.best;
    dynamics_update_bias_gpu(Pg.B);
    dynamics_set_optimizer_state_gpu(checkpoint.optimizer_state);
    rng(checkpoint.rng_state);
    start_epoch = checkpoint.epoch + 1;
    fprintf('[ARC checkpoint] resumed dynamical systems GPU training from epoch %d%s', checkpoint.epoch, newline);
end
arc_timer = tic;
for ep = start_epoch:opts.epochs
    [loss_sum, Pg] = dynamics_train_epoch_gpu(x, lambda, Pg, opts, ep);
    hist(ep) = single(loss_sum / max(1,n_steps));
    if should_closed_loop_validate(ep, opts)
        [closed, Pg] = dynamics_closed_loop_validation_gpu_training(x, Pg, opts, closed_eval_set);
        closed_hist.wd(ep) = single(closed.wasserstein_distance);
        plot_closed_loop_validation_trajectories(closed, opts, ep);
        if closed.wasserstein_distance < best.wd
            best = struct('loss', hist(ep), 'wd', closed.wasserstein_distance, 'B', Pg.B, 'epoch', ep);
        end
    elseif hist(ep) < best.loss && ~isfinite(best.wd)
        best = struct('loss', hist(ep), 'wd', inf, 'B', Pg.B, 'epoch', ep);
    end
    live_hist = pack_dynamics_live_history(hist, closed_hist);
    update_live_training_plot("dynamics", live_hist, ep, opts);
    update_arc_training_progress("dynamics", live_hist, ep, opts, best);
    if arc_checkpoint_due(opts, arc_timer, ep)
        Pg.B = dynamics_get_bias_gpu();
        checkpoint = arc_make_training_checkpoint('dynamics', 'dynamical_systems', 'gpu', ep, Pg, best, hist, opts, dynamics_get_optimizer_state_gpu());
        checkpoint.closed_loop_validation = closed_hist;
        arc_save_training_checkpoint(opts, checkpoint);
        meta = primary_bias_training_metadata();
        meta.mex = mex_runtime_metadata('gpu', 'dynamical_systems');
        result = struct('backend', 'gpu', 'history', hist, 'best', best, ...
            'closed_loop_validation', closed_hist, 'final_B', Pg.B, 'model', Pg, ...
            'options', opts, 'training_metadata', meta, ...
            'checkpoint', arc_checkpoint_public_info(opts, checkpoint, true));
        result.model.recurrent_mode = Pg.recurrent_mode;
        result.model.decoder_mode = Pg.decoder_mode;
        result.dynamics = dynamics_training_metadata(x, opts);
        result.training = struct('trainable_parameters', 'hidden_bias_only');
        result = attach_architecture_metadata(result, Pg, opts);
        return;
    end
end
if best.epoch == 0
    best = struct('loss', hist(end), 'wd', inf, 'B', Pg.B, 'epoch', opts.epochs);
end
Pbest = Pg;
Pbest.B = best.B;
assert_bias_only_update(P_initial, Pg, 'train_dynamics_gpu');
meta = primary_bias_training_metadata();
meta.mex = mex_runtime_metadata('gpu', 'dynamical_systems');
result = struct('backend', 'gpu', 'history', hist, 'best', best, ...
    'closed_loop_validation', closed_hist, 'final_B', Pg.B, 'model', Pbest, ...
    'options', opts, 'training_metadata', meta);
result.model.recurrent_mode = Pbest.recurrent_mode;
result.model.decoder_mode = Pbest.decoder_mode;
result.checkpoint = arc_checkpoint_public_info(opts, struct(), false);
result.dynamics = dynamics_training_metadata(x, opts);
result.training = struct('trainable_parameters', 'hidden_bias_only');
result = attach_architecture_metadata(result, Pbest, opts);
end
