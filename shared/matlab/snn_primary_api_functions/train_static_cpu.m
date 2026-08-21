% train_static_cpu.m
function result = train_static_cpu(domain, data, P, opts)
P_initial = P;
hist = init_static_history(opts.epochs);
best = struct('loss', inf, 'metric', -inf, 'B', P.B, 'epoch', 0);
for ep = 1:opts.epochs
    order = int32(randperm(size(data.X_train,2)));
    [loss_sum, metric, gB] = static_epoch_cpu(domain, data.X_train, data.Y_train, P, opts, order, true);
    P = adam_bias_update(P, gB, single(get_lr(ep, opts.epochs, opts.SCHED)), size(data.X_train,2));
    hist.train_loss(ep) = single(loss_sum / max(1,size(data.X_train,2)));
    hist.train_metric(ep) = single(metric);
    if should_validate(ep, opts)
        [val, ~] = static_eval_cpu(domain, data.X_val, data.Y_val, P, opts);
        hist.val_loss(ep) = single(val.loss);
        hist.val_metric(ep) = single(val.metric);
        if is_better(domain, val.loss, val.metric, best.loss, best.metric)
            best = struct('loss', val.loss, 'metric', val.metric, 'B', P.B, 'epoch', ep);
        end
    end
    update_live_training_plot(domain, hist, ep, opts);
    update_arc_training_progress(domain, hist, ep, opts, best);
end
Pbest = P;
Pbest.B = best.B;
assert_bias_only_update(P_initial, P, 'train_static_cpu');
[test, ~] = static_eval_cpu(domain, data.X_test, data.Y_test, Pbest, opts);
if domain == "regression"
    test = attach_regression_test_stats(test, data, 'test');
end
result = package_static_result(domain, 'cpu', P, best, hist, test, opts);
end

