% train_static_gpu.m
function result = train_static_gpu(domain, data, P, opts)
P_initial = P;
Pg = init_static_gpu(domain, data, P, opts);
cleanup = onCleanup(@() clear_static_gpu(domain)); %#ok<NASGU>
hist = init_static_history(opts.epochs);
best = struct('loss', inf, 'metric', -inf, 'B', Pg.B, 'epoch', 0);
[checkpoint, resumed] = arc_load_training_checkpoint(opts);
start_epoch = 1;
if resumed
    arc_validate_checkpoint_kind(checkpoint, 'static', char(domain), 'gpu');
    assert_checkpoint_split_policy(checkpoint, opts, data, domain);
    Pg = checkpoint.model;
    hist = checkpoint.history;
    best = checkpoint.best;
    static_update_bias_gpu(domain, Pg.B, Pg.W_out);
    static_set_optimizer_state_gpu(domain, checkpoint.optimizer_state);
    rng(checkpoint.rng_state);
    start_epoch = checkpoint.epoch + 1;
    fprintf('[ARC checkpoint] resumed %s GPU training from epoch %d%s', char(domain), checkpoint.epoch, newline);
end
arc_timer = tic;
for ep = start_epoch:opts.epochs
    order = int32(randperm(size(data.X_train,2)));
    [loss_sum, metric, Pg] = static_train_epoch_gpu(domain, data, 'train', order, Pg, opts, ep);
    hist.train_loss(ep) = single(loss_sum / max(1,size(data.X_train,2)));
    hist.train_metric(ep) = single(metric);
    if should_validate(ep, opts)
        val = static_validate_gpu(domain, data, 'val', opts);
        hist.val_loss(ep) = single(val.loss);
        hist.val_metric(ep) = single(val.metric);
        if is_better(domain, val.loss, val.metric, best.loss, best.metric)
            best = struct('loss', val.loss, 'metric', val.metric, 'B', Pg.B, 'epoch', ep);
        end
    end
    update_live_training_plot(domain, hist, ep, opts);
    update_arc_training_progress(domain, hist, ep, opts, best);
    if arc_checkpoint_due(opts, arc_timer, ep)
        Pg.B = static_get_bias_gpu(domain);
        checkpoint = arc_make_training_checkpoint('static', char(domain), 'gpu', ep, Pg, best, hist, opts, static_get_optimizer_state_gpu(domain));
        arc_save_training_checkpoint(opts, checkpoint);
        result = package_static_result(domain, 'gpu', Pg, best, hist, struct(), opts);
        result.checkpoint = arc_checkpoint_public_info(opts, checkpoint, true);
        return;
    end
end
static_update_bias_gpu(domain, best.B, Pg.W_out);
test = static_predict_gpu(domain, data, 'test', opts);
if domain == "regression" && ~has_valid_regression_stats(test)
    Pcpu = Pg;
    Pcpu.B = best.B;
    [cpu_test, ~] = static_eval_cpu(domain, data.X_test, data.Y_test, Pcpu, opts);
    test.Z = cpu_test.Z;
    test.Y = cpu_test.Y;
    test.regression = regression_test_stats_task_units(test.Z, test.Y, data);
    test.regression.source = 'cpu_reference_outputs';
end
assert_bias_only_update(P_initial, Pg, 'train_static_gpu');
result = package_static_result(domain, 'gpu', Pg, best, hist, test, opts);
result.checkpoint = arc_checkpoint_public_info(opts, struct(), false);
end

function assert_checkpoint_split_policy(checkpoint, opts, data, domain)
% Refuse legacy regression checkpoints whose row split can contain exact
% feature-plus-target duplicates in more than one partition.
if domain ~= "regression" || ...
        ~logical(get_opt(opts, 'group_exact_duplicate_rows', false)) || ...
        double(get_opt(data, 'exact_duplicate_group_count', 0)) < 1
    return;
end
checkpoint_summary = get_opt(get_opt(checkpoint, 'options', struct()), ...
    'data_summary', struct());
if ~isfield(checkpoint, 'options') || ...
        ~logical(get_opt(checkpoint.options, 'group_exact_duplicate_rows', false)) || ...
        ~strcmp(char(get_opt(checkpoint_summary, 'split_policy', '')), ...
        'grouped_exact_feature_target_rows')
    error('snn_primary_api:legacyRegressionCheckpointSplit', ...
        ['This regression checkpoint predates duplicate-grouped splitting. ' ...
         'Remove or archive the checkpoint and restart training from epoch 1.']);
end
for key = {'idx_train', 'idx_val', 'idx_test'}
    name = key{1};
    if ~isfield(checkpoint.options, name) || ~isfield(opts, name)
        error('snn_primary_api:checkpointRegressionSplitMismatch', ...
            'Checkpoint/current options are missing %s indices.', name);
    end
    saved_indices = checkpoint.options.(name);
    current_indices = opts.(name);
    if ~isequal(uint32(saved_indices(:)), uint32(current_indices(:)))
        error('snn_primary_api:checkpointRegressionSplitMismatch', ...
            'Checkpoint %s indices do not match the current duplicate-grouped split.', name);
    end
end
checkpoint_hash = char(get_opt(checkpoint_summary, 'dataset_sha256', ''));
current_hash = char(get_opt(get_opt(opts, 'data_summary', struct()), 'dataset_sha256', ''));
if ~isempty(checkpoint_hash) && ~isempty(current_hash) && ...
        ~strcmp(checkpoint_hash, current_hash)
    error('snn_primary_api:checkpointRegressionDatasetMismatch', ...
        'Checkpoint dataset hash does not match the current regression dataset.');
end
end

function value = get_opt(S, name, default_value)
if isstruct(S) && isfield(S, name) && ~isempty(S.(name))
    value = S.(name);
else
    value = default_value;
end
end
