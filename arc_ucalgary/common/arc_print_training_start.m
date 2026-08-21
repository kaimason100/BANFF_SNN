function arc_print_training_start(task_name, opts, model_file)
%ARC_PRINT_TRAINING_START Print the key hyperparameters before training.

fprintf('[ARC %s] model output: %s%s', task_name, model_file, newline);
fprintf('[ARC %s] epochs: %g | N_hidden: %g | N_rec: %g | seed: %g%s', ...
    task_name, get_field(opts, 'epochs', NaN), get_field(opts, 'N_hidden', NaN), ...
    get_field(opts, 'N_rec', NaN), get_field(opts, 'seed', NaN), newline);
if isfield(opts, 'arch')
    fprintf('[ARC %s] architecture: recurrent=%s | decoder=%s | signed_distribution=%s | full_rank_p_rec=%.6g | storage=%s | resolved_storage=%s | sparse_threshold=%.6g | max_dense_N=%g | max_sparse_nnz=%g | max_recurrent_GB=%.3g%s', ...
        task_name, char(get_field(opts.arch, 'recurrent_mode', "unknown")), ...
        char(get_field(opts.arch, 'decoder_mode', "unknown")), ...
        char(get_field(opts.arch, 'signed_decoder_distribution', "unknown")), ...
        double(get_field(opts.arch, 'full_rank_p_rec', NaN)), ...
        char(get_field(opts.arch, 'full_rank_storage', "unknown")), ...
        char(get_field(opts.arch, 'resolved_storage', "not_resolved_yet")), ...
        double(get_field(opts.arch, 'full_rank_sparse_threshold', NaN)), ...
        double(get_field(opts.arch, 'max_dense_full_rank_N', NaN)), ...
        double(get_field(opts.arch, 'max_sparse_full_rank_nnz', NaN)), ...
        double(get_field(opts.arch, 'max_full_rank_recurrent_bytes', NaN)) / 2^30, newline);
end
if isfield(opts, 'seed_list')
    fprintf('[ARC %s] seed_list: %s | split_seed: %g%s', task_name, ...
        mat2str(opts.seed_list), get_field(opts, 'split_seed', NaN), newline);
end
if isfield(opts, 'task_tag')
    fprintf('[ARC %s] task_tag: %s%s', task_name, char(opts.task_tag), newline);
end
if isfield(opts, 'system_name')
    fprintf('[ARC %s] system_name: %s%s', task_name, char(opts.system_name), newline);
end
if isfield(opts, 'batch_size')
    fprintf('[ARC %s] batch_size: %g%s', task_name, opts.batch_size, newline);
end
if isfield(opts, 'validate_every')
    fprintf('[ARC %s] validate_every: %g%s', task_name, opts.validate_every, newline);
end
if isfield(opts, 'closed_loop_validate_every')
    fprintf('[ARC %s] closed_loop_validate_every: %g%s', task_name, opts.closed_loop_validate_every, newline);
end
if isfield(opts, 'arc_closed_loop_validation_backend')
    fprintf('[ARC %s] training closed-loop validation backend: %s%s', ...
        task_name, char(opts.arc_closed_loop_validation_backend), newline);
end
if isfield(opts, 'closed_loop_validation_ics')
    fprintf('[ARC %s] training closed-loop validation: time %.6g | ics %g | ic_jitter %.6g | ic_seed %g%s', ...
        task_name, get_field(opts, 'closed_loop_validation_time', NaN), opts.closed_loop_validation_ics, ...
        get_field(opts, 'closed_loop_ic_jitter', NaN), get_field(opts, 'closed_loop_ic_seed', NaN), newline);
end
if isfield(opts, 'closed_loop_test_time') || isfield(opts, 'closed_loop_test_ics')
    fprintf('[ARC %s] local final-test settings saved in opts: time %.6g | ics %g%s', ...
        task_name, get_field(opts, 'closed_loop_test_time', NaN), get_field(opts, 'closed_loop_test_ics', NaN), newline);
end
if isfield(opts, 'arc_checkpoint') && isfield(opts.arc_checkpoint, 'file')
    fprintf('[ARC %s] checkpoint file: %s | cutoff: %.2f h%s', task_name, ...
        opts.arc_checkpoint.file, double(get_field(opts.arc_checkpoint, 'max_seconds', NaN)) / 3600, newline);
end
if isfield(opts, 'SCHED')
    fprintf('[ARC %s] lr schedule: %s | start %.6g | end %.6g%s', task_name, ...
        char(opts.SCHED.type), double(opts.SCHED.lr_start), double(opts.SCHED.lr_end), newline);
end
end

function value = get_field(s, name, fallback)
if isstruct(s) && isfield(s, name)
    value = s.(name);
else
    value = fallback;
end
end
