% update_arc_training_progress.m
function update_arc_training_progress(kind, hist, ep, opts, best)
%UPDATE_ARC_TRAINING_PROGRESS Print non-graphical progress for cluster jobs.
%   This is disabled unless opts.arc_progress.enable is true, so local Live
%   Editor scripts keep their plotting behaviour unchanged.
cfg = get_opt(opts, 'arc_progress', struct());
if ~logical(get_opt(cfg, 'enable', false))
    return;
end
every = max(1, round(double(get_opt(cfg, 'every', max(1, get_opt(opts, 'validate_every', 100))))));
static_val_due = isfield(opts, 'validate_every') && should_validate(ep, opts);
closed_val_due = isfield(opts, 'closed_loop_validate_every') && should_closed_loop_validate(ep, opts);
if ~(ep == 1 || ep == opts.epochs || mod(ep, every) == 0 || static_val_due || closed_val_due)
    return;
end
persistent progress_state
if isempty(progress_state) || ep == 1
    progress_state = struct('tic_id', tic, 'last_epoch', 0);
    fprintf('[ARC %s] starting training for %d epoch(s)%s', char(kind), opts.epochs, newline);
end
if ep == progress_state.last_epoch
    return;
end
progress_state.last_epoch = ep;
elapsed = toc(progress_state.tic_id);
pct = 100 * double(ep) / max(1, double(opts.epochs));
eta = elapsed * max(0, double(opts.epochs - ep)) / max(1, double(ep));
line = sprintf('[ARC %s] epoch %d/%d | %5.1f%%%% | train loss %s', ...
    char(kind), ep, opts.epochs, pct, fmt_progress_number(current_train_loss(hist, ep)));
if isstruct(hist) && isfield(hist, 'closed_wd')
    line = sprintf('%s | closed WD %s', line, ...
        fmt_progress_number(history_value(hist.closed_wd, ep)));
elseif isstruct(hist)
    metric_name = live_metric_label(kind);
    line = sprintf('%s | train %s %s | val loss %s | val %s %s', line, ...
        metric_name, fmt_progress_number(history_value(hist.train_metric, ep)), ...
        fmt_progress_number(history_value(hist.val_loss, ep)), ...
        metric_name, fmt_progress_number(history_value(hist.val_metric, ep)));
end
if isstruct(hist) && isfield(hist, 'closed_wd')
    line = sprintf('%s | best epoch %d | selected train loss %s | elapsed %s | ETA %s', line, ...
        get_opt(best, 'epoch', 0), fmt_progress_number(get_opt(best, 'loss', NaN)), ...
        format_progress_seconds(elapsed), format_progress_seconds(eta));
else
    line = sprintf('%s | best epoch %d | best loss %s | elapsed %s | ETA %s', line, ...
        get_opt(best, 'epoch', 0), fmt_progress_number(get_opt(best, 'loss', NaN)), ...
        format_progress_seconds(elapsed), format_progress_seconds(eta));
end
if isfield(best, 'metric')
    line = sprintf('%s | best metric %s', line, fmt_progress_number(best.metric));
elseif isfield(best, 'wd')
    line = sprintf('%s | best WD %s', line, fmt_progress_number(best.wd));
end
fprintf('%s%s', line, newline);
end
