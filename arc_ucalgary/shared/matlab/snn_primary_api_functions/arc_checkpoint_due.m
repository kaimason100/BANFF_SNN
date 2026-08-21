% arc_checkpoint_due.m
function tf = arc_checkpoint_due(opts, timer_id, ep)
%ARC_CHECKPOINT_DUE True when an ARC job should stop and resume later.
cfg = get_opt(opts, 'arc_checkpoint', struct());
if ~logical(get_opt(cfg, 'enable', false))
    tf = false;
    return;
end
max_seconds = double(get_opt(cfg, 'max_seconds', 23*3600));
tf = ep < opts.epochs && toc(timer_id) >= max_seconds;
end
