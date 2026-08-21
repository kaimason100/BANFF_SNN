% arc_checkpoint_public_info.m
function info = arc_checkpoint_public_info(opts, checkpoint, needs_resubmit)
cfg = get_opt(opts, 'arc_checkpoint', struct());
info = struct();
info.enabled = logical(get_opt(cfg, 'enable', false));
info.needs_resubmit = logical(needs_resubmit);
info.file = char(get_opt(cfg, 'file', ''));
info.submit_script = char(get_opt(cfg, 'submit_script', ''));
info.array_id = char(get_opt(cfg, 'array_id', ''));
if isstruct(checkpoint) && isfield(checkpoint, 'epoch')
    info.epoch = checkpoint.epoch;
else
    info.epoch = [];
end
info.complete = info.enabled && ~info.needs_resubmit;
end
