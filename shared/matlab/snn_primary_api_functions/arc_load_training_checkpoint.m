% arc_load_training_checkpoint.m
function [checkpoint, loaded] = arc_load_training_checkpoint(opts)
%ARC_LOAD_TRAINING_CHECKPOINT Load an ARC timeout checkpoint, if present.
%   Checkpoints are deliberately opt-in. Local scripts do not set
%   opts.arc_checkpoint.enable, so local training starts from scratch exactly
%   as before.
checkpoint = struct();
loaded = false;
cfg = get_opt(opts, 'arc_checkpoint', struct());
if ~logical(get_opt(cfg, 'enable', false))
    return;
end
checkpoint_file = char(get_opt(cfg, 'file', ''));
if isempty(checkpoint_file) || exist(checkpoint_file, 'file') ~= 2
    return;
end
S = load(checkpoint_file, 'checkpoint');
if ~isfield(S, 'checkpoint')
    error('snn_primary_api:checkpointInvalid', 'Checkpoint file exists but does not contain a checkpoint struct.');
end
checkpoint = S.checkpoint;
loaded = true;
end
