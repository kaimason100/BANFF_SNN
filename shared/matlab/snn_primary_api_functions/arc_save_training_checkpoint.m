% arc_save_training_checkpoint.m
function arc_save_training_checkpoint(opts, checkpoint)
cfg = get_opt(opts, 'arc_checkpoint', struct());
checkpoint_file = char(get_opt(cfg, 'file', ''));
if isempty(checkpoint_file)
    error('snn_primary_api:checkpointPath', 'opts.arc_checkpoint.file must be set when ARC checkpointing is enabled.');
end
checkpoint_dir = fileparts(checkpoint_file);
if ~isempty(checkpoint_dir) && exist(checkpoint_dir, 'dir') ~= 7
    mkdir(checkpoint_dir);
end
save(checkpoint_file, 'checkpoint', '-v7.3');
fprintf('[ARC checkpoint] saved epoch %d to %s%s', checkpoint.epoch, checkpoint_file, newline);
end

