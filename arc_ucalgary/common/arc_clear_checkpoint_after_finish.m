function arc_clear_checkpoint_after_finish(result)
%ARC_CLEAR_CHECKPOINT_AFTER_FINISH Remove stale checkpoint state after success.
if ~isstruct(result) || ~isfield(result, 'checkpoint') || ~isfield(result.checkpoint, 'file')
    return;
end
checkpoint_file = char(result.checkpoint.file);
if isempty(checkpoint_file) || exist(checkpoint_file, 'file') ~= 2
    return;
end
delete(checkpoint_file);
fprintf('[ARC checkpoint] removed completed checkpoint: %s%s', checkpoint_file, newline);
end
