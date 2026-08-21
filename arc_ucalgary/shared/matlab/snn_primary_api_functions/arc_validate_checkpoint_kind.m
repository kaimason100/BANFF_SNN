% arc_validate_checkpoint_kind.m
function arc_validate_checkpoint_kind(checkpoint, kind, domain, backend)
%ARC_VALIDATE_CHECKPOINT_KIND Prevent accidental resume from a different task.
if ~isfield(checkpoint, 'kind') || ~strcmp(char(checkpoint.kind), char(kind)) || ...
        ~isfield(checkpoint, 'domain') || ~strcmp(char(checkpoint.domain), char(domain)) || ...
        ~isfield(checkpoint, 'backend') || ~strcmp(char(checkpoint.backend), char(backend))
    error('snn_primary_api:checkpointMismatch', ...
        'Checkpoint kind/domain/backend does not match the requested training run.');
end
required = {'epoch','model','best','history','optimizer_state','rng_state'};
for ii = 1:numel(required)
    if ~isfield(checkpoint, required{ii})
        error('snn_primary_api:checkpointInvalid', 'Checkpoint is missing required field "%s".', required{ii});
    end
end
end

