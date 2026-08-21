% validate_dynamics_data.m
function validate_dynamics_data(x, lambda, context)
if isempty(lambda)
    error('snn_primary_api:badDynamicsLambda', '%s lambda must be nonempty.', context);
end
if iscell(x)
    if ~iscell(lambda) || numel(lambda) ~= numel(x)
        error('snn_primary_api:badDynamicsLambda', '%s cell lambda must match cell data.', context);
    end
    for ii = 1:numel(x)
        validate_dynamics_data(x{ii}, lambda{ii}, sprintf('%s block %d', context, ii));
    end
elseif isstruct(x) && isfield(x, 'pool')
    required = {'pool','steps','max_start_idx','train_blocks'};
    for ii = 1:numel(required)
        if ~isfield(x, required{ii})
            error('snn_primary_api:badDynamicsPool', '%s dynamics pool is missing %s.', context, required{ii});
        end
    end
    if isempty(x.pool) || ndims(x.pool) ~= 2 || any(~isfinite(x.pool(:)))
        error('snn_primary_api:badDynamicsPool', '%s dynamics pool must be a finite 2-D matrix.', context);
    end
    if x.steps < 2 || x.steps > size(x.pool,2)
        error('snn_primary_api:badDynamicsSteps', ...
            '%s block length %d is incompatible with pool length %d.', context, x.steps, size(x.pool,2));
    end
    if x.max_start_idx < 1 || x.max_start_idx ~= size(x.pool,2)-x.steps+1
        error('snn_primary_api:badDynamicsStarts', '%s has inconsistent max_start_idx.', context);
    end
    if x.train_blocks < 1
        error('snn_primary_api:badDynamicsBlocks', '%s train_blocks must be positive.', context);
    end
    if ~isvector(lambda) || numel(lambda) ~= x.steps
        error('snn_primary_api:dynamicsLambdaLength', ...
            '%s lambda length %d does not match block length %d.', context, numel(lambda), x.steps);
    end
else
    if isempty(x) || ndims(x) ~= 2 || size(x,2) < 2 || any(~isfinite(x(:)))
        error('snn_primary_api:badDynamicsMatrix', '%s dynamics data must be a finite D x T matrix with T >= 2.', context);
    end
    if ~isvector(lambda) || numel(lambda) ~= size(x,2)
        error('snn_primary_api:dynamicsLambdaLength', ...
            '%s lambda length %d does not match trajectory length %d.', context, numel(lambda), size(x,2));
    end
end
end

