% apply_saved_dynamics_metadata.m
function opts = apply_saved_dynamics_metadata(opts, meta)
%APPLY_SAVED_DYNAMICS_METADATA Prefer training-time normalization at test time.
if isfield(meta, 'mu') && isfield(meta, 'sigma') && ~isempty(meta.mu) && ~isempty(meta.sigma)
    opts.dynamics_mu = single(meta.mu);
    opts.dynamics_sigma = single(meta.sigma);
end
if isfield(meta, 'system_name') && ~isempty(meta.system_name)
    opts.system_name = char(meta.system_name);
end
end

