function opts = arc_apply_network_seed(opts)
%ARC_APPLY_NETWORK_SEED Apply the SLURM array seed to one training job.
%   ARC jobs run one network initialization seed per array item. This helper
%   removes seed_list so snn_primary_api trains exactly one network per job.

seed_value = str2double(getenv('ARC_NETWORK_SEED'));
if ~(isscalar(seed_value) && isfinite(seed_value) && seed_value == round(seed_value) && seed_value >= 1)
    seed_value = get_field(opts, 'seed', 1);
end
opts.seed = seed_value;
opts.init_seed = seed_value;
if isfield(opts, 'seed_list')
    opts = rmfield(opts, 'seed_list');
end
end

function value = get_field(s, name, fallback)
if isstruct(s) && isfield(s, name)
    value = s.(name);
else
    value = fallback;
end
end
