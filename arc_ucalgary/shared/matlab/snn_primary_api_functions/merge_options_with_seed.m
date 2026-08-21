% merge_options_with_seed.m
function opts = merge_options_with_seed(base, override)
%MERGE_OPTIONS_WITH_SEED Merge options while preserving intended seeds.
%   If a caller sets opts.seed but not opts.init_seed, the network
%   initialization seed follows opts.seed. Saved-model test calls preserve
%   the saved init_seed unless the caller explicitly overrides seed/init_seed.
if nargin < 2 || isempty(override)
    override = struct();
end
user_set_seed = isstruct(override) && isfield(override, 'seed');
user_set_init_seed = isstruct(override) && isfield(override, 'init_seed');
opts = merge_struct(base, override);
if ~isfield(opts, 'seed') || isempty(opts.seed)
    warning('snn_primary_api:defaultFallback', ...
        'Using default option in merge_options_with_seed: opts.seed = 42 because the field was not supplied.');
    opts.seed = 42;
end
if ~user_set_init_seed && (user_set_seed || ~isfield(opts, 'init_seed') || isempty(opts.init_seed))
    if user_set_seed
        warning('snn_primary_api:defaultFallback', ...
            'Using opts.seed as default in merge_options_with_seed: opts.init_seed = opts.seed because opts.init_seed was not supplied.');
    else
        warning('snn_primary_api:defaultFallback', ...
            'Using default option in merge_options_with_seed: opts.init_seed = opts.seed because opts.init_seed was not supplied.');
    end
    opts.init_seed = opts.seed;
end
if ~isfield(opts, 'split_seed') || isempty(opts.split_seed)
    warning('snn_primary_api:defaultFallback', ...
        'Using default option in merge_options_with_seed: opts.split_seed = 42 because the field was not supplied.');
    opts.split_seed = 42;
end
opts = normalize_arch_options(opts);
opts = normalize_network_options(opts);
end
