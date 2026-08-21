% normalize_network_options.m
% Helper for snn_primary_api.

function opts = normalize_network_options(opts)
%NORMALIZE_NETWORK_OPTIONS Validate shared network sparsity options.
if ~isfield(opts, 'NET') || ~isstruct(opts.NET)
    error('snn_primary_api:netOptions', 'opts.NET must be a struct.');
end
if ~isfield(opts.NET, 'p_rec') || isempty(opts.NET.p_rec)
    opts.NET.p_rec = single(1);
end
opts.NET.p_rec = single(opts.NET.p_rec);
if ~(isscalar(opts.NET.p_rec) && isfinite(opts.NET.p_rec) && opts.NET.p_rec >= 0 && opts.NET.p_rec <= 1)
    error('snn_primary_api:netPRec', ...
        'opts.NET.p_rec must be a finite low-rank recurrent probability in [0, 1].');
end
if ~isfield(opts.NET, 'variance_correction') || isempty(opts.NET.variance_correction)
    opts.NET.variance_correction = true;
end
opts.NET.variance_correction = logical(opts.NET.variance_correction);
if ~isfield(opts.NET, 'dale') || ~isstruct(opts.NET.dale)
    opts.NET.dale = struct('enable', true, 'p_exc', 0.5, 'sign', []);
end
end
