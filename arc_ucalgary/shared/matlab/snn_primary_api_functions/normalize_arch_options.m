% normalize_arch_options.m
% Helper for snn_primary_api.

function opts = normalize_arch_options(opts)
%NORMALIZE_ARCH_OPTIONS Fill and validate architecture option fields.
if ~isfield(opts, 'arch') || isempty(opts.arch)
    opts.arch = default_arch_options();
else
    opts.arch = merge_struct(default_arch_options(), opts.arch);
end

opts.arch.recurrent_mode = lower(string(opts.arch.recurrent_mode));
opts.arch.decoder_mode = lower(string(opts.arch.decoder_mode));
opts.arch.signed_decoder_distribution = lower(string(opts.arch.signed_decoder_distribution));
if opts.arch.signed_decoder_distribution == "normal"
    opts.arch.signed_decoder_distribution = "gaussian";
end
opts.arch.full_rank_storage = lower(string(opts.arch.full_rank_storage));
opts.arch.full_rank_p_rec = single(opts.arch.full_rank_p_rec);
opts.arch.full_rank_sparse_threshold = single(opts.arch.full_rank_sparse_threshold);
opts.arch.full_rank_remove_self_connections = logical(opts.arch.full_rank_remove_self_connections);
max_dense_full_rank_N = double(opts.arch.max_dense_full_rank_N);
max_sparse_full_rank_nnz = double(opts.arch.max_sparse_full_rank_nnz);
max_full_rank_recurrent_bytes = double(opts.arch.max_full_rank_recurrent_bytes);
opts.arch.max_dense_full_rank_N = int32(max_dense_full_rank_N);
opts.arch.max_sparse_full_rank_nnz = int64(max_sparse_full_rank_nnz);
opts.arch.max_full_rank_recurrent_bytes = max_full_rank_recurrent_bytes;

if ~any(opts.arch.recurrent_mode == ["low_rank", "full_rank"])
    error('snn_primary_api:archRecurrentMode', ...
        'opts.arch.recurrent_mode must be "low_rank" or "full_rank", got "%s".', ...
        char(opts.arch.recurrent_mode));
end
if ~any(opts.arch.decoder_mode == ["shared", "signed"])
    error('snn_primary_api:archDecoderMode', ...
        'opts.arch.decoder_mode must be "shared" or "signed", got "%s".', ...
        char(opts.arch.decoder_mode));
end
if ~any(opts.arch.signed_decoder_distribution == ["gaussian", "uniform"])
    error('snn_primary_api:archSignedDecoderDistribution', ...
        'opts.arch.signed_decoder_distribution must be "gaussian", "normal" or "uniform", got "%s".', ...
        char(opts.arch.signed_decoder_distribution));
end
if ~any(opts.arch.full_rank_storage == ["auto", "dense", "sparse"])
    error('snn_primary_api:archFullRankStorage', ...
        'opts.arch.full_rank_storage must be "auto", "dense" or "sparse", got "%s".', ...
        char(opts.arch.full_rank_storage));
end
if ~(isscalar(opts.arch.full_rank_p_rec) && isfinite(opts.arch.full_rank_p_rec) && opts.arch.full_rank_p_rec >= 0 && opts.arch.full_rank_p_rec <= 1)
    error('snn_primary_api:archFullRankPRec', ...
        'opts.arch.full_rank_p_rec must be a finite probability in [0, 1].');
end
if ~(isscalar(opts.arch.full_rank_sparse_threshold) && isfinite(opts.arch.full_rank_sparse_threshold) && opts.arch.full_rank_sparse_threshold >= 0 && opts.arch.full_rank_sparse_threshold <= 1)
    error('snn_primary_api:archFullRankSparseThreshold', ...
        'opts.arch.full_rank_sparse_threshold must be a finite probability in [0, 1].');
end
if ~(isscalar(max_dense_full_rank_N) && isfinite(max_dense_full_rank_N) && max_dense_full_rank_N == floor(max_dense_full_rank_N) && max_dense_full_rank_N >= 1)
    error('snn_primary_api:archMaxDenseFullRankN', ...
        'opts.arch.max_dense_full_rank_N must be a positive integer.');
end
if ~(isscalar(max_sparse_full_rank_nnz) && isfinite(max_sparse_full_rank_nnz) && max_sparse_full_rank_nnz == floor(max_sparse_full_rank_nnz) && max_sparse_full_rank_nnz >= 0)
    error('snn_primary_api:archMaxSparseFullRankNnz', ...
        'opts.arch.max_sparse_full_rank_nnz must be a non-negative integer.');
end
if ~(isscalar(max_full_rank_recurrent_bytes) && isfinite(max_full_rank_recurrent_bytes) && max_full_rank_recurrent_bytes > 0)
    error('snn_primary_api:archMaxFullRankRecurrentBytes', ...
        'opts.arch.max_full_rank_recurrent_bytes must be a positive finite byte limit.');
end

if opts.arch.full_rank_storage == "auto"
    if opts.arch.full_rank_p_rec <= opts.arch.full_rank_sparse_threshold
        opts.arch.resolved_storage = "sparse";
    else
        opts.arch.resolved_storage = "dense";
    end
else
    opts.arch.resolved_storage = opts.arch.full_rank_storage;
end
end
