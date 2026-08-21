% attach_architecture_metadata.m
% Helper for snn_primary_api.

function result = attach_architecture_metadata(result, P, opts)
%ATTACH_ARCHITECTURE_METADATA Store requested and resolved architecture state.
opts = normalize_arch_options(opts);
P = ensure_model_architecture_fields(P, opts);

arch = struct();
arch.requested = opts.arch;
arch.resolved = struct();
arch.resolved.recurrent_mode = string(P.recurrent_mode);
arch.resolved.decoder_mode = string(P.decoder_mode);
arch.resolved.signed_decoder_distribution = string(P.signed_decoder_distribution);
arch.resolved.recurrent_storage = string(P.recurrent_storage);
arch.resolved.full_rank_p_rec = single(opts.arch.full_rank_p_rec);
arch.resolved.full_rank_sparse_threshold = single(opts.arch.full_rank_sparse_threshold);
arch.resolved.max_dense_full_rank_N = int32(opts.arch.max_dense_full_rank_N);
arch.resolved.max_sparse_full_rank_nnz = int64(opts.arch.max_sparse_full_rank_nnz);
arch.resolved.max_full_rank_recurrent_bytes = double(opts.arch.max_full_rank_recurrent_bytes);
arch.model = struct();
arch.model.N_hidden = int32(P.N_hidden);
arch.model.N_in = int32(P.N_in);
arch.model.N_out = int32(P.N_out);
arch.model.N_rec = int32(P.N_rec);

arch.full_rank = struct();
if isfield(P, 'W_rec') && ~isempty(P.W_rec)
    nnz_rec = nnz(P.W_rec);
    arch.full_rank.nnz_recurrent = int64(nnz_rec);
    if numel(P.W_rec) > 0
        arch.full_rank.density_recurrent = single(double(nnz_rec) ./ double(numel(P.W_rec)));
    else
        arch.full_rank.density_recurrent = single(0);
    end
else
    arch.full_rank.nnz_recurrent = [];
    arch.full_rank.density_recurrent = [];
end
arch.full_rank.remove_self_connections = logical(opts.arch.full_rank_remove_self_connections);

arch.decoder = struct();
arch.decoder.is_shared = string(P.decoder_mode) == "shared";
arch.decoder.is_signed_independent = string(P.decoder_mode) == "signed";
if string(P.decoder_mode) == "signed"
    arch.decoder.distribution = string(P.signed_decoder_distribution);
else
    arch.decoder.distribution = "not_applicable";
end

base_seed = get_opt(opts, 'seed', 0);
seed_defaults = struct('init_seed', uint64(get_opt(opts, 'init_seed', base_seed)), ...
    'decoder_seed', [], 'full_rank_mask_seed', [], 'full_rank_weight_seed', []);
if isfield(P, 'arch_seeds') && isstruct(P.arch_seeds)
    arch.seeds = merge_struct(seed_defaults, P.arch_seeds);
else
    arch.seeds = seed_defaults;
end

result.arch = arch;
end
