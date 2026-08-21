% ensure_model_architecture_fields.m
% Helper for snn_primary_api.

function P = ensure_model_architecture_fields(P, opts)
%ENSURE_MODEL_ARCHITECTURE_FIELDS Backfill architecture fields on saved models.
opts = normalize_arch_options(opts);
if ~isfield(P, 'recurrent_mode') || isempty(P.recurrent_mode)
    P.recurrent_mode = opts.arch.recurrent_mode;
end
if ~isfield(P, 'decoder_mode') || isempty(P.decoder_mode)
    P.decoder_mode = opts.arch.decoder_mode;
end
if ~isfield(P, 'full_rank_storage') || isempty(P.full_rank_storage)
    P.full_rank_storage = opts.arch.resolved_storage;
end
if ~isfield(P, 'recurrent_storage') || isempty(P.recurrent_storage)
    P.recurrent_storage = P.full_rank_storage;
end
if ~isfield(P, 'signed_decoder_distribution') || isempty(P.signed_decoder_distribution)
    P.signed_decoder_distribution = opts.arch.signed_decoder_distribution;
end
if ~isfield(P, 'arch') || isempty(P.arch)
    P.arch = opts.arch;
end
if ~isfield(P, 'W_rec')
    P.W_rec = zeros(0, 0, 'single');
end
if ~isfield(P, 'W_rec_mask')
    P.W_rec_mask = false(0, 0);
end
if ~isfield(P, 'rec_post_idx')
    P.rec_post_idx = int32([]);
end
if ~isfield(P, 'rec_pre_idx')
    P.rec_pre_idx = int32([]);
end
if ~isfield(P, 'rec_w')
    P.rec_w = single([]);
end
if ~isfield(P, 'rec_nnz')
    P.rec_nnz = int32(numel(P.rec_w));
end
end
