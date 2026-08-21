% default_arch_options.m
% Helper for snn_primary_api.

function arch = default_arch_options()
%DEFAULT_ARCH_OPTIONS Architecture defaults matching the original model.
arch = struct();
arch.recurrent_mode = "low_rank";
arch.decoder_mode = "signed";
arch.signed_decoder_distribution = "uniform";
arch.full_rank_p_rec = single(1.0);
arch.full_rank_remove_self_connections = true;
arch.full_rank_storage = "auto";
arch.full_rank_sparse_threshold = single(0.10);
arch.max_dense_full_rank_N = int32(6000);
arch.max_sparse_full_rank_nnz = int64(20000000);
arch.max_full_rank_recurrent_bytes = double(2.5 * 2^30);
end
