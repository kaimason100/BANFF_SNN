% validate_sparse_gpu_edge_batch.m
% Helper for snn_primary_api.

function validate_sparse_gpu_edge_batch(P, batch_size, context)
%VALIDATE_SPARSE_GPU_EDGE_BATCH Guard CUDA sparse recurrence launch indexing.
if string(get_opt(P, 'recurrent_mode', "low_rank")) ~= "full_rank"
    return;
end
storage = string(get_opt(P, 'recurrent_storage', get_opt(P, 'full_rank_storage', "dense")));
if storage ~= "sparse"
    return;
end
edge_count = double(get_opt(P, 'rec_nnz', 0));
batch_count = double(batch_size);
edge_batch = edge_count * batch_count;
if edge_batch > double(intmax('int32'))
    error('snn_primary_api:sparseGpuEdgeBatchOverflow', ...
        ['%s would launch %.3g sparse recurrent edge-batch operations (%g edges x batch %g), ', ...
         'which exceeds int32 CUDA kernel indexing. Reduce opts.arch.full_rank_p_rec, ', ...
         'use low_rank recurrence, reduce N_hidden, or lower opts.batch_size.'], ...
        context, edge_batch, edge_count, batch_count);
end
end
