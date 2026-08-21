% gpu_recurrent_arguments.m
% Helper for snn_primary_api.

function [W_rec_dense, storage_id, rec_post_idx, rec_pre_idx, rec_w, rec_nnz] = gpu_recurrent_arguments(P)
%GPU_RECURRENT_ARGUMENTS Pack full-rank recurrence for CUDA.
%   Sparse full-rank recurrence uses 0-based edge arrays:
%   rec_post_idx(e), rec_pre_idx(e), rec_w(e). Dense recurrence uses W_rec.
storage_id = recurrent_storage_id(P);
if string(get_opt(P, 'recurrent_mode', "low_rank")) ~= "full_rank"
    W_rec_dense = zeros(0, 0, 'single');
    rec_post_idx = int32([]);
    rec_pre_idx = int32([]);
    rec_w = single([]);
    rec_nnz = int32(0);
    return;
end

if storage_id == int32(1)
    W_rec_dense = zeros(0, 0, 'single');
    if ~isfield(P, 'rec_w') || isempty(P.rec_w)
        [post_idx, pre_idx, w_val] = find(P.W_rec);
        rec_post_idx = int32(post_idx(:) - 1);
        rec_pre_idx = int32(pre_idx(:) - 1);
        rec_w = single(w_val(:));
    else
        rec_post_idx = int32(P.rec_post_idx(:));
        rec_pre_idx = int32(P.rec_pre_idx(:));
        rec_w = single(P.rec_w(:));
    end
    rec_nnz = int32(numel(rec_w));
else
    W_rec_dense = single(full(P.W_rec));
    rec_post_idx = int32([]);
    rec_pre_idx = int32([]);
    rec_w = single([]);
    rec_nnz = int32(0);
end
end
