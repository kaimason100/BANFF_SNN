% make_primary_model.m
function P = make_primary_model(N_in, N_out, opts)
%MAKE_PRIMARY_MODEL Construct one fixed random SNN plus trainable biases.
%   The architecture is deliberately fixed after initialization: input
%   encoders, recurrent feedback basis, output decoder, recurrent encoder and
%   neuron constants are not trained. Learning changes only P.B, the hidden
%   bias/current offset vector, using e-prop gradients accumulated elsewhere.

opts = normalize_arch_options(opts);
opts = normalize_network_options(opts);
% Task and architecture dimensions. N_rec is the dimensionality of the
% recurrent low-dimensional feedback basis. Shared-decoder mode takes W_out
% from its leading rows; all modes retain N_rec >= N_out for shape-compatible
% architecture comparisons.
N_hidden = opts.N_hidden;
N_rec = opts.N_rec;
if N_rec < N_out
    error('snn_primary_api:badRecurrentReadoutSize N_rec (%d) must be >= task output dimension N_out_task (%d). Increase opts.N_rec or reduce the task dimension.', ...
        N_rec, N_out);
end
dale = opts.NET.dale;
% Generate all fixed random matrices from index-stable random numbers. This
% makes the same seed give the same network across tasks up to submatrix size.
[W_in, W_out_base_rec, W_out, Eta_rec, dself, dale_sign] = init_index_stable( ...
    N_in, N_hidden, N_out, N_rec, dale, opts.SCALE, opts.NET.p_rec, ...
    opts.NET.variance_correction, uint64(get_opt(opts, 'init_seed', opts.seed)));
n = opts.neuron;
% The synaptic current is represented by a rise/decay cascade, which requires
% two positive time constants with rise faster than decay.
if ~(n.tau_s_rise > 0 && n.tau_s_decay > 0 && n.tau_s_rise < n.tau_s_decay)
    error('Require 0 < tau_s_rise < tau_s_decay.');
end
P = struct();
% Fixed structural parameters.
P.W_in = single(W_in);
P.W_out_base_rec = single(W_out_base_rec);
P.W_out = single(W_out);
P.Eta_rec = single(Eta_rec);
P.dself = single(dself);
P.dale_sign = int8(dale_sign(:));
P.N_in = N_in;
P.N_hidden = N_hidden;
P.N_out = N_out;
P.N_rec = N_rec;
P.recurrent_mode = opts.arch.recurrent_mode;
P.decoder_mode = opts.arch.decoder_mode;
P.signed_decoder_distribution = opts.arch.signed_decoder_distribution;
P.full_rank_storage = opts.arch.resolved_storage;
P.recurrent_storage = opts.arch.resolved_storage;
P.full_rank_p_rec = single(opts.arch.full_rank_p_rec);
P.full_rank_sparse_threshold = single(opts.arch.full_rank_sparse_threshold);
P.max_sparse_full_rank_nnz = int64(opts.arch.max_sparse_full_rank_nnz);
P.max_full_rank_recurrent_bytes = double(opts.arch.max_full_rank_recurrent_bytes);
P.arch = opts.arch;
P.arch.full_rank_storage = opts.arch.full_rank_storage;
P.arch.resolved_storage = opts.arch.resolved_storage;
init_seed = uint64(get_opt(opts, 'init_seed', opts.seed));
P.arch_seeds = struct( ...
    'init_seed', init_seed, ...
    'decoder_seed', init_seed + uint64(1009), ...
    'full_rank_mask_seed', init_seed + uint64(2003), ...
    'full_rank_weight_seed', init_seed + uint64(3001));
if opts.arch.decoder_mode == "signed"
    P.W_out = signed_decoder_index_stable(N_out, N_hidden, opts, P.arch_seeds.decoder_seed);
end
if opts.arch.recurrent_mode == "full_rank"
    [P.W_rec, P.W_rec_mask] = full_rank_recurrent_index_stable(P, opts);
    if issparse(P.W_rec)
        [post_idx, pre_idx, w_val] = find(P.W_rec);
        P.rec_post_idx = int32(post_idx(:) - 1);
        P.rec_pre_idx = int32(pre_idx(:) - 1);
        P.rec_w = single(w_val(:));
        P.rec_nnz = int32(numel(w_val));
    else
        P.rec_post_idx = int32([]);
        P.rec_pre_idx = int32([]);
        P.rec_w = single([]);
        P.rec_nnz = int32(0);
    end
else
    P.W_rec = zeros(0, 0, 'single');
    P.W_rec_mask = false(0, 0);
    P.rec_post_idx = int32([]);
    P.rec_pre_idx = int32([]);
    P.rec_w = single([]);
    P.rec_nnz = int32(0);
end
% Trainable parameter and optimizer buffers. P.B is the only model parameter
% updated during training; m_b/v_b/vhat_b are Adam/AMSGrad state for P.B.
P.B = single(zeros(N_hidden,1) + 20);
P.m_b = single(zeros(N_hidden,1));
P.v_b = single(zeros(N_hidden,1));
P.vhat_b = single(zeros(N_hidden,1));
P.t_adam = 0;
P.adam = opts.adam;
% Single input-amplitude normalization used by every task. Data loaders may
% standardize features, but this is the only 1/sqrt(N_in) network scaling.
P.INPUT_SCALE = single(1/sqrt(double(N_in)));
P.SCALE_rec = single(opts.SCALE.rec);
P.dt = single(opts.dt);
% Exact one-step exponential decay factors for membrane, adaptation and
% synaptic filters over one simulation timestep dt.
P.alpha = exp(-opts.dt/n.tau_u);
P.beta = exp(-opts.dt/n.tau_w);
P.gamma_sr = exp(-opts.dt/n.tau_s_rise);
P.gamma_sd = exp(-opts.dt/n.tau_s_decay);
P.oneMinusAlpha = single(1) - P.alpha;
P.oneMinusBeta = single(1) - P.beta;
P.E_L = single(n.E_L);
P.V_th = single(n.V_th);
P.V_reset = single(n.V_reset);
% Retain the optional general coupling field for API compatibility. Active
% publication task scripts set n.a_param to zero, giving pure spike-triggered adaptation.
P.a_eff = max(single(n.a_param), single(0));
P.b_param = single(n.b_param);
% Surrogate derivative parameters: phi_u scales the triangular surrogate,
% delta_u sets its half-width around threshold.
P.phi_u = single(n.phi_u);
P.delta_u = single(n.delta_u);
% Jump size for the rise-filter state corresponding to one spike.
P.spike_jump_sr = -log(max(P.gamma_sr, realmin('single'))) / opts.dt;
end

function W_out = signed_decoder_index_stable(N_out, N_hidden, opts, seed)
dist = string(opts.arch.signed_decoder_distribution);
switch dist
    case "gaussian"
        W = index_stable_gaussian_matrix(N_out, N_hidden, 60, seed);
        W_out = opts.SCALE.dec .* W ./ sqrt(single(N_hidden));
    case "uniform"
        W = index_stable_uniform_minus1_plus1_matrix(N_out, N_hidden, 61, seed);
        W_out = opts.SCALE.dec .* sqrt(single(3)) .* W ./ sqrt(single(N_hidden));
    otherwise
        error('snn_primary_api:archSignedDecoderDistribution', ...
            'Unknown signed decoder distribution "%s".', char(dist));
end
end

function [W_rec, mask] = full_rank_recurrent_index_stable(P, opts)
N = double(P.N_hidden);
storage = string(opts.arch.resolved_storage);
if storage == "dense" && N > double(opts.arch.max_dense_full_rank_N)
    bytes = double(N) * double(N) * 4;
    error('snn_primary_api:denseFullRankTooLarge', ...
        ['Dense full-rank recurrence would allocate %.2f GB for N_hidden=%d. ', ...
        'Use opts.arch.full_rank_storage = "sparse" and reduce opts.arch.full_rank_p_rec.'], ...
        bytes / 2^30, N);
end

if storage == "sparse"
    guard_sparse_full_rank_allocation(P, opts);
    [W_rec, mask] = full_rank_recurrent_sparse(P, opts);
    return;
end

mask = index_stable_bernoulli_matrix(N, N, opts.arch.full_rank_p_rec, 70, P.arch_seeds.full_rank_mask_seed);
if opts.arch.full_rank_remove_self_connections
    mask(1:(N + 1):end) = false;
end
mean_fan_in = max(single(1), single(nnz(mask)) ./ single(N));
W_raw = index_stable_gaussian_matrix(N, N, 71, P.arch_seeds.full_rank_weight_seed);
W_rec = opts.SCALE.rec .* W_raw .* single(mask) ./ sqrt(mean_fan_in);
dale_sign = single(P.dale_sign(:)).';
W_rec = abs(W_rec) .* repmat(dale_sign, N, 1);
W_rec = single(W_rec .* single(mask));
mask = logical(mask);
end

function [W_rec, mask] = full_rank_recurrent_sparse(P, opts)
N = double(P.N_hidden);
p_rec = single(opts.arch.full_rank_p_rec);
guard_sparse_full_rank_allocation(P, opts);
block_cols = max(1, min(N, 512));
rows_cell = {};
cols_cell = {};
vals_cell = {};
mask_rows_cell = {};
mask_cols_cell = {};
nnz_total = 0;
for c0 = 1:block_cols:N
    c1 = min(N, c0 + block_cols - 1);
    cols = c0:c1;
    block_mask = index_stable_bernoulli_matrix(N, numel(cols), p_rec, 70, P.arch_seeds.full_rank_mask_seed, cols);
    if opts.arch.full_rank_remove_self_connections
        local = cols(cols >= 1 & cols <= N);
        block_mask(sub2ind(size(block_mask), local(:), (local(:) - c0 + 1))) = false;
    end
    [ii, jj_local] = find(block_mask);
    if isempty(ii)
        continue;
    end
    jj = cols(jj_local).';
    raw = index_stable_gaussian_values(ii, jj, 71, P.arch_seeds.full_rank_weight_seed);
    dale = single(P.dale_sign(jj)).';
    vals = opts.SCALE.rec .* abs(raw(:)) .* dale(:);
    rows_cell{end+1,1} = uint32(ii(:)); %#ok<AGROW>
    cols_cell{end+1,1} = uint32(jj(:)); %#ok<AGROW>
    vals_cell{end+1,1} = single(vals(:)); %#ok<AGROW>
    mask_rows_cell{end+1,1} = uint32(ii(:)); %#ok<AGROW>
    mask_cols_cell{end+1,1} = uint32(jj(:)); %#ok<AGROW>
    nnz_total = nnz_total + numel(ii);
end
if nnz_total == 0
    W_rec = sparse(N, N);
    mask = sparse(N, N) > 0;
    return;
end
rows = double(vertcat(rows_cell{:}));
cols = double(vertcat(cols_cell{:}));
vals = single(vertcat(vals_cell{:}));
mean_fan_in = max(single(1), single(numel(vals)) ./ single(N));
vals = vals ./ sqrt(mean_fan_in);
% MATLAB sparse construction only supports double/logical numeric storage on
% some releases used on ARC (for example R2023a). Keep the host sparse matrix
% in double precision; GPU execution still receives single-precision edge
% weights through P.rec_w, and CPU recurrence casts the current back to single.
W_rec = sparse(rows, cols, double(vals), N, N);
mask = sparse(double(vertcat(mask_rows_cell{:})), double(vertcat(mask_cols_cell{:})), true, N, N);
end

function guard_sparse_full_rank_allocation(P, opts)
N = double(P.N_hidden);
p_rec = double(opts.arch.full_rank_p_rec);
estimated_nnz = p_rec * N * N;
if opts.arch.full_rank_remove_self_connections
    estimated_nnz = estimated_nnz - p_rec * N;
end
estimated_nnz = max(0, estimated_nnz);

% Conservative host-side estimate: MATLAB sparse W_rec, sparse logical mask,
% and CUDA edge-list copies may coexist in the model struct.
bytes_sparse_w = estimated_nnz * (4 + 8) + double(N + 1) * 8;
bytes_sparse_mask = estimated_nnz * (1 + 8) + double(N + 1) * 8;
bytes_edge_lists = estimated_nnz * (4 + 4 + 4);
estimated_bytes = bytes_sparse_w + bytes_sparse_mask + bytes_edge_lists;

if estimated_nnz > double(opts.arch.max_sparse_full_rank_nnz)
    error('snn_primary_api:sparseFullRankTooManyEdges', ...
        ['Sparse full-rank recurrence is estimated at %.3g edges for N_hidden=%d, p_rec=%.4g, ', ...
         'above opts.arch.max_sparse_full_rank_nnz=%d. Reduce opts.arch.full_rank_p_rec, ', ...
         'use low_rank recurrence, or reduce N_hidden. If this is a GPU run, also consider ', ...
         'a smaller batch size because sparse edge lists are expanded across the batch.'], ...
        estimated_nnz, P.N_hidden, p_rec, opts.arch.max_sparse_full_rank_nnz);
end
if estimated_bytes > double(opts.arch.max_full_rank_recurrent_bytes)
    error('snn_primary_api:sparseFullRankTooLarge', ...
        ['Sparse full-rank recurrence is estimated at %.2f GB for N_hidden=%d, p_rec=%.4g, ', ...
         'above opts.arch.max_full_rank_recurrent_bytes=%.2f GB. Reduce opts.arch.full_rank_p_rec, ', ...
         'use low_rank recurrence, reduce N_hidden, or lower GPU batch size when sparse edge lists are used.'], ...
        estimated_bytes / 2^30, P.N_hidden, p_rec, double(opts.arch.max_full_rank_recurrent_bytes) / 2^30);
end
end

function mask = index_stable_bernoulli_matrix(N_rows, N_cols, p, mat_id, seed, col_ids)
if nargin < 6
    col_ids = 1:N_cols;
end
if p <= 0
    mask = false(N_rows, N_cols);
elseif p >= 1
    mask = true(N_rows, N_cols);
else
    mask = u01_indexstable(mat_id, 1:N_rows, col_ids, seed) < double(p);
end
end

function W = index_stable_uniform_minus1_plus1_matrix(N_rows, N_cols, mat_id, seed)
W = single(2 .* u01_indexstable(mat_id, 1:N_rows, 1:N_cols, seed) - 1);
end

function W = index_stable_gaussian_matrix(N_rows, N_cols, mat_id, seed)
u = u01_indexstable(mat_id, 1:N_rows, 1:N_cols, seed);
u = min(max(u, realmin), 1 - eps);
W = single(sqrt(2) .* erfinv(2 .* u - 1));
end

function W = index_stable_gaussian_values(row_ids, col_ids, mat_id, seed)
u = u01_indexstable_pairs(mat_id, row_ids(:), col_ids(:), seed);
u = min(max(u, realmin), 1 - eps);
W = single(sqrt(2) .* erfinv(2 .* u - 1));
end

function u = u01_indexstable_pairs(mat_id, i, j, seed)
i64 = uint64(i(:));
j64 = uint64(j(:));
A = const64('9E3779B97F4A7C15');
B = const64('BF58476D1CE4E5B9');
C = const64('94D049BB133111EB');
x = add64u(uint64(seed), mul64u(A, uint64(mat_id)+1));
x = add64u(x, mul64u(B, i64));
x = add64u(x, mul64u(C, j64));
k = splitmix64_mod(x);
u = (double(bitshift(k,-11)) + 0.5) / 2^53;
end
