% init_static_gpu.m
function Pg = init_static_gpu(domain, data, P, opts)
%INIT_STATIC_GPU Transfer fixed SNN and static datasets to the GPU MEX.
%   The GPU implementation uses the same model fields as the CPU reference.
%   MATLAB owns the readable structure P; the MEX keeps resident GPU copies
%   for efficient repeated epochs.
mex_name = static_mex_name(domain);
if exist(mex_name, 'file') ~= 3
    error('snn_primary_api:mexMissing', 'GPU MEX "%s" is not compiled or not on the MATLAB path.', mex_name);
end
assert_mex_current(mex_name, mex_source_file(domain));
feval(mex_name, 'clear');
compat = P.W_out;
[recurrent_mode_id, decoder_mode_id] = architecture_mode_ids(P);
[W_rec_dense, rec_storage_id, rec_post_idx, rec_pre_idx, rec_w, rec_nnz] = gpu_recurrent_arguments(P);
batch_size = int32(max(1, opts.batch_size));
validate_sparse_gpu_edge_batch(P, batch_size, 'init_static_gpu');
% Initialize resident GPU model: fixed weights, trainable bias, neuron
% constants, input scaling and readout-window dimensions.
feval(mex_name, 'init', ...
    P.W_in, P.W_out_base_rec, P.W_out, P.Eta_rec, P.dself, P.B, compat, ...
    P.alpha, P.oneMinusAlpha, P.beta, P.oneMinusBeta, ...
    P.gamma_sr, P.gamma_sd, P.gamma_sr, P.gamma_sd, ...
    P.E_L, P.V_th, P.V_reset, P.a_eff, P.b_param, ...
    P.phi_u, P.delta_u, P.INPUT_SCALE, P.SCALE_rec, ...
    P.spike_jump_sr, P.spike_jump_sr, ...
    int32(opts.steps_present), int32(P.N_hidden), int32(P.N_rec), int32(P.N_in), int32(P.N_out), ...
    int32(opts.k_avg_start), int32(opts.steps_avg), batch_size, ...
    W_rec_dense, recurrent_mode_id, decoder_mode_id, rec_storage_id, ...
    rec_post_idx, rec_pre_idx, rec_w, rec_nnz);
% Initialize GPU-side Adam/AMSGrad state for the bias vector.
feval(mex_name, 'init_optim', single(opts.adam.b1), single(opts.adam.b2), single(opts.adam.eps), ...
    single(0), single(0.9), single(0.999), single(1e-8), single(0));
feval(mex_name, 'set_data', 'train', data.X_train, data.Y_train, true);
feval(mex_name, 'set_data', 'val', data.X_val, data.Y_val, true);
feval(mex_name, 'set_data', 'test', data.X_test, data.Y_test, true);
Pg = P;
end
