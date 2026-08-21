% init_dynamics_gpu.m
function Pg = init_dynamics_gpu(P, opts, steps_override)
%INIT_DYNAMICS_GPU Transfer fixed DS SNN to the resident GPU MEX.
%   The dynamical-system MEX keeps the network state resident across epochs.
%   The mathematical model is the same as the CPU reference; this function
%   only prepares GPU memory, constants and optimizer state.
if nargin < 3 || isempty(steps_override)
    steps = max(2, round(double(opts.T_sim) / double(opts.dt)) + 1);
else
    steps = max(2, round(steps_override));
end
mex_name = 'snn_time_loop_gpu_mex';
if exist(mex_name, 'file') ~= 3
    error('snn_primary_api:mexMissing', 'GPU MEX "%s" is not compiled or not on the MATLAB path.', mex_name);
end
assert_mex_current(mex_name, mex_source_file('dynamical_systems'));
feval(mex_name, 'clear');
compat = P.W_out;
[recurrent_mode_id, decoder_mode_id] = architecture_mode_ids(P);
[W_rec_dense, rec_storage_id, rec_post_idx, rec_pre_idx, rec_w, rec_nnz] = gpu_recurrent_arguments(P);
validate_sparse_gpu_edge_batch(P, int32(1), 'init_dynamics_gpu');
% Initialize resident GPU model with fixed weights, trainable bias, neuron
% constants, input scaling and rollout length.
feval(mex_name, 'init', ...
    P.W_in, P.W_out_base_rec, P.W_out, P.Eta_rec, P.dself, P.B, compat, ...
    P.alpha, P.oneMinusAlpha, P.beta, P.oneMinusBeta, ...
    P.gamma_sr, P.gamma_sd, P.gamma_sr, P.gamma_sd, ...
    P.E_L, P.V_th, P.V_reset, P.a_eff, P.b_param, ...
    P.phi_u, P.delta_u, P.INPUT_SCALE, P.SCALE_rec, ...
    P.spike_jump_sr, P.spike_jump_sr, ...
    int32(steps), int32(P.N_out), int32(P.N_hidden), int32(P.N_rec), ...
    W_rec_dense, recurrent_mode_id, decoder_mode_id, rec_storage_id, ...
    rec_post_idx, rec_pre_idx, rec_w, rec_nnz);
% Initialize GPU-side Adam/AMSGrad state for bias-only learning.
feval(mex_name, 'init_optim', single(opts.adam.b1), single(opts.adam.b2), single(opts.adam.eps), ...
    single(0), single(0.9), single(0.999), single(1e-8), single(0));
Pg = P;
end
