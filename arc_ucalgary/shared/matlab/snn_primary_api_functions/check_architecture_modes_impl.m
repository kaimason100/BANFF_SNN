% check_architecture_modes.m
% Helper for snn_primary_api.

function result = check_architecture_modes_impl(opts)
%CHECK_ARCHITECTURE_MODES Quick sanity checks for supported architectures.
if nargin < 1 || isempty(opts)
    opts = struct();
end
base = merge_options_with_seed(default_static_options("classification", "cpu", "check"), opts);
base.synthetic = true;
base.N_hidden = 32;
base.N_rec = max(4, get_opt(base, 'N_rec', 4));
base.epochs = 1;
base.batch_size = min(4, get_opt(base, 'batch_size', 4));
base.steps_present = min(12, get_opt(base, 'steps_present', 12));
base.steps_avg = min(6, get_opt(base, 'steps_avg', 6));
base.k_avg_start = max(1, base.steps_present - base.steps_avg + 1);

modes = architecture_mode_table();
cpu = repmat(struct('recurrent_mode', "", 'decoder_mode', "", 'signed_decoder_distribution', "", ...
    'recurrent_storage', "", 'loss', single(NaN), ...
    'bias_only', false, 'dale_columns_ok', true, 'diagonal_zero_ok', true, ...
    'signed_decoder_independent', true, 'full_rank_mask_matches_weights', true, ...
    'signed_decoder_mixed_sign', true, 'uniform_decoder_variance_ok', true, ...
    'metadata_ok', true, 'recurrent_current_single', true, ...
    'recurrent_current_orientation_ok', true, 'full_rank_density', single(NaN)), numel(modes), 1);
gpu = repmat(struct('recurrent_mode', "", 'decoder_mode', "", 'signed_decoder_distribution', "", ...
    'recurrent_storage', "", 'status', "skipped", ...
    'loss', single(NaN), 'reason', ""), numel(modes), 1);

default_opts = base;
default_opts = rmfield_if_present(default_opts, 'arch');
explicit_opts = base;
explicit_opts.arch = default_arch_options();
data_default = load_static_data("classification", default_opts);
P_default = make_primary_model(size(data_default.X_train,1), size(data_default.Y_train,1), default_opts);
P_explicit = make_primary_model(size(data_default.X_train,1), size(data_default.Y_train,1), explicit_opts);
default_matches_original = isequaln(P_default.W_in, P_explicit.W_in) && ...
    isequaln(P_default.W_out_base_rec, P_explicit.W_out_base_rec) && ...
    isequaln(P_default.W_out, P_explicit.W_out) && ...
    isequaln(P_default.Eta_rec, P_explicit.Eta_rec) && ...
    isequaln(P_default.dself, P_explicit.dself);

for ii = 1:numel(modes)
    one_opts = base;
    one_opts.arch = modes(ii);
    data = load_static_data("classification", one_opts);
    P = make_primary_model(size(data.X_train,1), size(data.Y_train,1), one_opts);
    P_before = P;
    order = int32(1:size(data.X_train,2));
    [loss_sum, ~, gB] = static_epoch_cpu("classification", data.X_train, data.Y_train, P, one_opts, order, true);
    P = adam_bias_update(P, gB, single(get_lr(1, one_opts.epochs, one_opts.SCHED)), size(data.X_train,2));
    assert_bias_only_update(P_before, P, 'check_architecture_modes cpu');

    cpu(ii).recurrent_mode = one_opts.arch.recurrent_mode;
    cpu(ii).decoder_mode = one_opts.arch.decoder_mode;
    cpu(ii).signed_decoder_distribution = one_opts.arch.signed_decoder_distribution;
    cpu(ii).recurrent_storage = P_before.recurrent_storage;
    cpu(ii).loss = single(loss_sum / max(1, size(data.X_train,2)));
    cpu(ii).bias_only = true;
    r_probe = single((1:P_before.N_hidden).' ./ max(1, P_before.N_hidden));
    I_probe = recurrent_current(P_before, r_probe);
    cpu(ii).recurrent_current_single = isa(I_probe, 'single');
    if one_opts.arch.recurrent_mode == "full_rank"
        cpu(ii).dale_columns_ok = check_dale_columns(P_before.W_rec, P_before.dale_sign, P_before.W_rec_mask);
        cpu(ii).diagonal_zero_ok = all(diag(full(P_before.W_rec)) == 0);
        cpu(ii).full_rank_mask_matches_weights = check_full_rank_mask_matches_weights(P_before);
        cpu(ii).full_rank_density = single(nnz(P_before.W_rec_mask) ./ numel(P_before.W_rec_mask));
        cpu(ii).recurrent_current_orientation_ok = max(abs(double(I_probe(:)) - double(single(P_before.W_rec * r_probe)))) <= 1e-6;
    end
    if one_opts.arch.decoder_mode == "signed"
        shared_decoder = one_opts.SCALE.dec .* P_before.W_out_base_rec(1:P_before.N_out,:);
        cpu(ii).signed_decoder_independent = ~isequaln(P_before.W_out, shared_decoder);
        cpu(ii).signed_decoder_mixed_sign = any(P_before.W_out(:) > 0) && any(P_before.W_out(:) < 0);
        if one_opts.arch.signed_decoder_distribution == "uniform"
            target_var = double(one_opts.SCALE.dec)^2 / double(P_before.N_hidden);
            observed_var = var(double(P_before.W_out(:)), 1);
            cpu(ii).uniform_decoder_variance_ok = abs(observed_var - target_var) <= max(5e-4, 0.35 * target_var);
        end
    end
    result_probe = attach_architecture_metadata(struct(), P_before, one_opts);
    cpu(ii).metadata_ok = isfield(result_probe.arch, 'requested') && ...
        isfield(result_probe.arch, 'resolved') && ...
        result_probe.arch.resolved.recurrent_mode == string(P_before.recurrent_mode) && ...
        result_probe.arch.resolved.decoder_mode == string(P_before.decoder_mode);

    gpu(ii).recurrent_mode = one_opts.arch.recurrent_mode;
    gpu(ii).decoder_mode = one_opts.arch.decoder_mode;
    gpu(ii).signed_decoder_distribution = one_opts.arch.signed_decoder_distribution;
    gpu(ii).recurrent_storage = P_before.recurrent_storage;
    try
        if exist(static_mex_name("classification"), 'file') == 3
            Pg = init_static_gpu("classification", data, P_before, one_opts);
            cleanup = onCleanup(@() clear_static_gpu("classification")); %#ok<NASGU>
            [gpu_loss, ~, ~] = static_train_epoch_gpu("classification", data, 'train', order, Pg, one_opts, 1);
            gpu(ii).status = "ran";
            gpu(ii).loss = single(gpu_loss / max(1, size(data.X_train,2)));
        else
            gpu(ii).reason = "classification GPU MEX is not compiled or not on path";
        end
    catch ME
        gpu(ii).status = "skipped";
        gpu(ii).reason = string(ME.message);
    end
end

primary_step_text = fileread(which('primary_step'));
static_sample_text = fileread(which('static_sample_cpu'));
lsti_used = contains(primary_step_text, 'advance_u_w') && ...
    contains(primary_step_text, 'cascade_advance(P, rho') && ...
    contains(primary_step_text, 'single(1)-rho') && ...
    contains(static_sample_text, 'elig_step(P, elig, rho');

result = struct();
result.options = base;
result.default_matches_original = default_matches_original;
result.cpu = cpu;
result.gpu = gpu;
result.all_cpu_ran = all(isfinite([cpu.loss]));
result.bias_only_ok = all([cpu.bias_only]);
result.full_rank_dale_columns_ok = all([cpu.dale_columns_ok]);
result.full_rank_diagonal_zero_ok = all([cpu.diagonal_zero_ok]);
result.full_rank_mask_matches_weights = all([cpu.full_rank_mask_matches_weights]);
result.signed_decoder_independent = all([cpu.signed_decoder_independent]);
result.signed_decoder_mixed_sign = all([cpu.signed_decoder_mixed_sign]);
result.uniform_decoder_variance_ok = all([cpu.uniform_decoder_variance_ok]);
result.metadata_ok = all([cpu.metadata_ok]);
result.recurrent_current_single_ok = all([cpu.recurrent_current_single]);
result.recurrent_current_orientation_ok = all([cpu.recurrent_current_orientation_ok]);
result.lsti_rho_path_present = lsti_used;
result.sparsity = check_sparsity_controls(base);
result.comparison_modes_preserve_full_rank_options = check_comparison_mode_preservation();
assert(result.default_matches_original, 'Default architecture did not reproduce explicit low_rank/shared initialization.');
assert(result.all_cpu_ran, 'Not all architecture modes ran on CPU.');
assert(result.bias_only_ok, 'Bias-only invariant failed.');
assert(result.full_rank_dale_columns_ok, 'Full-rank W_rec violates Dale signs column-wise.');
assert(result.full_rank_diagonal_zero_ok, 'Full-rank W_rec diagonal is not zero.');
assert(result.full_rank_mask_matches_weights, 'Full-rank W_rec has nonzero weights outside its stored mask.');
assert(result.signed_decoder_independent, 'Signed decoder matched the shared low-rank recurrent decoder.');
assert(result.signed_decoder_mixed_sign, 'Signed decoder did not contain both positive and negative weights.');
assert(result.uniform_decoder_variance_ok, 'Uniform signed decoder variance scaling is outside tolerance.');
assert(result.metadata_ok, 'Architecture metadata fields are missing or inconsistent.');
assert(result.recurrent_current_single_ok, 'recurrent_current did not return single precision for every architecture.');
assert(result.recurrent_current_orientation_ok, 'Full-rank recurrent_current orientation does not match W_rec * r.');
assert(result.lsti_rho_path_present, 'LSTI rho path was not detected in CPU primary/eligibility code.');
assert(result.sparsity.low_rank_zero_p_rec_ok, 'Low-rank opts.NET.p_rec=0 did not zero Eta_rec.');
assert(result.sparsity.full_rank_zero_p_rec_ok, 'Full-rank opts.arch.full_rank_p_rec=0 did not zero W_rec.');
assert(result.sparsity.full_rank_sparse_mask_ok, 'Sparse full-rank W_rec mask/weights failed consistency checks.');
assert(result.comparison_modes_preserve_full_rank_options, 'Architecture comparison modes did not preserve caller full-rank sparsity/storage options.');
end

function modes = architecture_mode_table()
template = default_arch_options();
modes = repmat(template, 9, 1);
modes(1).recurrent_mode = "low_rank";  modes(1).decoder_mode = "shared";
modes(2).recurrent_mode = "low_rank";  modes(2).decoder_mode = "signed"; modes(2).signed_decoder_distribution = "gaussian";
modes(3).recurrent_mode = "low_rank";  modes(3).decoder_mode = "signed"; modes(3).signed_decoder_distribution = "uniform";
modes(4).recurrent_mode = "full_rank"; modes(4).decoder_mode = "shared"; modes(4).full_rank_storage = "dense";
modes(5).recurrent_mode = "full_rank"; modes(5).decoder_mode = "signed"; modes(5).signed_decoder_distribution = "gaussian"; modes(5).full_rank_storage = "dense";
modes(6).recurrent_mode = "full_rank"; modes(6).decoder_mode = "signed"; modes(6).signed_decoder_distribution = "uniform"; modes(6).full_rank_storage = "dense";
modes(7).recurrent_mode = "full_rank"; modes(7).decoder_mode = "shared"; modes(7).full_rank_storage = "sparse"; modes(7).full_rank_p_rec = single(0.25);
modes(8).recurrent_mode = "full_rank"; modes(8).decoder_mode = "signed"; modes(8).signed_decoder_distribution = "gaussian"; modes(8).full_rank_storage = "sparse"; modes(8).full_rank_p_rec = single(0.25);
modes(9).recurrent_mode = "full_rank"; modes(9).decoder_mode = "signed"; modes(9).signed_decoder_distribution = "uniform"; modes(9).full_rank_storage = "sparse"; modes(9).full_rank_p_rec = single(0.25);
end

function ok = check_dale_columns(W_rec, dale_sign, mask)
ok = true;
for jj = 1:size(W_rec, 2)
    active = mask(:, jj);
    if any(active)
        expected = single(dale_sign(jj));
        ok = ok && all(sign(W_rec(active, jj)) == expected);
    end
end
end

function ok = check_full_rank_mask_matches_weights(P)
tol = single(0);
ok = isequal(size(P.W_rec), size(P.W_rec_mask)) && ...
    all(full(P.W_rec(~logical(P.W_rec_mask))) == tol) && ...
    nnz(P.W_rec) <= nnz(P.W_rec_mask);
end

function out = check_sparsity_controls(base)
out = struct();

low_zero = base;
low_zero.NET.p_rec = single(0);
low_zero.arch.recurrent_mode = "low_rank";
P_low_zero = make_primary_model(3, 2, low_zero);
out.low_rank_zero_p_rec_ok = all(P_low_zero.Eta_rec(:) == 0) && all(P_low_zero.dself(:) == 0);

full_zero = base;
full_zero.arch.recurrent_mode = "full_rank";
full_zero.arch.full_rank_p_rec = single(0);
P_full_zero = make_primary_model(3, 2, full_zero);
out.full_rank_zero_p_rec_ok = all(P_full_zero.W_rec(:) == 0) && ~any(P_full_zero.W_rec_mask(:));

full_sparse = base;
full_sparse.arch.recurrent_mode = "full_rank";
full_sparse.arch.full_rank_p_rec = single(0.25);
full_sparse.arch.full_rank_storage = "sparse";
full_sparse.arch.full_rank_remove_self_connections = true;
P_full_sparse = make_primary_model(3, 2, full_sparse);
density = nnz(P_full_sparse.W_rec_mask) ./ numel(P_full_sparse.W_rec_mask);
expected = double(full_sparse.arch.full_rank_p_rec) * (1 - 1 / double(full_sparse.N_hidden));
tolerance = max(0.20, 4 / sqrt(double(numel(P_full_sparse.W_rec_mask))));
out.full_rank_sparse_density = density;
out.full_rank_sparse_expected_density = expected;
out.full_rank_sparse_mask_ok = check_full_rank_mask_matches_weights(P_full_sparse) && ...
    all(diag(P_full_sparse.W_rec_mask) == 0) && ...
    abs(density - expected) <= tolerance;
end

function ok = check_comparison_mode_preservation()
base_arch = default_arch_options();
base_arch.full_rank_p_rec = single(0.05);
base_arch.full_rank_storage = "auto";
base_arch.full_rank_sparse_threshold = single(0.10);
base_arch.max_dense_full_rank_N = int32(1234);
base_arch.max_sparse_full_rank_nnz = int64(9999);
base_arch.max_full_rank_recurrent_bytes = double(123456789);
modes = architecture_comparison_modes("uniform", base_arch);
ok = true;
for ii = 1:numel(modes)
    arch = modes(ii).arch;
    if arch.recurrent_mode == "full_rank"
        ok = ok && arch.full_rank_p_rec == base_arch.full_rank_p_rec && ...
            arch.full_rank_storage == base_arch.full_rank_storage && ...
            arch.full_rank_sparse_threshold == base_arch.full_rank_sparse_threshold && ...
            arch.max_dense_full_rank_N == base_arch.max_dense_full_rank_N && ...
            arch.max_sparse_full_rank_nnz == base_arch.max_sparse_full_rank_nnz && ...
            arch.max_full_rank_recurrent_bytes == base_arch.max_full_rank_recurrent_bytes;
    end
end
end
