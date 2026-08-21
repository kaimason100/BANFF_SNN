% Package orientation: SPSA GPU support. This path is separate from the primary e-prop-style bias-training API and should be reviewed as an alternative optimizer implementation.

function result = spsa_gpu_run_task(task_name, varargin)
%SPSA_GPU_RUN_TASK Train one proof task with separate copied GPU MEX loss probes.
%   Supported tasks are "bc", "yacht", and "vanderpol". This function does
%   not call or modify the release GPU training backend; it only uses the
%   copied MEX binaries in spsa_gpu/bin/<mexext>.

cfg = parse_run_options(varargin{:});
repo_root = spsa_gpu_add_paths(fileparts(mfilename('fullpath')));
[kind, domain, opts, model_file] = spsa_gpu_task_options(task_name, repo_root);
if ~isempty(cfg.model_file)
    model_file = cfg.model_file;
end
continuation = spsa_gpu_prepare_continuation(cfg.continue_from, task_name, kind, domain, opts, cfg.additional_epochs);
if continuation.enabled
    opts = continuation.options;
    if isempty(cfg.model_file)
        model_file = spsa_gpu_continuation_model_file(continuation.source_file, cfg.additional_epochs);
    end
end
if ~isempty(cfg.epochs)
    if continuation.enabled
        error('spsa_gpu:epochOverrideWithContinuation', ...
            'Epochs cannot be combined with ContinueFrom; use AdditionalEpochs for continuation jobs.');
    end
    opts.epochs = cfg.epochs;
end
if ~isfield(opts, 'spsa_continuation') || isempty(opts.spsa_continuation)
    % A standard run holds the terminal schedule at opts.epochs after a
    % checkpoint resume; explicit fine-tuning supplies its own boundary.
    opts.spsa_continuation = struct('source_epochs', opts.epochs);
end
opts.spsa_gpu_checks = cfg.checks;
opts.spsa_gpu_compile_note = 'Uses copied resident GPU MEX files under spsa_gpu/bin; release backend files are not modified.';
if cfg.enable_checkpoint
    opts.arc_checkpoint = spsa_gpu_configure_checkpoint(model_file, cfg.submit_script);
end

if kind == "static"
    result = train_static_spsa_gpu(domain, opts, model_file, cfg.run_checks, cfg.run_final_test, continuation);
else
    result = train_dynamics_spsa_gpu(opts, model_file, cfg.run_checks, cfg.run_final_test, continuation);
end
result.model_file = model_file;
result.training.method = 'SPSA_black_box_hidden_bias_only_copied_gpu_mex';
result.training.backend_scope = 'separate_spsa_gpu_module';
result.training.backend_integrity = 'release backend code and MEX outputs are not modified by this module';
result.continuation = continuation.public;

if isfield(result, 'checkpoint') && isfield(result.checkpoint, 'needs_resubmit') && result.checkpoint.needs_resubmit
    return;
end
if cfg.save_model
    model_dir = fileparts(model_file);
    if exist(model_dir, 'dir') ~= 7
        mkdir(model_dir);
    end
    save(model_file, 'result', '-v7.3');
    fprintf('[spsa_gpu] Saved trained network to:\n%s\n', model_file);
end
spsa_gpu_clear_completed_checkpoint(result);
end

function cfg = parse_run_options(varargin)
cfg = struct();
cfg.save_model = true;
cfg.run_checks = true;
cfg.run_final_test = true;
cfg.model_file = '';
cfg.enable_checkpoint = false;
cfg.submit_script = '';
cfg.continue_from = '';
cfg.additional_epochs = [];
cfg.epochs = [];
cfg.checks = struct('max_static_samples', 8, 'max_dynamics_steps', 101, ...
    'loss_abs_tol', single(5e-4), 'output_abs_tol', single(5e-3), ...
    'release_gpu_abs_tol', single(1e-5));
if mod(numel(varargin), 2) ~= 0
    error('spsa_gpu:options', 'Options must be name/value pairs.');
end
for ii = 1:2:numel(varargin)
    key = lower(string(varargin{ii}));
    val = varargin{ii+1};
    switch key
        case {"save_model", "savemodel"}
            cfg.save_model = logical(val);
        case {"run_checks", "runchecks"}
            cfg.run_checks = logical(val);
        case {"run_final_test", "runfinaltest"}
            cfg.run_final_test = logical(val);
        case {"model_file", "modelfile"}
            cfg.model_file = char(val);
        case {"enable_checkpoint", "enablecheckpoint"}
            cfg.enable_checkpoint = logical(val);
        case {"continue_from", "continuefrom", "continuemodelfile"}
            cfg.continue_from = char(val);
        case {"additional_epochs", "additionalepochs"}
            cfg.additional_epochs = double(val);
        case "epochs"
            cfg.epochs = double(val);
        case {"submit_script", "submitscript"}
            cfg.submit_script = char(val);
        case "checks"
            cfg.checks = merge_struct(cfg.checks, val);
        otherwise
            error('spsa_gpu:options', 'Unknown option "%s".', key);
    end
end
if ~isempty(cfg.epochs) && ~(isscalar(cfg.epochs) && isfinite(cfg.epochs) && ...
        cfg.epochs >= 1 && cfg.epochs == floor(cfg.epochs))
    error('spsa_gpu:epochs', 'Epochs must be a positive integer.');
end
end

function continuation = spsa_gpu_prepare_continuation(source_file, task_name, kind, domain, default_opts, additional_epochs)
% Load a completed SPSA result as the starting point of a new fine-tuning
% phase. Saved results retain the selected model but not an optimizer state
% matched to that selected bias vector, so Adam is intentionally reset.
continuation = struct('enabled', false, 'model', struct(), 'history', [], ...
    'spsa_history', struct(), 'closed_loop_validation', struct(), 'best', struct(), ...
    'source_epochs', 0, 'additional_epochs', 0, 'source_file', '', 'options', default_opts, ...
    'public', struct('enabled', false, 'boundaries', zeros(1, 0)));
if isempty(source_file)
    return;
end
if ~(isscalar(additional_epochs) && isfinite(additional_epochs) && additional_epochs >= 1 && additional_epochs == floor(additional_epochs))
    error('spsa_gpu:additionalEpochs', ...
        'AdditionalEpochs must be a positive integer when ContinueFrom is supplied.');
end
if exist(source_file, 'file') ~= 2
    error('spsa_gpu:continuationModelMissing', 'Continuation source file does not exist: %s', source_file);
end
S = load(source_file, 'result');
if ~isfield(S, 'result') || ~isstruct(S.result)
    error('spsa_gpu:continuationInvalid', 'Continuation source must contain a result struct.');
end
source = S.result;
required = {'model', 'best', 'history', 'options'};
for ii = 1:numel(required)
    if ~isfield(source, required{ii})
        error('spsa_gpu:continuationInvalid', 'Continuation result is missing "%s".', required{ii});
    end
end
if kind == "static"
    if ~isfield(source, 'domain') || ~strcmpi(char(source.domain), char(domain))
        error('spsa_gpu:continuationTaskMismatch', 'Continuation source is not a %s SPSA result.', char(domain));
    end
    source_task = char(get_opt(source.options, 'task_tag', ''));
    expected_task = char(get_opt(default_opts, 'task_tag', ''));
    if ~strcmpi(source_task, expected_task)
        error('spsa_gpu:continuationTaskMismatch', ...
            'Continuation source is for task "%s", not "%s".', source_task, expected_task);
    end
else
    source_system = char(get_opt(source.options, 'system_name', ''));
    expected_system = char(get_opt(default_opts, 'system_name', ''));
    if ~strcmpi(source_system, expected_system)
        error('spsa_gpu:continuationTaskMismatch', 'Continuation source is not for the %s dynamical system.', expected_system);
    end
end
source_epochs = spsa_gpu_history_length(source.history);
if source_epochs < 1
    error('spsa_gpu:continuationInvalid', 'Continuation source has no training history.');
end
opts = source.options;
if isfield(opts, 'arc_checkpoint')
    opts = rmfield(opts, 'arc_checkpoint');
end
if isfield(opts, 'spsa_continuation')
    opts = rmfield(opts, 'spsa_continuation');
end
opts.epochs = source_epochs + additional_epochs;
opts.spsa_continuation = struct('source_epochs', source_epochs, ...
    'additional_epochs', additional_epochs, 'schedule', 'hold_terminal_values');
continuation.enabled = true;
continuation.model = source.model;
continuation.history = source.history;
continuation.best = source.best;
continuation.source_epochs = source_epochs;
continuation.additional_epochs = additional_epochs;
continuation.source_file = char(source_file);
continuation.options = opts;
if kind == "dynamics"
    if ~isfield(source, 'spsa_history') || ~isfield(source, 'closed_loop_validation')
        error('spsa_gpu:continuationInvalid', 'Dynamics continuation source is missing SPSA or closed-loop history.');
    end
    continuation.spsa_history = source.spsa_history;
    continuation.closed_loop_validation = source.closed_loop_validation;
end
boundaries = spsa_gpu_continuation_boundaries(source, source_epochs);
continuation.public = struct('enabled', true, 'source_file', char(source_file), ...
    'source_epochs', source_epochs, 'additional_epochs', additional_epochs, ...
    'optimizer_state', 'reset', 'schedule', 'hold_terminal_values', ...
    'starting_model', 'validation_selected_model', 'boundaries', boundaries);
end

function boundaries = spsa_gpu_continuation_boundaries(source, source_epochs)
% Store each phase boundary in cumulative epoch coordinates. The terminal
% result already contains the concatenated history, so this metadata lets
% plots distinguish original training from every later fine-tuning phase.
boundaries = zeros(1, 0);
if isfield(source, 'continuation') && isstruct(source.continuation)
    prior = get_opt(source.continuation, 'boundaries', []);
    if isempty(prior) && logical(get_opt(source.continuation, 'enabled', false))
        prior = get_opt(source.continuation, 'source_epochs', []);
    end
    prior = double(prior(:).');
    boundaries = prior(isfinite(prior) & prior >= 1 & prior < source_epochs);
end
boundaries = unique([boundaries, double(source_epochs)], 'stable');
end

function n = spsa_gpu_history_length(history)
if isnumeric(history)
    n = numel(history);
elseif isstruct(history)
    fields = fieldnames(history);
    if isempty(fields)
        n = 0;
    else
        n = numel(history.(fields{1}));
    end
else
    n = 0;
end
end

function target = spsa_gpu_copy_history_prefix(target, source)
if isnumeric(target) && isnumeric(source)
    n = min(numel(target), numel(source));
    target(1:n) = source(1:n);
    return;
end
if isstruct(target) && isstruct(source)
    fields = fieldnames(target);
    for ii = 1:numel(fields)
        name = fields{ii};
        if isfield(source, name) && isnumeric(target.(name)) && isnumeric(source.(name))
            n = min(numel(target.(name)), numel(source.(name)));
            target.(name)(1:n) = source.(name)(1:n);
        end
    end
end
end

function P = spsa_gpu_reset_adam_state(P)
P.m_b = zeros(size(P.B), 'like', P.B);
P.v_b = zeros(size(P.B), 'like', P.B);
P.vhat_b = zeros(size(P.B), 'like', P.B);
P.t_adam = 0;
end

function model_file = spsa_gpu_continuation_model_file(source_file, additional_epochs)
[folder, base, ext] = fileparts(source_file);
model_file = fullfile(folder, sprintf('%s_continued_%depochs%s', base, additional_epochs, ext));
end

function [kind, domain, opts, model_file] = spsa_gpu_task_options(task_name, repo_root)
task_name = lower(string(task_name));
opts = struct();
opts.seed = 1;
opts.init_seed = opts.seed;
opts.split_seed = 42;
opts.N_hidden = 32000;
opts.N_rec = 10;
opts.SCALE = struct('enc', single(2), 'rec', single(0.05), 'dec', single(0.1));
opts.NET = struct('p_rec', 1, 'variance_correction', true, ...
    'dale', struct('enable', true, 'p_exc', 0.5, 'sign', []));
opts.arch = struct();
opts.arch.recurrent_mode = "low_rank";
opts.arch.decoder_mode = "signed";
opts.arch.signed_decoder_distribution = "uniform";
opts.arch.full_rank_p_rec = single(1.0);
opts.arch.full_rank_remove_self_connections = true;
opts.arch.full_rank_storage = "auto";
opts.arch.full_rank_sparse_threshold = single(0.10);
opts.arch.max_dense_full_rank_N = int32(6000);
opts.arch.max_sparse_full_rank_nnz = int64(20000000);
opts.arch.max_full_rank_recurrent_bytes = double(2.5 * 2^30);
opts.neuron = struct('tau_u', single(50e-3), 'tau_w', single(500e-3), ...
    'tau_s_rise', single(2e-3), 'tau_s_decay', single(50e-3), ...
    'E_L', single(-70), 'V_th', single(-50), 'V_reset', single(-65), ...
    'a_param', single(0), 'b_param', single(0.5), ...
    'phi_u', single(1), 'delta_u', single(0.8));
opts.SCHED = struct('type', 'exponential', 'lr_start', single(5e-2), 'lr_end', single(1e-3));
opts.adam = struct('b1', single(0.9), 'b2', single(0.999), 'eps', single(1e-8));
opts.spsa = struct('c_start', single(0.75), 'c_end', single(0.05), ...
    'gradient_clip_norm', single(inf), 'use_common_random_numbers', true, ...
    'loss_split', 'train');
opts.live_plot = struct('enable', false, 'every', 100);

model_dir = fullfile(repo_root, 'outputs', 'models');
if exist(model_dir, 'dir') ~= 7
    mkdir(model_dir);
end

switch task_name
    case "bc"
        kind = "static";
        domain = "classification";
        opts.task_tag = 'bc';
        opts.dataset_file = 'breast_cancer_dataset.mat';
        opts.dt = single(1e-3);
        opts.PRESENT = struct('T', single(0.300), 'avg_frac', single(0.5));
        opts.steps_present = max(1, round(opts.PRESENT.T / opts.dt));
        opts.steps_avg = max(1, round(opts.PRESENT.avg_frac * opts.steps_present));
        opts.k_avg_start = opts.steps_present - opts.steps_avg + 1;
        opts.epochs = 5000;
        opts.batch_size = 32;
        opts.validate_every = 5;
        model_file = fullfile(model_dir, sprintf('classification_BC_lowrank_SPSA_GPU_primary_seed%03d.mat', opts.seed));
    case "yacht"
        kind = "static";
        domain = "regression";
        opts.task_tag = 'yacht';
        opts.dataset_file = 'yacht_dataset.mat';
        opts.dt = single(1e-3);
        opts.PRESENT = struct('T', single(0.300), 'avg_frac', single(0.5));
        opts.steps_present = max(1, round(opts.PRESENT.T / opts.dt));
        opts.steps_avg = max(1, round(opts.PRESENT.avg_frac * opts.steps_present));
        opts.k_avg_start = opts.steps_present - opts.steps_avg + 1;
        opts.epochs = 50000;
        opts.batch_size = 32;
        opts.validate_every = 5;
        model_file = fullfile(model_dir, sprintf('regression_yacht_lowrank_SPSA_GPU_primary_seed%03d.mat', opts.seed));
    case "vanderpol"
        kind = "dynamics";
        domain = "dynamical_systems"; %#ok<NASGU>
        opts.system_name = 'vanderpol';
        opts.dyn_sys_rate = 8;
        opts.T_sim = single(5);
        opts.long_sim_time = single(2000);
        opts.burn_in_time = single(10);
        opts.train_blocks = 1;
        opts.closed_loop_validate_every = 100;
        opts.closed_loop_validation_time = single(50);
        opts.closed_loop_validation_warmup_time = single(5);
        opts.closed_loop_validation_ics = 5;
        opts.closed_loop_test_time = single(50);
        opts.closed_loop_test_warmup_time = single(5);
        opts.closed_loop_test_ics = 5;
        opts.closed_loop_ic_jitter = single(0.01);
        opts.closed_loop_ic_seed = 1001;
        opts.closed_loop_test_ic_seed = 123;
        opts.dt = single(1e-3);
        opts.use_multistep = true;
        opts.W_warmup = round(0.030 / opts.dt);
        opts.H_free = round(0.055 / opts.dt);
        opts.epochs = 2e5;
        opts.batch_size = 1;
        model_file = fullfile(model_dir, sprintf('dynamical_systems_vanderpol_lowrank_SPSA_GPU_primary_seed%03d.mat', opts.seed));
    otherwise
        error('spsa_gpu:badTask', 'Unknown SPSA GPU task "%s".', task_name);
end
end

function result = train_static_spsa_gpu(domain, opts, model_file, run_checks, run_final_test, continuation)
opts = merge_options_with_seed(default_static_options(domain, "gpu", "train"), opts);
data = load_static_data(domain, opts);
validate_static_data(data, domain, 'spsa_gpu_static');
opts.data_summary = summarize_static_data(data);
P = make_primary_model(size(data.X_train, 1), size(data.Y_train, 1), opts);
P_initial = P;
hist = init_static_history(opts.epochs);
hist.spsa_loss_plus = nan(opts.epochs, 1, 'single');
hist.spsa_loss_minus = nan(opts.epochs, 1, 'single');
hist.spsa_c = nan(opts.epochs, 1, 'single');
hist.spsa_grad_norm = nan(opts.epochs, 1, 'single');
hist.lr = nan(opts.epochs, 1, 'single');
best = struct('loss', inf, 'metric', -inf, 'B', P.B, 'epoch', 0);
start_epoch = 1;
if continuation.enabled
    P = spsa_gpu_reset_adam_state(continuation.model);
    hist = spsa_gpu_copy_history_prefix(hist, continuation.history);
    best = continuation.best;
    start_epoch = continuation.source_epochs + 1;
    fprintf('[spsa_gpu continuation] starting %s fine-tuning from selected epoch %d for %d additional epochs.\n', ...
        char(domain), continuation.best.epoch, continuation.additional_epochs);
end
P_initial = P;
[checkpoint, resumed] = spsa_gpu_load_checkpoint(opts);
if resumed
    validate_spsa_checkpoint(checkpoint, 'static', char(domain));
    P = checkpoint.model;
    hist = checkpoint.history;
    best = checkpoint.best;
    rng(checkpoint.rng_state);
    start_epoch = checkpoint.epoch + 1;
    fprintf('[spsa_gpu checkpoint] resumed %s from epoch %d\n', char(domain), checkpoint.epoch);
end

if run_checks && ~resumed
    equivalence = spsa_gpu_check_static_equivalence(domain, data, P, opts);
else
    equivalence = struct('skipped', true);
end
spsa_gpu_init_static(domain, data, P, opts);
cleanup = onCleanup(@() spsa_gpu_clear_static(domain)); %#ok<NASGU>
timer_id = tic;
for ep = start_epoch:opts.epochs
    lr = spsa_gpu_learning_rate(ep, opts);
    c = spsa_gpu_perturbation_size(ep, opts);
    delta = single(2 .* (rand(P.N_hidden, 1, 'single') > single(0.5)) - 1);
    loss_plus = spsa_gpu_static_loss(domain, P.B + c .* delta, char(opts.spsa.loss_split), opts);
    loss_minus = spsa_gpu_static_loss(domain, P.B - c .* delta, char(opts.spsa.loss_split), opts);
    gB = ((loss_plus - loss_minus) ./ (single(2) .* c)) .* delta;
    gB = spsa_gpu_clip_gradient(gB, opts.spsa.gradient_clip_norm);
    P = adam_bias_update(P, gB, lr, 1);
    train = spsa_gpu_static_validate(domain, P.B, 'train', opts);
    hist.train_loss(ep) = single(train.loss);
    hist.train_metric(ep) = single(train.metric);
    if should_validate(ep, opts)
        val = spsa_gpu_static_validate(domain, P.B, 'val', opts);
        hist.val_loss(ep) = single(val.loss);
        hist.val_metric(ep) = single(val.metric);
        if is_better(domain, val.loss, val.metric, best.loss, best.metric)
            best = struct('loss', val.loss, 'metric', val.metric, 'B', P.B, 'epoch', ep);
        end
    end
    hist.spsa_loss_plus(ep) = single(loss_plus);
    hist.spsa_loss_minus(ep) = single(loss_minus);
    hist.spsa_c(ep) = single(c);
    hist.spsa_grad_norm(ep) = single(norm(double(gB)));
    hist.lr(ep) = single(lr);
    print_static_progress(domain, hist, ep, opts, best);
    if spsa_gpu_checkpoint_due(opts, timer_id, ep)
        checkpoint = make_spsa_checkpoint('static', char(domain), ep, P, best, hist, opts);
        spsa_gpu_save_checkpoint(opts, checkpoint);
        result = package_static_result(domain, 'gpu', P, best, hist, struct(), opts);
        result.equivalence_checks = equivalence;
        result.checkpoint = spsa_gpu_checkpoint_public_info(opts, checkpoint, true);
        return;
    end
end

Pbest = P;
Pbest.B = best.B;
assert_bias_only_update(P_initial, P, 'spsa_gpu_static');
if run_final_test
    test = spsa_gpu_static_predict(domain, data, Pbest.B, 'test', opts);
else
    test = struct('status', 'not_run_on_arc', ...
        'message', 'ARC is training-only; run the saved-model test locally.');
end
result = package_static_result(domain, 'gpu', P, best, hist, test, opts);
result.training.optimizer = 'SPSA two-sided scalar loss probe with existing Adam bias update';
result.training.trainable_parameters = 'hidden_bias_only';
result.training.spsa = opts.spsa;
result.equivalence_checks = equivalence;
result.checkpoint = spsa_gpu_checkpoint_public_info(opts, struct(), false);
end

function result = train_dynamics_spsa_gpu(opts, model_file, run_checks, run_final_test, continuation)
opts = merge_options_with_seed(default_dynamics_options("gpu", "train"), opts);
opts.dynamics_split = 'train';
[x, P] = make_dynamics_problem(opts);
lambda = make_lambda_sequence_for_data(x, opts);
validate_dynamics_data(x, lambda, 'spsa_gpu_dynamics');
P_initial = P;
hist = zeros(opts.epochs, 1, 'single');
closed_hist = init_closed_loop_history(opts.epochs);
spsa_hist = struct('loss_plus', nan(opts.epochs, 1, 'single'), ...
    'loss_minus', nan(opts.epochs, 1, 'single'), 'c', nan(opts.epochs, 1, 'single'), ...
    'grad_norm', nan(opts.epochs, 1, 'single'), 'lr', nan(opts.epochs, 1, 'single'));
best = struct('loss', inf, 'wd', inf, 'B', P.B, 'epoch', 0);
start_epoch = 1;
if continuation.enabled
    P = spsa_gpu_reset_adam_state(continuation.model);
    hist = spsa_gpu_copy_history_prefix(hist, continuation.history);
    spsa_hist = spsa_gpu_copy_history_prefix(spsa_hist, continuation.spsa_history);
    closed_hist = spsa_gpu_copy_history_prefix(closed_hist, continuation.closed_loop_validation);
    best = continuation.best;
    start_epoch = continuation.source_epochs + 1;
    fprintf('[spsa_gpu continuation] starting dynamics fine-tuning from selected epoch %d for %d additional epochs.\n', ...
        continuation.best.epoch, continuation.additional_epochs);
end
P_initial = P;
[checkpoint, resumed] = spsa_gpu_load_checkpoint(opts);
if resumed
    validate_spsa_checkpoint(checkpoint, 'dynamics', 'dynamical_systems');
    P = checkpoint.model;
    hist = checkpoint.history;
    spsa_hist = checkpoint.spsa_history;
    closed_hist = checkpoint.closed_loop_validation;
    best = checkpoint.best;
    rng(checkpoint.rng_state);
    start_epoch = checkpoint.epoch + 1;
    fprintf('[spsa_gpu checkpoint] resumed dynamics from epoch %d\n', checkpoint.epoch);
end
if run_checks && ~resumed
    equivalence = spsa_gpu_check_dynamics_equivalence(x, lambda, P, opts);
else
    equivalence = struct('skipped', true);
end
spsa_gpu_init_dynamics(P, opts, max_sequence_steps(x));
cleanup = onCleanup(@() spsa_gpu_clear_dynamics()); %#ok<NASGU>
opts_closed = opts;
opts_closed.T_sim = single(get_opt(opts, 'closed_loop_validation_time', opts.T_sim));
opts_closed.closed_loop_warmup_time = single(get_opt(opts, ...
    'closed_loop_validation_warmup_time', get_opt(opts, 'closed_loop_test_warmup_time', 0)));
closed_eval_set = make_closed_loop_eval_set(opts_closed);
spsa_gpu_assert_closed_loop_eval_set(closed_eval_set, opts_closed, P.N_out);
timer_id = tic;
for ep = start_epoch:opts.epochs
    lr = spsa_gpu_learning_rate(ep, opts);
    c = spsa_gpu_perturbation_size(ep, opts);
    delta = single(2 .* (rand(P.N_hidden, 1, 'single') > single(0.5)) - 1);
    starts = spsa_gpu_dynamics_epoch_starts(x);
    loss_plus = spsa_gpu_dynamics_loss(x, lambda, P.B + c .* delta, opts, starts);
    loss_minus = spsa_gpu_dynamics_loss(x, lambda, P.B - c .* delta, opts, starts);
    gB = ((loss_plus - loss_minus) ./ (single(2) .* c)) .* delta;
    gB = spsa_gpu_clip_gradient(gB, opts.spsa.gradient_clip_norm);
    P = adam_bias_update(P, gB, lr, 1);
    hist(ep) = spsa_gpu_dynamics_loss(x, lambda, P.B, opts, starts);
    spsa_hist.loss_plus(ep) = single(loss_plus);
    spsa_hist.loss_minus(ep) = single(loss_minus);
    spsa_hist.c(ep) = single(c);
    spsa_hist.grad_norm(ep) = single(norm(double(gB)));
    spsa_hist.lr(ep) = single(lr);
    if spsa_gpu_checkpoint_due(opts, timer_id, ep)
        checkpoint = make_spsa_checkpoint('dynamics', 'dynamical_systems', ep, P, best, hist, opts);
        checkpoint.spsa_history = spsa_hist;
        checkpoint.closed_loop_validation = closed_hist;
        spsa_gpu_save_checkpoint(opts, checkpoint);
        result = package_dynamics_result(P, best, hist, closed_hist, spsa_hist, x, opts, equivalence);
        result.checkpoint = spsa_gpu_checkpoint_public_info(opts, checkpoint, true);
        result.model_file = model_file;
        return;
    end
    if should_closed_loop_validate(ep, opts)
        closed = spsa_gpu_dynamics_closed_loop(P, opts, closed_eval_set);
        % Closed-loop validation uses a longer rollout and therefore
        % reinitializes the resident MEX with a different sequence length.
        % Restore the training-sized resident MEX before the next epoch.
        spsa_gpu_init_dynamics(P, opts, max_sequence_steps(x));
        closed_hist.wd(ep) = single(closed.wasserstein_distance);
        if closed.wasserstein_distance < best.wd
            best = struct('loss', hist(ep), 'wd', closed.wasserstein_distance, 'B', P.B, 'epoch', ep);
        end
    elseif hist(ep) < best.loss && ~isfinite(best.wd)
        best = struct('loss', hist(ep), 'wd', inf, 'B', P.B, 'epoch', ep);
    end
    print_dynamics_progress(hist, closed_hist, spsa_hist, ep, opts, best);
    if spsa_gpu_checkpoint_due(opts, timer_id, ep)
        checkpoint = make_spsa_checkpoint('dynamics', 'dynamical_systems', ep, P, best, hist, opts);
        checkpoint.spsa_history = spsa_hist;
        checkpoint.closed_loop_validation = closed_hist;
        spsa_gpu_save_checkpoint(opts, checkpoint);
        result = package_dynamics_result(P, best, hist, closed_hist, spsa_hist, x, opts, equivalence);
        result.checkpoint = spsa_gpu_checkpoint_public_info(opts, checkpoint, true);
        result.model_file = model_file;
        return;
    end
end
if best.epoch == 0
    best = struct('loss', hist(end), 'wd', inf, 'B', P.B, 'epoch', opts.epochs);
end
assert_bias_only_update(P_initial, P, 'spsa_gpu_dynamics');
result = package_dynamics_result(P, best, hist, closed_hist, spsa_hist, x, opts, equivalence);
if run_final_test
    opts_test = opts;
    opts_test.T_sim = single(get_opt(opts, 'closed_loop_test_time', opts.T_sim));
    opts_test.closed_loop_warmup_time = single(get_opt(opts, 'closed_loop_test_warmup_time', 5));
    opts_test.closed_loop_validation_ics = get_opt(opts, 'closed_loop_test_ics', get_opt(opts, 'closed_loop_validation_ics', 1));
    opts_test.closed_loop_ic_seed = get_opt(opts, 'closed_loop_test_ic_seed', get_opt(opts, 'closed_loop_ic_seed', 1001));
    opts_test.closed_loop_ic_include_reference = logical(get_opt(opts, 'closed_loop_test_include_reference', false));
    opts_test.closed_loop_ic_role = 'test';
    test_eval_set = make_closed_loop_eval_set(opts_test);
    assert_closed_loop_test_ics_held_out(test_eval_set, opts);
    result.test = spsa_gpu_dynamics_closed_loop(result.model, opts_test, test_eval_set);
    result.test.test_initial_conditions_held_out_from_validation = true;
    result.test.validation_test_initial_condition_overlap_count = 0;
else
    result.test = struct('status', 'not_run_on_arc', ...
        'message', 'ARC is training-only; run the saved-model test locally.');
end
result.checkpoint = spsa_gpu_checkpoint_public_info(opts, struct(), false);
end

function result = package_dynamics_result(P, best, hist, closed_hist, spsa_hist, x, opts, equivalence)
Pbest = P;
Pbest.B = best.B;
meta = primary_bias_training_metadata();
meta.mex = spsa_gpu_mex_metadata('dynamical_systems');
result = struct('backend', 'gpu', 'history', hist, 'best', best, ...
    'closed_loop_validation', closed_hist, 'final_B', P.B, 'model', Pbest, ...
    'options', opts, 'training_metadata', meta);
result.model.recurrent_mode = Pbest.recurrent_mode;
result.model.decoder_mode = Pbest.decoder_mode;
result.dynamics = dynamics_training_metadata(x, opts);
result.training = struct('trainable_parameters', 'hidden_bias_only', ...
    'optimizer', 'SPSA two-sided scalar loss probe with existing Adam bias update', ...
    'spsa', opts.spsa);
result.spsa_history = spsa_hist;
result.equivalence_checks = equivalence;
result = attach_architecture_metadata(result, Pbest, opts);
end

function spsa_gpu_init_static(domain, data, P, opts)
mex_name = spsa_gpu_mex_name(domain);
require_spsa_mex(mex_name);
feval(mex_name, 'clear');
compat = P.W_out;
[recurrent_mode_id, decoder_mode_id] = architecture_mode_ids(P);
[W_rec_dense, rec_storage_id, rec_post_idx, rec_pre_idx, rec_w, rec_nnz] = gpu_recurrent_arguments(P);
batch_size = int32(max(1, opts.batch_size));
validate_sparse_gpu_edge_batch(P, batch_size, 'spsa_gpu_init_static');
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
feval(mex_name, 'init_optim', single(opts.adam.b1), single(opts.adam.b2), single(opts.adam.eps), ...
    single(0), single(0.9), single(0.999), single(1e-8), single(0));
feval(mex_name, 'set_data', 'train', data.X_train, data.Y_train, true);
feval(mex_name, 'set_data', 'val', data.X_val, data.Y_val, true);
feval(mex_name, 'set_data', 'test', data.X_test, data.Y_test, true);
end

function out = spsa_gpu_static_validate(domain, B, split, opts)
mex_name = spsa_gpu_mex_name(domain);
feval(mex_name, 'update_bias', single(B));
[loss_sum, metric_raw, count] = feval(mex_name, 'validate_primary', char(split), int32(opts.batch_size));
out = struct('loss', single(loss_sum ./ max(1, double(count))), ...
    'metric', single(normalize_static_metric(domain, metric_raw, count)), 'count', count);
end

function loss = spsa_gpu_static_loss(domain, B, split, opts)
out = spsa_gpu_static_validate(domain, B, split, opts);
loss = single(out.loss);
end

function pred = spsa_gpu_static_predict(domain, data, B, split, opts)
mex_name = spsa_gpu_mex_name(domain);
feval(mex_name, 'update_bias', single(B));
[loss_sum, metric_raw, count, Z] = feval(mex_name, 'predict_primary', char(split), int32(opts.batch_size));
[~, Y] = split_arrays(data, split);
pred = struct('loss', single(loss_sum ./ max(1, double(count))), ...
    'metric', single(normalize_static_metric(domain, metric_raw, count)), ...
    'count', count, 'Z', single(Z), 'Y', single(Y));
if domain == "regression"
    pred = attach_regression_test_stats(pred, data, split);
end
end

function spsa_gpu_init_dynamics(P, opts, steps_override)
mex_name = spsa_gpu_mex_name("dynamical_systems");
require_spsa_mex(mex_name);
if nargin < 3 || isempty(steps_override)
    steps = max(2, round(double(opts.T_sim) / double(opts.dt)) + 1);
else
    steps = max(2, round(steps_override));
end
feval(mex_name, 'clear');
compat = P.W_out;
[recurrent_mode_id, decoder_mode_id] = architecture_mode_ids(P);
[W_rec_dense, rec_storage_id, rec_post_idx, rec_pre_idx, rec_w, rec_nnz] = gpu_recurrent_arguments(P);
validate_sparse_gpu_edge_batch(P, int32(1), 'spsa_gpu_init_dynamics');
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
feval(mex_name, 'init_optim', single(opts.adam.b1), single(opts.adam.b2), single(opts.adam.eps), ...
    single(0), single(0.9), single(0.999), single(1e-8), single(0));
end

function starts = spsa_gpu_dynamics_epoch_starts(x)
if isstruct(x) && isfield(x, 'pool')
    starts = zeros(1, x.train_blocks, 'uint32');
    for bb = 1:x.train_blocks
        starts(bb) = randi(x.max_start_idx, 1, 'uint32');
    end
else
    starts = uint32(1);
end
end

function loss = spsa_gpu_dynamics_loss(x, lambda, B, opts, starts)
mex_name = spsa_gpu_mex_name("dynamical_systems");
feval(mex_name, 'update_bias', single(B));
loss_sum = single(0);
count_sum = 0;
if isstruct(x) && isfield(x, 'pool')
    for bb = 1:numel(starts)
        xb = x.pool(:, double(starts(bb)):double(starts(bb)) + x.steps - 1);
        [~, block_loss, block_count] = feval(mex_name, 'run_primary_eval', single(xb), logical(lambda));
        loss_sum = loss_sum + single(block_loss);
        count_sum = count_sum + double(block_count);
    end
elseif iscell(x)
    for bb = 1:numel(x)
        [~, block_loss, block_count] = feval(mex_name, 'run_primary_eval', single(x{bb}), logical(lambda{bb}));
        loss_sum = loss_sum + single(block_loss);
        count_sum = count_sum + double(block_count);
    end
else
    [~, loss_sum, count_sum] = feval(mex_name, 'run_primary_eval', single(x), logical(lambda));
end
loss = single(loss_sum ./ max(1, count_sum));
end

function [summary, B_gpu] = spsa_gpu_dynamics_eval_matrix(x, lambda, B)
mex_name = spsa_gpu_mex_name("dynamical_systems");
feval(mex_name, 'update_bias', single(B));
[Z, loss_sum, count] = feval(mex_name, 'run_primary_eval', single(x), logical(lambda));
Z = valid_dynamics_predictions(single(Z), size(x, 2));
summary = struct('loss', single(loss_sum ./ max(1, double(count))), ...
    'Z', single(Z), 'num_valid_prediction_columns', size(Z, 2));
B_gpu = feval(mex_name, 'get_bias');
end

function closed = spsa_gpu_dynamics_closed_loop(P, opts, eval_set)
if nargin < 3 || isempty(eval_set)
    eval_set = make_closed_loop_eval_set(opts);
end
spsa_gpu_init_dynamics(P, opts, max(cellfun(@(xx) size(xx, 2), eval_set.x_true)));
n_ic = numel(eval_set.x_true);
wd_by_ic = nan(n_ic, 1, 'single');
pred_by_ic = cell(n_ic, 1);
truth_by_ic = cell(n_ic, 1);
test_x0_norm_by_ic = cell(n_ic, 1);
truth_diagnostic_by_ic = cell(n_ic, 1);
for ic = 1:n_ic
    [summary, ~] = spsa_gpu_dynamics_eval_matrix(eval_set.x_true{ic}, eval_set.lambda{ic}, P.B);
    pred = single(summary.Z(:, 1:summary.num_valid_prediction_columns).');
    [pred, truth, test_x0_norm, truth_diagnostic] = spsa_gpu_closed_loop_test_segment(pred, eval_set, ic);
    wd_by_ic(ic) = single(phase_portrait_wasserstein_distance(double(pred), double(truth), opts.wd));
    pred_by_ic{ic} = pred;
    truth_by_ic{ic} = truth;
    test_x0_norm_by_ic{ic} = test_x0_norm;
    truth_diagnostic_by_ic{ic} = truth_diagnostic;
end
closed = struct();
closed.wasserstein_distance = single(mean(wd_by_ic, 'omitnan'));
closed.wasserstein_distance_by_ic = wd_by_ic;
closed.x0_list = eval_set.x0_list;
closed.test_x0_norm_by_ic = test_x0_norm_by_ic;
closed.truth_diagnostic_by_ic = truth_diagnostic_by_ic;
closed.truth_simulation_failed_by_ic = cellfun(@spsa_gpu_truth_diagnostic_failed, truth_diagnostic_by_ic);
closed.closed_loop_ic_seed = spsa_gpu_eval_set_field(eval_set, 'closed_loop_ic_seed', []);
closed.closed_loop_test_ic_seed = closed.closed_loop_ic_seed;
closed.closed_loop_ic_jitter = spsa_gpu_eval_set_field(eval_set, 'closed_loop_ic_jitter', []);
closed.closed_loop_ic_include_reference = logical(spsa_gpu_eval_set_field(eval_set, 'closed_loop_ic_include_reference', true));
closed.closed_loop_ic_role = char(spsa_gpu_eval_set_field(eval_set, 'closed_loop_ic_role', 'unspecified'));
closed.closed_loop_warmup_time = single(spsa_gpu_eval_set_field(eval_set, 'warmup_time', 0));
closed.closed_loop_test_time = single(spsa_gpu_eval_set_field(eval_set, 'test_time', NaN));
closed.closed_loop_warmup_steps = int32(spsa_gpu_eval_set_field(eval_set, 'warmup_steps', 0));
closed.closed_loop_truth_initialization = 'network_terminal_warmup_state';
closed.pred_norm = pred_by_ic{1};
closed.true_norm = truth_by_ic{1};
closed.pred_norm_by_ic = pred_by_ic;
closed.true_norm_by_ic = truth_by_ic;
closed.n_initial_conditions = n_ic;
closed.gpu_bias = single(P.B);
end

function spsa_gpu_assert_closed_loop_eval_set(eval_set, opts_closed, n_out)
% Fail before training if an ARC transfer mixed incompatible closed-loop code.
required_functions = {'make_closed_loop_eval_set', 'closed_loop_truth_from_network_state', ...
    'assert_closed_loop_test_ics_held_out'};
for ii = 1:numel(required_functions)
    if exist(required_functions{ii}, 'file') ~= 2
        error('spsa_gpu:closedLoopDependencyMissing', ...
            'Missing closed-loop dependency "%s". Transfer the matched shared/matlab/snn_primary_api_functions files.', ...
            required_functions{ii});
    end
end
required_fields = {'x_true', 'warmup_steps', 'test_time', 'test_x0_norm'};
for ii = 1:numel(required_fields)
    if ~isfield(eval_set, required_fields{ii})
        error('spsa_gpu:closedLoopEvaluationSetOutdated', ...
            ['Closed-loop evaluation code is outdated: eval_set lacks "%s". ', ...
             'Transfer make_closed_loop_eval_set.m with spsa_gpu_run_task.m.'], required_fields{ii});
    end
end
expected_warmup_steps = max(0, round(double(get_opt(opts_closed, 'closed_loop_warmup_time', 0)) / double(opts_closed.dt)));
if double(eval_set.warmup_steps) ~= expected_warmup_steps
    error('spsa_gpu:closedLoopWarmupMismatch', ...
        'Closed-loop evaluation set has %d warmup steps; SPSA requested %d. Transfer matching SPSA/shared files.', ...
        double(eval_set.warmup_steps), expected_warmup_steps);
end
if abs(double(eval_set.test_time) - double(opts_closed.T_sim)) > max(1e-6, 10 * eps(double(opts_closed.T_sim)))
    error('spsa_gpu:closedLoopTestTimeMismatch', ...
        'Closed-loop evaluation set test time does not match the SPSA validation time. Transfer matching SPSA/shared files.');
end
n_ic = numel(eval_set.x_true);
if n_ic < 1 || ~iscell(eval_set.test_x0_norm) || numel(eval_set.test_x0_norm) ~= n_ic
    error('spsa_gpu:closedLoopInitialStatesInvalid', ...
        'Closed-loop evaluation set has invalid test initial states. Transfer make_closed_loop_eval_set.m.');
end
for ic = 1:n_ic
    x0 = eval_set.test_x0_norm{ic};
    if numel(x0) ~= n_out || any(~isfinite(x0(:)))
        error('spsa_gpu:closedLoopInitialStateDimension', ...
            'Closed-loop test initial condition %d must be a finite %d-element state.', ic, n_out);
    end
    if size(eval_set.x_true{ic}, 2) <= expected_warmup_steps + 1
        error('spsa_gpu:closedLoopEvaluationTooShort', ...
            'Closed-loop evaluation trajectory %d is shorter than its warmup interval.', ic);
    end
end
end

function [pred_trim, truth_trim, test_x0_norm, truth_diagnostic] = spsa_gpu_closed_loop_test_segment(pred, eval_set, ic)
warmup_steps = max(0, round(double(spsa_gpu_eval_set_field(eval_set, 'warmup_steps', 0))));
if warmup_steps >= size(pred, 1)
    error('spsa_gpu:closedLoopWarmupTooLong', ...
        'Closed-loop warmup requires one of %d prediction steps as its terminal state. Reduce closed_loop_warmup_time or increase test duration.', size(pred, 1));
end

% Warm up only the autonomous network. The true system then starts from the
% network's decoded terminal warmup state, so both scored trajectories share
% the same initial condition at the start of the scored interval.
if warmup_steps > 0
    pred_trim = pred(warmup_steps + 1:end, :);
    test_x0_norm = single(pred(warmup_steps, :).');
else
    pred_trim = pred;
    initial_states = spsa_gpu_eval_set_field(eval_set, 'test_x0_norm', {});
    if iscell(initial_states) && numel(initial_states) >= ic && ~isempty(initial_states{ic})
        test_x0_norm = single(initial_states{ic});
    else
        test_x0_norm = single([]);
    end
end

[truth_trim, truth_diagnostic] = closed_loop_truth_from_network_state(test_x0_norm, eval_set);
if isempty(truth_trim)
    % Preserve the prediction length for diagnostics; the non-finite truth
    % continuation is represented by an infinite Wasserstein distance.
    truth_trim = nan(size(pred_trim), 'single');
    return;
end
n = min(size(pred_trim, 1), size(truth_trim, 1));
if n < 1
    error('spsa_gpu:emptyClosedLoopTestSegment', ...
        'Closed-loop test segment is empty after warmup trimming.');
end
pred_trim = pred_trim(1:n, :);
truth_trim = truth_trim(1:n, :);
end

function value = spsa_gpu_eval_set_field(eval_set, name, default_value)
if isstruct(eval_set) && isfield(eval_set, name) && ~isempty(eval_set.(name))
    value = eval_set.(name);
else
    value = default_value;
end
end

function tf = spsa_gpu_truth_diagnostic_failed(diagnostic)
tf = isstruct(diagnostic) && isfield(diagnostic, 'status') && ...
    ~(strcmp(diagnostic.status, 'ok') || strcmp(diagnostic.status, 'not_required'));
end

function checks = spsa_gpu_check_static_equivalence(domain, data, P, opts)
checks = struct('task_kind', 'static', 'domain', char(domain));
n = min(size(data.X_train, 2), round(double(opts.spsa_gpu_checks.max_static_samples)));
data_small = data;
data_small.X_train = data.X_train(:, 1:n);
data_small.Y_train = data.Y_train(:, 1:n);
data_small.X_val = data_small.X_train;
data_small.Y_val = data_small.Y_train;
data_small.X_test = data_small.X_train;
data_small.Y_test = data_small.Y_train;
[cpu, ~] = static_eval_cpu(domain, data_small.X_train, data_small.Y_train, P, opts);
spsa_gpu_init_static(domain, data_small, P, opts);
spsa = spsa_gpu_static_predict(domain, data_small, P.B, 'train', opts);
checks.cpu_vs_spsa_gpu_loss_abs = abs(double(cpu.loss) - double(spsa.loss));
checks.cpu_vs_spsa_gpu_output_abs = max(abs(double(cpu.Z(:)) - double(spsa.Z(:))), [], 'omitnan');
assert_equivalence(checks.cpu_vs_spsa_gpu_loss_abs, opts.spsa_gpu_checks.loss_abs_tol, 'CPU vs SPSA GPU static loss');
assert_equivalence(checks.cpu_vs_spsa_gpu_output_abs, opts.spsa_gpu_checks.output_abs_tol, 'CPU vs SPSA GPU static outputs');
release_mex = static_mex_name(domain);
if exist(release_mex, 'file') == 3
    release = static_eval_gpu(domain, data_small, P, opts, 'train');
    checks.release_gpu_available = true;
    checks.release_vs_spsa_gpu_loss_abs = abs(double(release.loss) - double(spsa.loss));
    checks.release_vs_spsa_gpu_output_abs = max(abs(double(release.Z(:)) - double(spsa.Z(:))), [], 'omitnan');
    assert_equivalence(checks.release_vs_spsa_gpu_loss_abs, opts.spsa_gpu_checks.release_gpu_abs_tol, 'Release GPU vs SPSA GPU static loss');
    assert_equivalence(checks.release_vs_spsa_gpu_output_abs, opts.spsa_gpu_checks.output_abs_tol, 'Release GPU vs SPSA GPU static outputs');
else
    checks.release_gpu_available = false;
end
end

function checks = spsa_gpu_check_dynamics_equivalence(x, lambda, P, opts)
checks = struct('task_kind', 'dynamics', 'domain', 'dynamical_systems');
[xc, lc] = small_dynamics_sequence(x, lambda, opts);
[cpu_loss_sum, ~, cpu_Z] = dynamics_epoch_cpu(xc, lc, P, opts, false);
cpu_loss = single(cpu_loss_sum ./ max(1, size(xc, 2) - 1));
spsa_gpu_init_dynamics(P, opts, size(xc, 2));
[spsa, ~] = spsa_gpu_dynamics_eval_matrix(xc, lc, P.B);
checks.cpu_vs_spsa_gpu_loss_abs = abs(double(cpu_loss) - double(spsa.loss));
checks.cpu_vs_spsa_gpu_output_abs = max(abs(double(cpu_Z(:)) - double(spsa.Z(:))), [], 'omitnan');
assert_equivalence(checks.cpu_vs_spsa_gpu_loss_abs, opts.spsa_gpu_checks.loss_abs_tol, 'CPU vs SPSA GPU dynamics loss');
assert_equivalence(checks.cpu_vs_spsa_gpu_output_abs, opts.spsa_gpu_checks.output_abs_tol, 'CPU vs SPSA GPU dynamics outputs');
if exist('snn_time_loop_gpu_mex', 'file') == 3
    [release, ~] = dynamics_eval_gpu(xc, lc, P, opts);
    checks.release_gpu_available = true;
    checks.release_vs_spsa_gpu_loss_abs = abs(double(release.loss) - double(spsa.loss));
    checks.release_vs_spsa_gpu_output_abs = max(abs(double(release.Z(:)) - double(spsa.Z(:))), [], 'omitnan');
    assert_equivalence(checks.release_vs_spsa_gpu_loss_abs, opts.spsa_gpu_checks.release_gpu_abs_tol, 'Release GPU vs SPSA GPU dynamics loss');
    assert_equivalence(checks.release_vs_spsa_gpu_output_abs, opts.spsa_gpu_checks.output_abs_tol, 'Release GPU vs SPSA GPU dynamics outputs');
else
    checks.release_gpu_available = false;
end
end

function [xc, lc] = small_dynamics_sequence(x, lambda, opts)
n = max(2, min(max_sequence_steps(x), round(double(opts.spsa_gpu_checks.max_dynamics_steps))));
if isstruct(x) && isfield(x, 'pool')
    xc = x.pool(:, 1:n);
    lc = lambda(1:n);
elseif iscell(x)
    xc = x{1}(:, 1:min(n, size(x{1}, 2)));
    lc = lambda{1}(1:size(xc, 2));
else
    xc = x(:, 1:min(n, size(x, 2)));
    lc = lambda(1:size(xc, 2));
end
end

function assert_equivalence(value, tol, label)
if ~(isfinite(value) && value <= double(tol))
    error('spsa_gpu:equivalenceCheckFailed', '%s check failed: value %.6g exceeds tolerance %.6g.', ...
        label, value, double(tol));
end
end

function cfg = spsa_gpu_configure_checkpoint(model_file, submit_script)
[model_dir, model_base] = fileparts(model_file);
checkpoint_dir = fullfile(model_dir, '..', 'checkpoints');
if exist(checkpoint_dir, 'dir') ~= 7
    mkdir(checkpoint_dir);
end
checkpoint_hours = str2double(getenv('SPSA_CHECKPOINT_HOURS'));
if ~isfinite(checkpoint_hours) || checkpoint_hours <= 0
    checkpoint_hours = 23;
end
cfg = struct('enable', true, 'max_seconds', checkpoint_hours * 3600, ...
    'file', fullfile(checkpoint_dir, [model_base '_checkpoint.mat']), ...
    'model_file', model_file, 'submit_script', submit_script);
end

function [checkpoint, loaded] = spsa_gpu_load_checkpoint(opts)
checkpoint = struct();
cfg = get_opt(opts, 'arc_checkpoint', struct());
if ~logical(get_opt(cfg, 'enable', false))
    loaded = false;
    return;
end
checkpoint_file = char(get_opt(cfg, 'file', ''));
if isempty(checkpoint_file) || exist(checkpoint_file, 'file') ~= 2
    loaded = false;
    return;
end
S = load(checkpoint_file, 'checkpoint');
if ~isfield(S, 'checkpoint')
    error('spsa_gpu:checkpointInvalid', 'Checkpoint file exists but does not contain a checkpoint struct.');
end
checkpoint = S.checkpoint;
loaded = true;
end

function tf = spsa_gpu_checkpoint_due(opts, timer_id, ep)
cfg = get_opt(opts, 'arc_checkpoint', struct());
if ~logical(get_opt(cfg, 'enable', false))
    tf = false;
    return;
end
tf = ep < opts.epochs && toc(timer_id) >= double(get_opt(cfg, 'max_seconds', 23 * 3600));
end

function checkpoint = make_spsa_checkpoint(kind, domain, ep, P, best, hist, opts)
checkpoint = struct('kind', char(kind), 'domain', char(domain), 'backend', 'spsa_gpu', ...
    'epoch', double(ep), 'model', P, 'best', best, 'history', hist, ...
    'options', opts, 'rng_state', rng, 'created_at', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')), ...
    'needs_resubmit', true, 'complete', false);
end

function spsa_gpu_save_checkpoint(opts, checkpoint)
cfg = get_opt(opts, 'arc_checkpoint', struct());
checkpoint_file = char(get_opt(cfg, 'file', ''));
if isempty(checkpoint_file)
    error('spsa_gpu:checkpointPath', 'opts.arc_checkpoint.file must be set.');
end
checkpoint_dir = fileparts(checkpoint_file);
if exist(checkpoint_dir, 'dir') ~= 7
    mkdir(checkpoint_dir);
end
save(checkpoint_file, 'checkpoint', '-v7.3');
fprintf('[spsa_gpu checkpoint] saved epoch %d to %s\n', checkpoint.epoch, checkpoint_file);
end

function spsa_gpu_clear_completed_checkpoint(result)
if ~isfield(result, 'checkpoint') || ~isfield(result.checkpoint, 'file')
    return;
end
if isfield(result.checkpoint, 'needs_resubmit') && result.checkpoint.needs_resubmit
    return;
end
checkpoint_file = char(result.checkpoint.file);
if ~isempty(checkpoint_file) && exist(checkpoint_file, 'file') == 2
    delete(checkpoint_file);
    fprintf('[spsa_gpu checkpoint] removed completed checkpoint %s\n', checkpoint_file);
end
end

function info = spsa_gpu_checkpoint_public_info(opts, checkpoint, needs_resubmit)
cfg = get_opt(opts, 'arc_checkpoint', struct());
info = struct('enabled', logical(get_opt(cfg, 'enable', false)), ...
    'file', char(get_opt(cfg, 'file', '')), 'needs_resubmit', logical(needs_resubmit), ...
    'submit_script', char(get_opt(cfg, 'submit_script', '')), 'array_id', getenv('SLURM_ARRAY_TASK_ID'));
if isstruct(checkpoint) && isfield(checkpoint, 'epoch')
    info.epoch = checkpoint.epoch;
else
    info.epoch = NaN;
end
info.complete = info.enabled && ~info.needs_resubmit;
end

function validate_spsa_checkpoint(checkpoint, kind, domain)
if ~strcmp(char(checkpoint.kind), char(kind)) || ~strcmp(char(checkpoint.domain), char(domain))
    error('spsa_gpu:checkpointMismatch', 'Checkpoint kind/domain does not match requested task.');
end
required = {'model', 'best', 'history', 'options', 'rng_state', 'epoch'};
for ii = 1:numel(required)
    if ~isfield(checkpoint, required{ii})
        error('spsa_gpu:checkpointInvalid', 'Checkpoint is missing "%s".', required{ii});
    end
end
end

function lr = spsa_gpu_learning_rate(ep, opts)
schedule_epochs = spsa_gpu_schedule_epochs(opts);
lr = single(get_lr(min(ep, schedule_epochs), schedule_epochs, opts.SCHED));
end

function c = spsa_gpu_perturbation_size(ep, opts)
schedule_epochs = spsa_gpu_schedule_epochs(opts);
c = spsa_gpu_c(min(ep, schedule_epochs), schedule_epochs, opts.spsa);
end

function schedule_epochs = spsa_gpu_schedule_epochs(opts)
continuation = get_opt(opts, 'spsa_continuation', struct());
schedule_epochs = double(get_opt(continuation, 'source_epochs', opts.epochs));
schedule_epochs = max(1, schedule_epochs);
end

function c = spsa_gpu_c(ep, epochs, spsa)
if epochs <= 1
    c = single(spsa.c_end);
    return;
end
t = single((double(ep) - 1) ./ max(1, double(epochs) - 1));
c = single(spsa.c_start .* (spsa.c_end ./ spsa.c_start) .^ t);
end

function g = spsa_gpu_clip_gradient(g, clip_norm)
if isfinite(clip_norm)
    gnorm = single(norm(double(g)));
    if gnorm > clip_norm && gnorm > 0
        g = g .* (clip_norm ./ gnorm);
    end
end
end

function name = spsa_gpu_mex_name(domain)
domain = lower(string(domain));
if domain == "classification"
    name = 'spsa_classify_time_loop_gpu_mex';
elseif domain == "regression"
    name = 'spsa_regress_time_loop_gpu_mex';
elseif any(domain == ["dynamics", "dynamical_systems"])
    name = 'spsa_dynamics_time_loop_gpu_mex';
else
    error('spsa_gpu:mexName', 'Unknown domain "%s".', domain);
end
end

function require_spsa_mex(mex_name)
if exist(mex_name, 'file') ~= 3
    error('spsa_gpu:mexMissing', ...
        'SPSA GPU MEX "%s" is missing. Run spsa_gpu/build/compile_spsa_gpu_mex.m first.', mex_name);
end
end

function spsa_gpu_clear_static(domain)
mex_name = spsa_gpu_mex_name(domain);
if exist(mex_name, 'file') == 3
    feval(mex_name, 'clear');
end
end

function spsa_gpu_clear_dynamics()
mex_name = spsa_gpu_mex_name("dynamical_systems");
if exist(mex_name, 'file') == 3
    feval(mex_name, 'clear');
end
end

function meta = spsa_gpu_mex_metadata(domain)
meta = struct('backend', 'spsa_gpu', 'domain', char(domain), ...
    'mex_name', spsa_gpu_mex_name(domain), 'mex_file', which(spsa_gpu_mex_name(domain)), ...
    'matlab_version', version, 'platform', computer);
try
    g = gpuDevice;
    meta.gpu_name = g.Name;
    meta.gpu_compute_capability = g.ComputeCapability;
catch
    meta.gpu_name = '';
    meta.gpu_compute_capability = '';
end
end

function print_static_progress(domain, hist, ep, opts, best)
if ep == 1 || should_validate(ep, opts) || ep == opts.epochs
    if domain == "classification"
        metric_label = 'acc'; metric_unit = '%';
    else
        metric_label = 'r'; metric_unit = '';
    end
    if isnan(hist.val_loss(ep))
        fprintf('[spsa_gpu] epoch %d | train loss %.6g | train %s %.6g%s | c %.4g | grad %.4g\n', ...
            ep, hist.train_loss(ep), metric_label, hist.train_metric(ep), metric_unit, hist.spsa_c(ep), hist.spsa_grad_norm(ep));
    else
        fprintf('[spsa_gpu] epoch %d | train loss %.6g | val loss %.6g | val %s %.6g%s | best %d\n', ...
            ep, hist.train_loss(ep), hist.val_loss(ep), metric_label, hist.val_metric(ep), metric_unit, best.epoch);
    end
end
end

function print_dynamics_progress(hist, closed_hist, spsa_hist, ep, opts, best)
if ep == 1 || should_closed_loop_validate(ep, opts) || ep == opts.epochs
    wd = closed_hist.wd(ep);
    if isnan(wd)
        fprintf('[spsa_gpu] epoch %d | train loss %.6g | c %.4g | grad %.4g\n', ...
            ep, hist(ep), spsa_hist.c(ep), spsa_hist.grad_norm(ep));
    else
        fprintf('[spsa_gpu] epoch %d | train loss %.6g | closed-loop wd %.6g | best %d\n', ...
            ep, hist(ep), wd, best.epoch);
    end
end
end
