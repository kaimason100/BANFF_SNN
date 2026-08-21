function result = run_arc_static_neuron_sweep_gpu(task_name, n_hidden)
%RUN_ARC_STATIC_NEURON_SWEEP_GPU Train one Yacht or BC neuron-sweep item.
%   The task and neuron count are selected by the dedicated ARC dispatcher.
%   All architecture, data-split and optimisation settings match the local
%   low-rank signed-uniform neuron-sweep Live Scripts.

if nargin ~= 2
    error('run_arc_static_neuron_sweep_gpu:arguments', ...
        'Provide task_name and n_hidden from the ARC sweep dispatcher.');
end
n_hidden = double(n_hidden);
valid_counts = [1000 2000 4000 8000 16000];
if ~isscalar(n_hidden) || ~ismember(n_hidden, valid_counts)
    error('run_arc_static_neuron_sweep_gpu:neuronCount', ...
        'n_hidden must be one of [%s].', num2str(valid_counts));
end
cfg = task_config(task_name);

script_dir = fileparts(mfilename('fullpath'));
repo_root = arc_resolve_repo_root(script_dir);
if exist(fullfile(repo_root, 'arc_ucalgary', 'common'), 'dir') == 7
    arc_common = fullfile(repo_root, 'arc_ucalgary', 'common');
else
    arc_common = fullfile(repo_root, 'common');
end
addpath(arc_common, '-end');
addpath(fullfile(repo_root, 'shared', 'matlab'), '-begin');
add_project_paths(repo_root);
set(0, 'DefaultFigureVisible', 'off');
run_name = sprintf('run_arc_%s_neuron_sweep_N%05d_gpu', cfg.model_stem, n_hidden);
arc_start_diary(repo_root, run_name);
arc_print_environment(run_name);

%% ---------------- Task ----------------
opts = struct();
opts.task_tag = cfg.task_tag;
opts.dataset_file = cfg.dataset_file;

%% ---------------- Network Size and Randomness ----------------
% The sweep follows the Lorenz convention: seed 1 and the five smaller
% neuron counts. The ordinary task training supplies the 32k model.
opts.N_hidden = n_hidden;
opts.N_rec = 10;
opts.seed = 1;
opts.init_seed = opts.seed;
opts.split_seed = 42;

%% ---------------- Weight Scaling and Connectivity ----------------
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

%% ---------------- Stimulus Time Window ----------------
opts.dt = single(1e-3);
opts.PRESENT = struct('T', single(0.300), 'avg_frac', single(0.5));
opts.steps_present = max(1, round(opts.PRESENT.T / opts.dt));
opts.steps_avg = max(1, round(opts.PRESENT.avg_frac * opts.steps_present));
opts.k_avg_start = opts.steps_present - opts.steps_avg + 1;

%% ---------------- Neuron and Synapse Parameters ----------------
opts.neuron = struct('tau_u', single(50e-3), 'tau_w', single(500e-3), ...
    'tau_s_rise', single(2e-3), 'tau_s_decay', single(50e-3), ...
    'E_L', single(-70), 'V_th', single(-50), 'V_reset', single(-65), ...
    'a_param', single(0), 'b_param', single(0.5), ...
    'phi_u', single(1), 'delta_u', single(0.8));

%% ---------------- Optimisation ----------------
opts.epochs = arc_epoch_override(cfg.epoch_env, cfg.default_epochs);
opts.batch_size = 32;
opts.validate_every = 5;
opts.SCHED = struct('type', 'exponential', 'lr_start', single(5e-2), 'lr_end', single(1e-3));
opts.adam = struct('b1', single(0.9), 'b2', single(0.999), 'eps', single(1e-8));

%% ---------------- ARC Progress and Save Location ----------------
opts.live_plot = struct('enable', false, 'every', 100);
opts.arc_progress = struct('enable', true, 'every', 100);
local_live_plot = opts.live_plot;
local_live_plot.enable = true;

model_dir = fullfile(repo_root, 'outputs', 'models');
if exist(model_dir, 'dir') ~= 7, mkdir(model_dir); end
model_file = fullfile(model_dir, sprintf('%s_gpu_N%05d_primary_seed%03d.mat', ...
    cfg.model_stem, opts.N_hidden, opts.seed));
opts = arc_configure_checkpoint(opts, model_file);
opts.arc_checkpoint.submit_script = static_sweep_submit_script(repo_root);

arc_compile_required_mex(repo_root, cfg.domain);
arc_print_training_start(run_name, opts, model_file);

%% ---------------- Run Training ----------------
result = snn_primary_api('train_static', cfg.domain, 'gpu', opts);
result.model_file = model_file;
result = restore_saved_options(result, local_live_plot);
if arc_resubmit_if_needed(result, repo_root)
    return;
end
arc_clear_checkpoint_after_finish(result);
save(model_file, 'result', '-v7.3');
fprintf('Saved %s neuron-sweep model (N_hidden=%d) to:\n%s\n', ...
    cfg.label, opts.N_hidden, model_file);
disp(result.test);
arc_print_result_summary(result);
end

function cfg = task_config(task_name)
switch lower(char(task_name))
    case {'yacht', 'regression_yacht'}
        cfg = struct('task_tag', 'yacht', 'dataset_file', 'yacht_dataset.mat', ...
            'domain', 'regression', 'model_stem', 'regression_yacht', ...
            'label', 'Yacht Hydrodynamics', 'default_epochs', 100000, ...
            'epoch_env', 'SNN_YACHT_SWEEP_EPOCHS');
    case {'bc', 'classification_bc'}
        cfg = struct('task_tag', 'bc', 'dataset_file', 'breast_cancer_dataset.mat', ...
            'domain', 'classification', 'model_stem', 'classification_BC', ...
            'label', 'Breast cancer', 'default_epochs', 5000, ...
            'epoch_env', 'SNN_BC_SWEEP_EPOCHS');
    otherwise
        error('run_arc_static_neuron_sweep_gpu:task', ...
            'task_name must be ''yacht'' or ''bc'', got "%s".', char(task_name));
end
end

function submit_script = static_sweep_submit_script(repo_root)
submit_script = fullfile(repo_root, 'arc_ucalgary', 'submit_arc_static_neuron_sweeps_array.slurm');
if exist(submit_script, 'file') ~= 2
    submit_script = fullfile(repo_root, 'submit_arc_static_neuron_sweeps_array.slurm');
end
end

function result = restore_saved_options(result, local_live_plot)
if isfield(result, 'options')
    result.options.live_plot = local_live_plot;
    if isfield(result.options, 'arc_progress')
        result.options = rmfield(result.options, 'arc_progress');
    end
    if isfield(result.options, 'arc_checkpoint')
        result.options = rmfield(result.options, 'arc_checkpoint');
    end
end
end

function value = arc_epoch_override(name, default_value)
raw = strtrim(getenv(name));
if isempty(raw)
    value = default_value;
    return;
end
value = str2double(raw);
if ~(isscalar(value) && isfinite(value) && value >= 1 && value == floor(value))
    error('run_arc_static_neuron_sweep_gpu:epochOverride', ...
        '%s must be a positive integer, got "%s".', name, raw);
end
end
