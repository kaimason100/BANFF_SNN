function result = run_arc_classification_afromnist_vai_gpu()
%RUN_ARC_CLASSIFICATION_AfroMNIST_Vai_GPU ARC GPU wrapper generated from Classification/train/GPU_implementation_classification_AfroMNIST_Vai.mlx.
% This wrapper preserves the local training options, disables live plots,
% enables text progress, compiles the required GPU MEX if needed, and saves
% the same result structure/model filename as the local GPU script.

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
arc_start_diary(repo_root, 'run_arc_classification_afromnist_vai_gpu');
arc_print_environment('run_arc_classification_afromnist_vai_gpu');

% GPU_implementation_classification_AfroMNIST_Vai.mlx
% Train the AfroMNIST_Vai primary-decoder classification task and save the trained network.
% The saved result contains options, training history, best bias vector, and test metrics.


%% ---------------- Task ----------------
% task_tag selects the dataset/task loader branch used by snn_primary_api.
% dataset_file is optional for some built-in tasks, but setting it makes the
% script explicit and easier to reproduce.
opts = struct();
opts.task_tag = 'afro_mnist_vai';
opts.dataset_file = 'afro_mnist_vai.mat';

%% ---------------- Network Size and Randomness ----------------
% N_hidden is the number of spiking neurons. GPU scripts use the ARC publication-scale
% default of 32k neurons; CPU scripts use a smaller value so local checks remain plausible.
% N_rec is the dimensionality of the recurrent decoder basis used internally.
% seed fixes deterministic weight generation and data splitting.
opts.N_hidden = 32000;
opts.N_rec = 10;
opts.seed = 1;
opts = arc_apply_network_seed(opts);
opts.split_seed = 42;

%% ---------------- Weight Scaling and Connectivity ----------------
% SCALE.enc controls input weight scale; SCALE.rec controls recurrent feedback
% strength; SCALE.dec controls the primary decoder scale.
% NET.p_rec is recurrent connection probability. variance_correction keeps the
% effective recurrent variance stable when p_rec changes.
% dale.enable applies excitatory/inhibitory signs to outgoing decoder rows.
opts.SCALE = struct('enc', single(2), 'rec', single(0.05), 'dec', single(0.1));
opts.NET = struct('p_rec', 1, 'variance_correction', true, ...
    'dale', struct('enable', true, 'p_exc', 0.5, 'sign', []));
opts = arc_apply_architecture_env(opts);

%% ---------------- Stimulus Time Window ----------------
% dt is the simulation timestep in seconds.
% PRESENT.T is stimulus duration per sample.
% PRESENT.avg_frac is the final fraction of timesteps averaged to form output.
% steps_present, steps_avg, and k_avg_start are derived values used by the MEX/API.
opts.dt = single(1e-3);
opts.PRESENT = struct('T', single(0.300), 'avg_frac', single(0.5));
opts.steps_present = max(1, round(opts.PRESENT.T / opts.dt));
opts.steps_avg = max(1, round(opts.PRESENT.avg_frac * opts.steps_present));
opts.k_avg_start = opts.steps_present - opts.steps_avg + 1;

%% ---------------- Neuron and Synapse Parameters ----------------
% tau_u and tau_w are membrane and adaptation time constants in seconds.
% tau_s_rise and tau_s_decay define the two-stage synaptic filter; require rise < decay.
% E_L, V_th, and V_reset are voltage parameters in mV.
% a_param and b_param control adaptation; phi_u and delta_u control surrogate-gradient width.
opts.neuron = struct('tau_u', single(50e-3), 'tau_w', single(500e-3), ...
    'tau_s_rise', single(2e-3), 'tau_s_decay', single(50e-3), ...
    'E_L', single(-70), 'V_th', single(-50), 'V_reset', single(-65), ...
    'a_param', single(0), 'b_param', single(0.5), ...
    'phi_u', single(1), 'delta_u', single(0.8));

%% ---------------- Optimisation ----------------
% Only the bias vector B is trained. W_out is fixed and is the primary decoder.
% epochs is the number of full passes over the training split.
% batch_size controls GPU mini-batches and CPU/check averaging where applicable.
% validate_every controls how often validation metrics are computed and best.B is updated.
% SCHED.type can be 'exponential' or 'cosine'. lr_start/lr_end are bias-learning rates.
% adam contains Adam/AMSGrad-style bias optimizer parameters.
opts.epochs = 2500;
opts.batch_size = 32;
opts.validate_every = 5;
opts.SCHED = struct('type', 'exponential', 'lr_start', single(5e-2), 'lr_end', single(1e-3));
opts.adam = struct('b1', single(0.9), 'b2', single(0.999), 'eps', single(1e-8));
%% ---------------- Live Training Plots ----------------
% The training loop refreshes an inline Live Editor figure instead of saving plots to a folder.
% Set enable=false to disable plotting, or increase every to refresh less often.
opts.live_plot = struct('enable', false, 'every', 100);
opts.arc_progress = struct('enable', true, 'every', opts.validate_every);
arc_local_live_plot = opts.live_plot;
arc_local_live_plot.enable = true;


%% ---------------- Save Location ----------------
% TEST scripts load this exact file. Change model_file here and in the matching
% TEST script if you want to keep multiple trained networks for the same task.
model_dir = fullfile(repo_root, 'outputs', 'models');
if exist(model_dir, 'dir') ~= 7, mkdir(model_dir); end
model_file = fullfile(model_dir, sprintf('classification_AfroMNIST_Vai_gpu_primary_seed%03d.mat', opts.seed));
opts = arc_configure_checkpoint(opts, model_file);

arc_compile_required_mex(repo_root, 'classification');
arc_print_training_start('run_arc_classification_afromnist_vai_gpu', opts, model_file);

%% ---------------- Run Training ----------------
% result.history stores train/validation curves.
% result.best stores the best validation loss/metric and the corresponding bias vector.
% result.test reports held-out performance after restoring result.best.B.
% For this task, result.test.metric is classification accuracy in percent.
result = snn_primary_api('train_static', 'classification', 'gpu', opts);
result.model_file = model_file;
result = arc_restore_saved_options(result, arc_local_live_plot);
if arc_resubmit_if_needed(result, repo_root)
    return;
end
arc_clear_checkpoint_after_finish(result);
save(model_file, 'result', '-v7.3');
fprintf('Saved trained network to:\n%s\n', model_file);
disp(result.test);

arc_print_result_summary(result);


function result = arc_restore_saved_options(result, local_live_plot)
% Keep saved ARC model options aligned with the matching local Live Editor script.
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

end
