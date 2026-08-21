function result = run_arc_dynamics_lorenz_neuron_sweep_gpu()
%RUN_ARC_DYNAMICS_LORENZ_NEURON_SWEEP_GPU ARC Lorenz low-rank neuron sweep.
% This wrapper preserves the Lorenz ARC training options and changes only
% N_hidden according to SLURM_ARRAY_TASK_ID. The architecture remains the
% publication low-rank signed-uniform implementation and uses seed 1.

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
arc_start_diary(repo_root, 'run_arc_dynamics_lorenz_neuron_sweep_gpu');
arc_print_environment('run_arc_dynamics_lorenz_neuron_sweep_gpu');

% GPU_implementation_dynamical_systems_lorenz_neuron_sweep.mlx
% Train one Lorenz low-rank signed-uniform neuron-count sweep item.

%% ---------------- Task and Simulation ----------------
% system_name chooses the target dynamical system generator.
% seed fixes deterministic network generation and synthetic fallback trajectories.
% dyn_sys_rate is passed to the trajectory simulator when utilities are available.
% T_sim sets each sampled training snippet length; long_sim_time and burn_in_time
% define the long true simulation pool used for random contiguous snippets.
opts = struct();
opts.system_name = 'lorenz';
opts.seed = 1;
opts.init_seed = opts.seed;
opts.split_seed = 42;
opts.dyn_sys_rate = 2;
opts.T_sim = single(20);
opts.long_sim_time = single(2000);
opts.burn_in_time = single(10);
opts.train_blocks = 1;
opts.closed_loop_validate_every = 100;
opts.arc_closed_loop_validation_backend = 'gpu_lightweight';
opts.closed_loop_validation_time = single(50);
opts.closed_loop_validation_ics = 5;
opts.closed_loop_test_time = single(50);
opts.closed_loop_test_warmup_time = single(5);
opts.closed_loop_test_ics = 5;
opts.closed_loop_ic_jitter = single(0.01);
opts.closed_loop_ic_seed = 1001;
opts.closed_loop_test_ic_seed = 123;
opts.dt = single(1e-3);

%% ---------------- Neuron Count Sweep ----------------
% The main Lorenz ARC training script already covers 32k neurons. This sweep
% trains smaller networks with the same seed and low-rank architecture.
lorenz_neuron_counts = [1000 2000 4000 8000 16000];
neuron_index = arc_lorenz_neuron_sweep_index(numel(lorenz_neuron_counts));

%% ---------------- Network Size and Randomness ----------------
% N_hidden is the number of spiking neurons selected by the ARC array index.
% N_rec is the recurrent decoder basis dimensionality used internally.
opts.N_hidden = lorenz_neuron_counts(neuron_index);
opts.N_rec = 10;

%% ---------------- Weight Scaling and Connectivity ----------------
% SCALE.enc controls input encoding, SCALE.rec recurrent feedback strength,
% and SCALE.dec the fixed primary decoder scale.
% NET fields control recurrent sparsity, variance correction, and Dale signs.
opts.SCALE = struct('enc', single(2), 'rec', single(0.05), 'dec', single(0.1));
opts.NET = struct('p_rec', 1, 'variance_correction', true, ...
    'dale', struct('enable', true, 'p_exc', 0.5, 'sign', []));
opts.arch = struct();
opts.arch.recurrent_mode = "low_rank";                  % "low_rank" or "full_rank"
opts.arch.decoder_mode = "signed";                      % "shared" or "signed"
opts.arch.signed_decoder_distribution = "uniform";      % "gaussian"/"normal" or "uniform"
opts.arch.full_rank_p_rec = single(1.0);
opts.arch.full_rank_remove_self_connections = true;
opts.arch.full_rank_storage = "auto";                   % "auto", "dense" or "sparse"
opts.arch.full_rank_sparse_threshold = single(0.10);
opts.arch.max_dense_full_rank_N = int32(6000);
opts.arch.max_sparse_full_rank_nnz = int64(20000000);
opts.arch.max_full_rank_recurrent_bytes = double(2.5 * 2^30);

%% ---------------- Free-Run Schedule ----------------
% use_multistep enables warmup/free-run cycles during training.
% W_warmup is the number of teacher-forced steps per cycle.
% H_free is the number of free-running feedback steps per cycle.
opts.use_multistep = true;
opts.W_warmup = round(0.030 / opts.dt);
opts.H_free = round(0.055/opts.dt);

%% ---------------- Neuron and Synapse Parameters ----------------
% tau_u/tau_w are membrane/adaptation time constants; tau_s_rise/tau_s_decay
% define the two-stage synaptic filter. Voltage terms are in mV.
% phi_u and delta_u define the surrogate-gradient shape around threshold.
opts.neuron = struct('tau_u', single(50e-3), 'tau_w', single(500e-3), ...
    'tau_s_rise', single(2e-3), 'tau_s_decay', single(50e-3), ...
    'E_L', single(-70), 'V_th', single(-50), 'V_reset', single(-65), ...
    'a_param', single(0), 'b_param', single(0.5), ...
    'phi_u', single(1), 'delta_u', single(0.8));

%% ---------------- Optimisation ----------------
% Only the bias vector B is trained. The output decoder W_out is fixed.
% epochs can be large for GPU training.
% SCHED controls bias-learning-rate decay. adam controls bias optimizer moments.
opts.epochs = 2e5;
opts.SCHED = struct('type', 'exponential', 'lr_start', single(5e-2), 'lr_end', single(1e-3));
opts.adam = struct('b1', single(0.9), 'b2', single(0.999), 'eps', single(1e-8));

%% ---------------- Live Training Plots ----------------
% ARC disables plotting but keeps text progress for batch logs.
opts.live_plot = struct('enable', false, 'every', 1000);
opts.arc_progress = struct('enable', true, 'every', 100);
arc_local_live_plot = opts.live_plot;
arc_local_live_plot.enable = true;

%% ---------------- Save Location ----------------
% One model is saved per neuron count. The N value is included in the filename
% so these sweep runs can coexist with the main 32k Lorenz model.
model_dir = fullfile(repo_root, 'outputs', 'models');
if exist(model_dir, 'dir') ~= 7, mkdir(model_dir); end
model_file = fullfile(model_dir, sprintf( ...
    'dynamical_systems_lorenz_gpu_N%05d_primary_seed%03d.mat', ...
    opts.N_hidden, opts.seed));
opts = arc_configure_checkpoint(opts, model_file);
opts.arc_checkpoint.submit_script = arc_lorenz_neuron_sweep_submit_script(repo_root);

arc_compile_required_mex(repo_root, 'dynamical_systems');
arc_print_training_start('run_arc_dynamics_lorenz_neuron_sweep_gpu', opts, model_file);

%% ---------------- Run Training ----------------
% result.history is mean teacher-forced one-step training loss per epoch.
% result.best stores the lowest closed-loop-WD bias vector once validation runs; otherwise it falls back to lowest training loss.
result = snn_primary_api('train_dynamics', 'dynamical_systems', 'gpu', opts);
result.model_file = model_file;
result = arc_restore_saved_options(result, arc_local_live_plot);
if arc_resubmit_if_needed(result, repo_root)
    return;
end
arc_clear_checkpoint_after_finish(result);
save(model_file, 'result', '-v7.3');
fprintf('Saved trained network to:\n%s\n', model_file);
disp(result.best);

arc_print_result_summary(result);


function idx = arc_lorenz_neuron_sweep_index(num_counts)
idx = str2double(getenv('SLURM_ARRAY_TASK_ID'));
if isnan(idx) || idx < 1 || idx > num_counts
    idx = 1;
end
idx = round(idx);
end


function submit_script = arc_lorenz_neuron_sweep_submit_script(repo_root)
submit_script = fullfile(repo_root, 'arc_ucalgary', 'submit_arc_lorenz_neuron_sweep_array.slurm');
if exist(submit_script, 'file') ~= 2
    submit_script = fullfile(repo_root, 'submit_arc_lorenz_neuron_sweep_array.slurm');
end
end


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
