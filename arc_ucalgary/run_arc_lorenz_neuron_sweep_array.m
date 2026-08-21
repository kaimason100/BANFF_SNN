function run_arc_lorenz_neuron_sweep_array()
%RUN_ARC_LORENZ_NEURON_SWEEP_ARRAY Dispatch one Lorenz neuron-sweep ARC task.
%   SLURM_ARRAY_TASK_ID selects N_hidden from [1k 2k 4k 8k 16k].

arc_root = fileparts(mfilename('fullpath'));
addpath(fullfile(arc_root, 'common'), '-end');
addpath(fullfile(arc_root, 'train'), '-end');
repo_root = arc_resolve_repo_root(arc_root);
addpath(fullfile(repo_root, 'shared', 'matlab'), '-begin');
add_project_paths(repo_root);

fprintf('[ARC Lorenz neuron sweep] selected array task %s%s', ...
    getenv('SLURM_ARRAY_TASK_ID'), newline);
if exist('run_arc_dynamics_lorenz_neuron_sweep_gpu', 'file') ~= 2
    error('run_arc_lorenz_neuron_sweep_array:missingTask', ...
        'run_arc_dynamics_lorenz_neuron_sweep_gpu is not on the MATLAB path.');
end
run_arc_dynamics_lorenz_neuron_sweep_gpu();
end
