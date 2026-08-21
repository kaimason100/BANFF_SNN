function run_arc_training_array()
%RUN_ARC_TRAINING_ARRAY Dispatch one ARC GPU training task from a SLURM array.
%   SLURM_ARRAY_TASK_ID selects the task. If it is missing, task 1 is used so
%   this function can also be run interactively for a smoke test.

repo_root = fileparts(mfilename('fullpath'));
% ARC is a self-contained bundle. Place its root and shared implementation
% ahead of any parent local repository that may already be on the path.
addpath(repo_root, '-begin');
addpath(fullfile(repo_root, 'shared', 'matlab'), '-begin');
addpath(fullfile(repo_root, 'common'), '-begin');
addpath(fullfile(repo_root, 'train'), '-begin');
add_project_paths(repo_root);

tasks = { ...
    'run_arc_classification_bc_gpu'
    'run_arc_classification_mnist_gpu'
    'run_arc_classification_afromnist_vai_gpu'
    'run_arc_regression_abalone_gpu'
    'run_arc_regression_toyota_gpu'
    'run_arc_regression_yacht_gpu'
    'run_arc_dynamics_lorenz_gpu'
    'run_arc_dynamics_sprotts_gpu'
    'run_arc_dynamics_vanderpol_gpu'
    };

seeds = [1 2 3];
array_id = str2double(getenv('SLURM_ARRAY_TASK_ID'));
if isnan(array_id) || array_id < 1 || array_id > numel(tasks) * numel(seeds)
    array_id = 1;
end
task_id = ceil(array_id / numel(seeds));
seed_index = mod(array_id - 1, numel(seeds)) + 1;
selected_seed = seeds(seed_index);
setenv('ARC_NETWORK_SEED', num2str(selected_seed));
setenv('ARC_BASE_TASK_ID', num2str(task_id));
fprintf('[ARC array] selected array item %d/%d: task %d/%d %s | seed %d%s', ...
    array_id, numel(tasks) * numel(seeds), task_id, numel(tasks), tasks{task_id}, selected_seed, newline);
if exist(tasks{task_id}, 'file') ~= 2
    error('run_arc_training_array:missingTask', ...
         ['Selected ARC task "%s" is not on the MATLAB path. ', ...
         'Resolved ARC bundle root was "%s". Confirm train/ exists in the standalone bundle.'], ...
        tasks{task_id}, repo_root);
end
feval(tasks{task_id});
end
