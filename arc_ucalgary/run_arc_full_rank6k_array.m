function run_arc_full_rank6k_array()
%RUN_ARC_FULL_RANK6K_ARRAY Dispatch one full-rank 6k proof ARC GPU task.
%   SLURM_ARRAY_TASK_ID selects breast-cancer classification, Yacht Hydrodynamics
%   regression, or Van der Pol dynamics. Each task uses seed 1 only.

arc_root = fileparts(mfilename('fullpath'));
addpath(fullfile(arc_root, 'common'), '-end');
addpath(fullfile(arc_root, 'train'), '-end');
repo_root = arc_resolve_repo_root(arc_root);
addpath(fullfile(repo_root, 'shared', 'matlab'), '-begin');
add_project_paths(repo_root);

tasks = { ...
    'run_arc_classification_bc_full_rank6k_gpu'
    'run_arc_regression_yacht_full_rank6k_gpu'
    'run_arc_dynamics_vanderpol_full_rank6k_gpu'
    };

array_id = str2double(getenv('SLURM_ARRAY_TASK_ID'));
if isnan(array_id) || array_id < 1 || array_id > numel(tasks)
    array_id = 1;
end
fprintf('[ARC full-rank 6k] selected proof task %d/%d: %s%s', ...
    array_id, numel(tasks), tasks{array_id}, newline);
if exist(tasks{array_id}, 'file') ~= 2
    error('run_arc_full_rank6k_array:missingTask', ...
        'Selected proof task "%s" is not on the MATLAB path.', tasks{array_id});
end
feval(tasks{array_id});
end
