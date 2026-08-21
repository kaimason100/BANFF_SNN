function run_arc_static_neuron_sweeps_array()
%RUN_ARC_STATIC_NEURON_SWEEPS_ARRAY Dispatch Yacht and BC neuron sweeps.
%   Array 1:5  -> Yacht [1k 2k 4k 8k 16k]
%   Array 6:10 -> BC    [1k 2k 4k 8k 16k]

arc_root = fileparts(mfilename('fullpath'));
addpath(fullfile(arc_root, 'common'), '-end');
addpath(fullfile(arc_root, 'train'), '-end');
repo_root = arc_resolve_repo_root(arc_root);
addpath(fullfile(repo_root, 'shared', 'matlab'), '-begin');
add_project_paths(repo_root);

neuron_counts = [1000 2000 4000 8000 16000];
array_id = str2double(getenv('SLURM_ARRAY_TASK_ID'));
if ~isscalar(array_id) || ~isfinite(array_id) || array_id ~= floor(array_id) || ...
        array_id < 1 || array_id > 2 * numel(neuron_counts)
    error('run_arc_static_neuron_sweeps_array:arrayId', ...
        'SLURM_ARRAY_TASK_ID must be an integer from 1 to %d.', 2 * numel(neuron_counts));
end

if array_id <= numel(neuron_counts)
    task_name = 'yacht';
    count_index = array_id;
else
    task_name = 'bc';
    count_index = array_id - numel(neuron_counts);
end
n_hidden = neuron_counts(count_index);

fprintf('[ARC static neuron sweep] array item %d/10: task=%s, N_hidden=%d%s', ...
    array_id, task_name, n_hidden, newline);
run_arc_static_neuron_sweep_gpu(task_name, n_hidden);
end
