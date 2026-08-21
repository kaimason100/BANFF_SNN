% Package orientation: SPSA GPU support. This path is separate from the primary e-prop-style bias-training API and should be reviewed as an alternative optimizer implementation.

function repo_root = spsa_gpu_add_paths(start_dir)
%SPSA_GPU_ADD_PATHS Add only the paths needed by the separate SPSA GPU module.
%   This module is additive: it does not modify the existing backend code or
%   replace the existing GPU MEX binaries.

if nargin < 1 || isempty(start_dir)
    start_dir = fileparts(mfilename('fullpath'));
end
repo_root = spsa_gpu_find_repo_root(start_dir);
module_root = fullfile(repo_root, 'spsa_gpu');

% Put this bundle root first so add_project_paths cannot resolve to a local
% development copy when ARC is launched from a nested checkout.
addpath_if_dir(repo_root, '-begin');
addpath_if_dir(fullfile(module_root, 'matlab'), '-begin');
addpath_if_dir(fullfile(module_root, 'build'), '-begin');
addpath_if_dir(fullfile(module_root, 'bin', mexext), '-begin');
addpath_if_dir(fullfile(repo_root, 'shared', 'matlab'), '-begin');
addpath_if_dir(fullfile(repo_root, 'shared', 'matlab', 'snn_primary_api_functions'), '-end');
addpath_if_dir(fullfile(repo_root, 'Classification', 'training', 'helpers'), '-end');
addpath_if_dir(fullfile(repo_root, 'dynamical_systems', 'utilities'), '-end');
addpath_if_dir(fullfile(repo_root, 'Regression', 'training'), '-end');

if exist(fullfile(repo_root, 'shared', 'matlab', 'add_project_paths.m'), 'file') == 2
    add_project_paths(repo_root);
end
end

function addpath_if_dir(path_name, position)
if exist(path_name, 'dir') == 7
    addpath(path_name, position);
end
end

function repo_root = spsa_gpu_find_repo_root(start_dir)
start_dir = char(start_dir);
candidates = {start_dir};
for depth = 0:10
    up = ascend_dir(start_dir, depth);
    if ~isempty(up)
        candidates{end+1} = up; %#ok<AGROW>
    end
end
candidates{end+1} = pwd; %#ok<AGROW>
candidates = unique(candidates, 'stable');
for ii = 1:numel(candidates)
    c = candidates{ii};
    if exist(fullfile(c, 'spsa_gpu', 'matlab', 'spsa_gpu_run_task.m'), 'file') == 2 && ...
            exist(fullfile(c, 'shared', 'matlab', 'snn_primary_api.m'), 'file') == 2
        repo_root = c;
        return;
    end
end
error('spsa_gpu:addPaths', 'Could not locate repository root from "%s".', start_dir);
end

function out = ascend_dir(in, n)
out = in;
for ii = 1:n
    parent = fileparts(out);
    if isempty(parent) || strcmp(parent, out)
        out = '';
        return;
    end
    out = parent;
end
end
