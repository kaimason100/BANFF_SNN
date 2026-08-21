% Package orientation: Shared MATLAB utility used across tasks. Read this with the caller open so input/output structs and saved-result fields are clear.

function added = add_project_paths(root)
%ADD_PROJECT_PATHS Add curated package folders to the MATLAB path.
%   ADDED = ADD_PROJECT_PATHS(ROOT) adds the main script, source, data, and
%   binary folders used by the reorganized package.

if nargin < 1 || isempty(root)
    root = project_root();
end

paths = {
    fullfile(root, 'Classification', 'train')
    fullfile(root, 'Classification', 'test')
    fullfile(root, 'Classification', 'training')
    fullfile(root, 'Classification', 'training', 'helpers')
    fullfile(root, 'Classification', 'analysis')
    fullfile(root, 'Classification', 'plotting')
    fullfile(root, 'Classification', 'build')
    fullfile(root, 'Classification', 'src', 'cuda')
    fullfile(root, 'Regression', 'train')
    fullfile(root, 'Regression', 'test')
    fullfile(root, 'Regression', 'training')
    fullfile(root, 'Regression', 'analysis')
    fullfile(root, 'Regression', 'plotting')
    fullfile(root, 'Regression', 'build')
    fullfile(root, 'Regression', 'src', 'cuda')
    fullfile(root, 'dynamical_systems', 'train')
    fullfile(root, 'dynamical_systems', 'test')
    fullfile(root, 'dynamical_systems', 'training')
    fullfile(root, 'dynamical_systems', 'training', 'legacy')
    fullfile(root, 'dynamical_systems', 'training', 'variants')
    fullfile(root, 'dynamical_systems', 'analysis')
    fullfile(root, 'dynamical_systems', 'plotting')
    fullfile(root, 'dynamical_systems', 'utilities')
    fullfile(root, 'dynamical_systems', 'utilities', 'network_generation')
    fullfile(root, 'dynamical_systems', 'utilities', 'network_generation', 'all_ds_tasks_32k')
    fullfile(root, 'dynamical_systems', 'utilities', 'network_generation', 'representational_drift')
    fullfile(root, 'dynamical_systems', 'build')
    fullfile(root, 'dynamical_systems', 'src', 'cuda')
    fullfile(root, 'shared', 'plotting')
    fullfile(root, 'shared', 'utilities')
    fullfile(root, 'shared', 'matlab')
    fullfile(root, 'shared', 'matlab', 'snn_primary_api_functions')
    fullfile(root, 'data')
    fullfile(root, 'data', 'external')
    fullfile(root, 'data', 'raw')
    fullfile(root, 'data', 'processed')
    fullfile(root, 'data', 'derived')
    fullfile(root, 'outputs')
    fullfile(root, 'outputs', 'repro')
    fullfile(root, 'outputs', 'fc')
    fullfile(root, 'outputs', 'batch')
    };

added = {};
for i = 1:numel(paths)
    if exist(paths{i}, 'dir') == 7
        addpath(paths{i});
        added{end+1,1} = paths{i}; %#ok<AGROW>
    end
end

bin_roots = {
    fullfile(root, 'Classification', 'bin')
    fullfile(root, 'Regression', 'bin')
    fullfile(root, 'dynamical_systems', 'bin')
    };

for i = 1:numel(bin_roots)
    if exist(bin_roots{i}, 'dir') == 7
        addpath(genpath(bin_roots{i}));
        added{end+1,1} = bin_roots{i}; %#ok<AGROW>
    end
end
end
