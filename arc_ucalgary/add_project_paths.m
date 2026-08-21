function added = add_project_paths(root)
%ADD_PROJECT_PATHS Add standalone ARC project folders to the MATLAB path.
%   ADDED = ADD_PROJECT_PATHS(ROOT) is signature-compatible with
%   shared/matlab/add_project_paths.m. This top-level ARC wrapper exists so
%   MATLAB path resolution cannot fail if the standalone folder shadows the
%   shared helper during startup.

if nargin < 1 || isempty(root)
    root = fileparts(mfilename('fullpath'));
end
root = char(root);

paths = {
    fullfile(root, 'common')
    fullfile(root, 'train')
    fullfile(root, 'shared', 'matlab')
    fullfile(root, 'shared', 'utilities')
    fullfile(root, 'Classification')
    fullfile(root, 'Classification', 'training', 'helpers')
    fullfile(root, 'Classification', 'src')
    fullfile(root, 'Classification', 'src', 'cuda')
    fullfile(root, 'Classification', 'bin', mexext)
    fullfile(root, 'Regression')
    fullfile(root, 'Regression', 'training')
    fullfile(root, 'Regression', 'src')
    fullfile(root, 'Regression', 'src', 'cuda')
    fullfile(root, 'Regression', 'bin', mexext)
    fullfile(root, 'dynamical_systems')
    fullfile(root, 'dynamical_systems', 'training')
    fullfile(root, 'dynamical_systems', 'src')
    fullfile(root, 'dynamical_systems', 'src', 'cuda')
    fullfile(root, 'dynamical_systems', 'utilities')
    fullfile(root, 'dynamical_systems', 'utilities', 'network_generation')
    fullfile(root, 'dynamical_systems', 'bin', mexext)
    fullfile(root, 'data')
    fullfile(root, 'data', 'raw')
    fullfile(root, 'data', 'external')
    fullfile(root, 'outputs')
    };

added = {};
for ii = 1:numel(paths)
    if exist(paths{ii}, 'dir') == 7
        addpath(paths{ii});
        added{end+1,1} = paths{ii}; %#ok<AGROW>
    end
end
end
