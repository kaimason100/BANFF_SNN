function root = setup_project_paths()
%SETUP_PROJECT_PATHS Add the reorganized package folders to the MATLAB path.
%   ROOT = SETUP_PROJECT_PATHS() returns the package root and ensures the
%   training, analysis, plotting, CUDA, binary, shared, data, and output
%   folders are discoverable from anywhere inside MATLAB.

root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'shared', 'matlab'));
added = add_project_paths(root);

if nargout == 0
    fprintf('[setup_project_paths] Added %d package path(s) rooted at:\n%s\n', ...
        numel(added), root);
end
end
