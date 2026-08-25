% Package orientation: Shared MATLAB utility used across tasks. Read this with the caller open so input/output structs and saved-result fields are clear.

function added = add_project_paths(root)
%ADD_PROJECT_PATHS Add the compact BANFF code folders to the MATLAB path.

if nargin < 1 || isempty(root)
    root = project_root();
end

paths = {root; fullfile(root, 'figures', 'matlab'); ...
    fullfile(root, 'figures', 'scripts'); ...
    fullfile(root, 'figures', 'third_party'); ...
    fullfile(root, 'data', 'raw')};

added = {};
for i = 1:numel(paths)
    if exist(paths{i}, 'dir') == 7
        addpath(paths{i});
        added{end+1,1} = paths{i}; %#ok<AGROW>
    end
end
end
