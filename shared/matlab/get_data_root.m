% Package orientation: Shared MATLAB utility used across tasks. Read this with the caller open so input/output structs and saved-result fields are clear.

function data_root = get_data_root()
%GET_DATA_ROOT Return the package data root.

data_root = fullfile(project_root(), 'data');
end
