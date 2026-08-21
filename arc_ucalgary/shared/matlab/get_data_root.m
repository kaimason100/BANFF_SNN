function data_root = get_data_root()
%GET_DATA_ROOT Return the package data root.

data_root = fullfile(project_root(), 'data');
end
