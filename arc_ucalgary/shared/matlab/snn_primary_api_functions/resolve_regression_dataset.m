% resolve_regression_dataset.m
function [file_name, feature_cols, target_col] = resolve_regression_dataset(opts)
task_tag = lower(string(get_opt(opts, 'task_tag', get_opt(opts, 'dataset', 'yacht'))));
explicit_file = get_opt(opts, 'dataset_file', '');
switch task_tag
    case {"yacht","yacht_dataset"}
        default_file = 'yacht_dataset.mat';
        feature_cols = 1:6;
        target_col = 7;
    case {"abalone","abalone_dataset"}
        default_file = 'abalone_dataset.mat';
        feature_cols = [];
        target_col = [];
    case {"toyota","toyota_dataset"}
        default_file = 'toyota_dataset.mat';
        feature_cols = [];
        target_col = 3;
    otherwise
        default_file = char(task_tag + "_dataset.mat");
        feature_cols = [];
        target_col = [];
end
file_name = resolve_dataset_path(explicit_file, default_file);
M = load_first_matrix(file_name);
ncol = size(M,2);
target_col = get_opt(opts, 'target_col', target_col);
if isempty(target_col)
    target_col = ncol;
end
if isempty(feature_cols)
    feature_cols = setdiff(1:ncol, target_col, 'stable');
end
feature_cols = get_opt(opts, 'feature_cols', feature_cols);
end
