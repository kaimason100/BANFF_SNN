% load_basic_classification_data.m
function data = load_basic_classification_data(task_tag, opts)
% Fallback loader used if the curated classification helper is not on path.
% Column layouts are kept identical to load_snn_classification_task.
task_tag = lower(string(task_tag));
switch task_tag
    case {"bc","breast_cancer","breast-cancer","iris_bc"}
        M = load_first_matrix(resolve_dataset_path(get_opt(opts, 'dataset_file', ''), ...
            'breast_cancer_dataset.mat'));
        y_raw = M(:,2);
        X_raw = single(M(:,3:end));
    otherwise
        error('snn_primary_api:classificationTask', 'Unknown classification task "%s".', task_tag);
end

assert_finite_classification_labels(y_raw, 'classification labels');
if iscategorical(y_raw) && any(isundefined(y_raw(:)))
    error('snn_primary_api:nonfiniteClassificationLabels', ...
        'classification labels contain undefined categorical values.');
end
[~,~,y_idx] = unique(string(categorical(y_raw)));
C = max(y_idx);
Y = zeros(numel(y_idx), C, 'single');
for i = 1:numel(y_idx)
    Y(i,y_idx(i)) = 1;
end
data = split_and_standardize(X_raw, single(Y), opts);
end
