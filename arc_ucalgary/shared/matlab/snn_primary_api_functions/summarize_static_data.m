% summarize_static_data.m
function summary = summarize_static_data(data)
summary = struct();
summary.n_train = size(data.X_train,2);
summary.n_val = size(data.X_val,2);
summary.n_test = size(data.X_test,2);
summary.n_features = size(data.X_train,1);
summary.n_outputs = size(data.Y_train,1);
summary.input_mean_abs_train = single(mean(abs(data.X_train(:))));
summary.input_std_train = single(std(single(data.X_train(:))));
summary.target_mean_train = single(mean(data.Y_train(:)));
summary.target_std_train = single(std(single(data.Y_train(:))));
if isfield(data, 'split_policy')
    summary.split_policy = char(data.split_policy);
else
    summary.split_policy = 'unspecified';
end
if isfield(data, 'exact_duplicate_group_count')
    summary.exact_duplicate_group_count = double(data.exact_duplicate_group_count);
else
    summary.exact_duplicate_group_count = NaN;
end
if isfield(data, 'dataset_file')
    summary.dataset_file = char(data.dataset_file);
else
    summary.dataset_file = '';
end
if isfield(data, 'dataset_sha256')
    summary.dataset_sha256 = char(data.dataset_sha256);
else
    summary.dataset_sha256 = '';
end
end
