% load_static_data.m
function data = load_static_data(domain, opts)
rng(get_opt(opts, 'split_seed', get_opt(opts, 'seed', 42)), 'twister');
if isfield(opts, 'synthetic') && opts.synthetic
    data = synthetic_static_data(domain, opts);
    return;
end

switch domain
    case "classification"
        task_tag = get_opt(opts, 'task_tag', 'bc');
        dataset_file = get_opt(opts, 'dataset_file', '');
        if any(strcmpi(char(task_tag), {'mnist','afro_mnist_vai','afro-mnist-vai','vai'}))
            data = load_image_classification_data(task_tag, dataset_file, opts);
        elseif exist('load_snn_classification_task', 'file') == 2
            TASK = load_snn_classification_task(task_tag, 'DatasetFile', dataset_file);
            data = struct();
            data.X_train = single(TASK.X_train.');
            data.Y_train = single(TASK.Y_train.');
            data.X_val = single(TASK.X_val.');
            data.Y_val = single(TASK.Y_val.');
            data.X_test = single(TASK.X_test.');
            data.Y_test = single(TASK.Y_test.');
            data.idx_train = uint32(TASK.idx_train(:));
            data.idx_val = uint32(TASK.idx_val(:));
            data.idx_test = uint32(TASK.idx_test(:));
            data.mu_X = single(TASK.mu_X);
            data.sigma_X = single(TASK.sigma_X);
            data.mu_y = single(0);
            data.sigma_y = single(1);
            data.task = TASK;
        else
            data = load_basic_classification_data(task_tag, opts);
        end
    case "regression"
        [file_name, feature_cols, target_col] = resolve_regression_dataset(opts);
        M = load_first_matrix(file_name);
        if size(M,2) < max([feature_cols(:); target_col])
            error('Regression dataset "%s" does not contain the requested columns.', file_name);
        end
        used_cols = [feature_cols(:).' target_col];
        if any(~isfinite(M(:,used_cols)), 'all')
            policy = get_nonfinite_policy(opts);
            if policy == "omit_rows"
                keep = ~any(~isfinite(M(:,used_cols)), 2);
                M = M(keep,:);
                if isempty(M)
                    error('snn_primary_api:emptyRegressionData', ...
                        'All rows in "%s" contain non-finite values in selected regression columns.', file_name);
                end
            else
                error('snn_primary_api:nanRegressionData', ...
                    ['Regression dataset "%s" contains non-finite values in selected columns. ', ...
                     'Clean the dataset or set opts.nonfinite_policy = ''omit_rows'' explicitly. ', ...
                     'The legacy opts.nan_policy name is still accepted.'], file_name);
            end
        end
        X_raw = single(M(:,feature_cols));
        Y_raw = single(M(:,target_col));
        data = split_and_standardize(X_raw, Y_raw, opts);
        data.dataset_file = char(file_name);
        data.dataset_sha256 = file_sha256(file_name);
    otherwise
        error('snn_primary_api:domain', 'Static domain must be classification or regression.');
end
data = sanitize_static_data(data, domain, opts);
end
