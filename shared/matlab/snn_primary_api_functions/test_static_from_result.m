% test_static_from_result.m
function result = test_static_from_result(domain, backend, train_result, opts)
if ~isfield(train_result, 'options')
    error('snn_primary_api:modelFile', 'Saved training result does not contain options.');
end
if domain == "regression" && is_toyota_task(train_result.options) && ...
        ~has_grouped_regression_split_metadata(train_result.options)
    error('snn_primary_api:legacyToyotaSplit', ...
        ['This Toyota model predates duplicate-group-aware splitting. Retrain it ', ...
         'before testing so identical feature-plus-target rows cannot cross partitions.']);
end
opts_eval = merge_options_with_seed(train_result.options, rmfield_if_present(opts, 'model_file'));
opts_eval = apply_saved_architecture_metadata(opts_eval, train_result);
data = load_static_data(domain, opts_eval);
validate_static_data(data, domain, 'test_static');
assert_static_dataset_provenance(train_result.options, data, domain);
if isfield(train_result, 'model') && isstruct(train_result.model)
    P = train_result.model;
else
    P = make_primary_model(size(data.X_train,1), size(data.Y_train,1), opts_eval);
end
P = ensure_model_architecture_fields(P, opts_eval);
P.B = best_bias_from_result(train_result);
if backend == "gpu"
    [test, ~] = static_eval_gpu(domain, data, P, opts_eval, 'test');
else
    [test, ~] = static_eval_cpu(domain, data.X_test, data.Y_test, P, opts_eval);
end
if domain == "regression"
    test = attach_regression_test_stats(test, data, 'test');
    if ~has_valid_regression_stats(test) && backend == "gpu"
        cpu_test = static_eval_cpu(domain, data.X_test, data.Y_test, P, opts_eval);
        test.Z = cpu_test.Z;
        test.Y = cpu_test.Y;
        test.regression = regression_test_stats_task_units(test.Z, test.Y, data);
        test.regression.source = 'cpu_reference_outputs';
        warning('snn_primary_api:gpuRegressionStatsFallback', ...
            ['GPU regression predictions were not finite. Regression summary ', ...
             'statistics were computed with the CPU reference outputs using the same saved model. ', ...
             'Recompile the GPU MEX to make the GPU prediction path return output arrays.']);
    end
end
result = struct();
result.domain = char(domain);
result.train_backend = get_result_field(train_result, 'backend', 'unknown');
result.test_backend = char(backend);
result.model_file = char(opts.model_file);
result.test = test;
result.options = opts_eval;
result = attach_architecture_metadata(result, P, opts_eval);
end

function tf = is_toyota_task(opts)
task = lower(string(get_opt(opts, 'task_tag', get_opt(opts, 'dataset', ''))));
tf = any(task == ["toyota" "toyota_dataset"]);
end

function tf = has_grouped_regression_split_metadata(opts)
summary = get_opt(opts, 'data_summary', struct());
tf = logical(get_opt(opts, 'group_exact_duplicate_rows', false)) && ...
    strcmp(char(get_opt(summary, 'split_policy', '')), ...
    'grouped_exact_feature_target_rows');
end

function assert_static_dataset_provenance(saved_opts, data, domain)
if domain ~= "regression"
    return;
end
saved_summary = get_opt(saved_opts, 'data_summary', struct());
saved_hash = char(get_opt(saved_summary, 'dataset_sha256', ''));
current_hash = char(get_opt(data, 'dataset_sha256', ''));
if ~isempty(saved_hash) && (~isempty(current_hash) && ~strcmp(saved_hash, current_hash))
    error('snn_primary_api:regressionDatasetHashMismatch', ...
        ['The regression dataset differs from the file used for training. ', ...
         'Restore the matching dataset before testing this saved model.']);
end
end
