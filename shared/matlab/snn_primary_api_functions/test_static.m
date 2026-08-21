% test_static.m
function result = test_static(domain, backend, opts)
if isfield(opts, 'model_files') && ~isempty(opts.model_files)
    result = test_static_model_files(domain, backend, opts);
    return;
end
if ~isfield(opts, 'model_file') || isempty(opts.model_file)
    error('snn_primary_api:modelFile', 'test_static requires opts.model_file from a training script.');
end
train_result = load_training_result(opts.model_file);
if isfield(train_result, 'seed_results')
    result = test_seed_list(@(one_result, one_opts) test_static_from_result(domain, backend, one_result, one_opts), train_result, opts);
    if logical(get_opt(opts, 'save_publication_analysis', true))
        save_publication_test_analysis(result, char(domain), opts);
    end
    return;
end
result = test_static_from_result(domain, backend, train_result, opts);
if logical(get_opt(opts, 'save_publication_analysis', true))
    save_publication_test_analysis(result, char(domain), opts);
end
end
