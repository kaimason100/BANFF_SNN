% test_dynamics.m
function result = test_dynamics(backend, opts)
if isfield(opts, 'model_files') && ~isempty(opts.model_files)
    result = test_dynamics_model_files(backend, opts);
    return;
end
if ~isfield(opts, 'model_file') || isempty(opts.model_file)
    error('snn_primary_api:modelFile', 'test_dynamics requires opts.model_file from a training script.');
end
train_result = load_training_result(opts.model_file);
if isfield(train_result, 'seed_results')
    result = test_seed_list(@(one_result, one_opts) test_dynamics_from_result(backend, one_result, one_opts), train_result, opts);
    if logical(get_opt(opts, 'save_publication_analysis', true))
        save_publication_test_analysis(result, 'dynamical_systems', opts);
    end
    return;
end
result = test_dynamics_from_result(backend, train_result, opts);
if logical(get_opt(opts, 'save_publication_analysis', true))
    save_publication_test_analysis(result, 'dynamical_systems', opts);
end
end
