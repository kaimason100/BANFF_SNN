% train_static.m
function result = train_static(domain, backend, opts)
opts = merge_options_with_seed(default_static_options(domain, backend, "train"), opts);
if isfield(opts, 'seed_list') && numel(opts.seed_list) > 1
    result = train_seed_list(@(one_opts) train_static(domain, backend, rmfield_if_present(one_opts, 'seed_list')), opts);
    return;
end
data = load_static_data(domain, opts);
validate_static_data(data, domain, 'train_static');
if domain == "regression"
    % Persist the realised partition so saved-model testing cannot drift when
    % split code changes after training.
    opts.idx_train = data.idx_train;
    opts.idx_val = data.idx_val;
    opts.idx_test = data.idx_test;
end
opts.data_summary = summarize_static_data(data);
P = make_primary_model(size(data.X_train,1), size(data.Y_train,1), opts);
if backend == "gpu"
    result = train_static_gpu(domain, data, P, opts);
else
    result = train_static_cpu(domain, data, P, opts);
end
end
