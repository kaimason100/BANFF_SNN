% spike_diagnostics_static.m
function diag = spike_diagnostics_static(domain, opts)
if ~isfield(opts, 'model_file') || isempty(opts.model_file)
    error('snn_primary_api:modelFile', 'spike_diagnostics_static requires opts.model_file.');
end
train_result = load_training_result(opts.model_file);
if isfield(train_result, 'seed_results')
    seed_index = max(1, min(numel(train_result.seed_results), round(get_opt(opts, 'seed_index', 1))));
    train_result = train_result.seed_results(seed_index);
end
opts_eval = merge_options_with_seed(train_result.options, rmfield_if_present(opts, 'model_file'));
opts_eval = apply_saved_architecture_metadata(opts_eval, train_result);
max_samples = max(1, round(get_opt(opts, 'max_samples', 8)));
save_files = logical(get_opt(opts, 'save_files', false));
output_dir = get_opt(opts, 'output_dir', fullfile(project_root(), 'outputs', 'diagnostics'));
data = load_static_data(domain, opts_eval);
validate_static_data(data, domain, 'spike_diagnostics_static');
if isfield(train_result, 'model') && isstruct(train_result.model)
    P = train_result.model;
else
    P = make_primary_model(size(data.X_train,1), size(data.Y_train,1), opts_eval);
end
P = ensure_model_architecture_fields(P, opts_eval);
P.B = best_bias_from_result(train_result);
n = min(max_samples, size(data.X_test,2));
[S, U, Iin, Irec, W] = static_spike_diagnostics_cpu(P, data.X_test(:,1:n), opts_eval);
task_tag = get_opt(opts_eval, 'task_tag', char(domain));
diag = save_spiking_diagnostics_from_raster(S, opts_eval.dt, task_tag, output_dir, ...
    'context', ['test_' char(domain)], 'save_files', save_files, ...
    'u_buffer', U, 'i_in', Iin, 'i_rec', Irec, 'w', W);
end
