% spike_diagnostics_dynamics.m
function diag = spike_diagnostics_dynamics(opts)
if ~isfield(opts, 'model_file') || isempty(opts.model_file)
    error('snn_primary_api:modelFile', 'spike_diagnostics_dynamics requires opts.model_file.');
end
train_result = load_training_result(opts.model_file);
if isfield(train_result, 'seed_results')
    seed_index = max(1, min(numel(train_result.seed_results), round(get_opt(opts, 'seed_index', 1))));
    train_result = train_result.seed_results(seed_index);
end
opts_eval = merge_options_with_seed(train_result.options, rmfield_if_present(opts, 'model_file'));
opts_eval = apply_saved_architecture_metadata(opts_eval, train_result);
opts_eval.dynamics_split = 'closed_loop';
save_files = logical(get_opt(opts, 'save_files', false));
output_dir = get_opt(opts, 'output_dir', fullfile(project_root(), 'outputs', 'diagnostics'));
[x, P] = make_dynamics_problem(opts_eval);
if isfield(train_result, 'model') && isstruct(train_result.model)
    P = train_result.model;
end
P = ensure_model_architecture_fields(P, opts_eval);
lambda = make_closed_loop_lambda(size(x,2));
validate_dynamics_data(x, lambda, 'spike_diagnostics_dynamics');
P.B = best_bias_from_result(train_result);
[S, U, Iin, Irec, W] = dynamics_spike_diagnostics_cpu(P, x, lambda, opts_eval);
diag = save_spiking_diagnostics_from_raster(S, opts_eval.dt, opts_eval.system_name, output_dir, ...
    'context', 'test_dynamics', 'save_files', save_files, ...
    'u_buffer', U, 'i_in', Iin, 'i_rec', Irec, 'w', W);
end
