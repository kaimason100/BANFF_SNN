% test_dynamics_from_result.m
function result = test_dynamics_from_result(backend, train_result, opts)
if ~isfield(train_result, 'options')
    error('snn_primary_api:modelFile', 'Saved training result does not contain options.');
end
opts_eval = merge_options_with_seed(train_result.options, rmfield_if_present(opts, 'model_file'));
opts_eval = apply_saved_architecture_metadata(opts_eval, train_result);
opts_eval.T_sim = single(get_opt(opts_eval, 'closed_loop_test_time', opts_eval.T_sim));
opts_eval.closed_loop_warmup_time = single(get_opt(opts_eval, 'closed_loop_test_warmup_time', ...
    get_opt(opts_eval, 'closed_loop_warmup_time', 5)));
opts_eval.closed_loop_validation_ics = get_opt(opts_eval, 'closed_loop_test_ics', get_opt(opts_eval, 'closed_loop_validation_ics', 1));
opts_eval.closed_loop_ic_seed = get_opt(opts_eval, 'closed_loop_test_ic_seed', 123);
if isfield(train_result, 'dynamics') && isstruct(train_result.dynamics)
    opts_eval = apply_saved_dynamics_metadata(opts_eval, train_result.dynamics);
end
opts_eval.closed_loop_ic_include_reference = logical(get_opt(opts_eval, ...
    'closed_loop_test_include_reference', false));
opts_eval.closed_loop_ic_role = 'test';
if isfield(opts_eval, 'closed_loop_x0_list') && ~isfield(opts, 'closed_loop_x0_list')
    opts_eval = rmfield(opts_eval, 'closed_loop_x0_list');
end
if isfield(train_result, 'model') && isstruct(train_result.model)
    P = train_result.model;
else
    sys = make_dynamics_system_for_api(opts_eval.system_name);
    P = make_primary_model(sys.dim, sys.dim, opts_eval);
end
P = ensure_model_architecture_fields(P, opts_eval);
P.B = best_bias_from_result(train_result);
eval_set = make_closed_loop_eval_set(opts_eval);
assert_closed_loop_test_ics_held_out(eval_set, train_result.options);
if backend == "gpu"
    test = dynamics_closed_loop_evaluation_gpu(P, opts_eval, eval_set);
else
    test = dynamics_closed_loop_evaluation_cpu(P, opts_eval, eval_set);
end
test.test_initial_conditions_held_out_from_validation = true;
test.validation_test_initial_condition_overlap_count = 0;
result = struct();
result.train_backend = get_result_field(train_result, 'backend', 'unknown');
result.test_backend = char(backend);
result.model_file = char(opts.model_file);
result.test = test;
result.options = opts_eval;
result = attach_architecture_metadata(result, P, opts_eval);
end
