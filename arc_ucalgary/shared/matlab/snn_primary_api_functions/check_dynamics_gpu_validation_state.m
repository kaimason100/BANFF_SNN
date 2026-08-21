% check_dynamics_gpu_validation_state.m
function result = check_dynamics_gpu_validation_state(opts)
%CHECK_DYNAMICS_GPU_VALIDATION_STATE Verify validation restores GPU state.
opts = merge_options_with_seed(default_dynamics_options("gpu", "check"), opts);
opts.N_hidden = min(double(get_opt(opts, 'N_hidden', 32)), 64);
opts.N_rec = min(double(get_opt(opts, 'N_rec', 4)), opts.N_hidden);
opts.T_sim = single(get_opt(opts, 'T_sim', 0.030));
opts.long_sim_time = single(get_opt(opts, 'long_sim_time', 0.120));
opts.burn_in_time = single(0);
opts.closed_loop_validation_time = single(get_opt(opts, 'closed_loop_validation_time', 0.030));
opts.closed_loop_validation_ics = 1;
opts.dynamics_split = 'train';
if exist('snn_time_loop_gpu_mex', 'file') ~= 3
    result = struct('status', 'skipped', 'reason', 'snn_time_loop_gpu_mex is not compiled or not on the MATLAB path.');
    return;
end
[x_train, P] = make_dynamics_problem(opts);
lambda = make_lambda_sequence_for_data(x_train, opts); %#ok<NASGU>
Pg = init_dynamics_gpu(P, opts, max_sequence_steps(x_train));
cleanup = onCleanup(@() clear_dynamics_gpu()); %#ok<NASGU>
state_before = dynamics_get_optimizer_state_gpu();
bias_before = dynamics_get_bias_gpu();
eval_set = make_closed_loop_eval_set(opts);
[~, Pg] = dynamics_closed_loop_validation_gpu_training(x_train, Pg, opts, eval_set);
state_after = dynamics_get_optimizer_state_gpu();
bias_after = dynamics_get_bias_gpu();
result = struct();
result.status = 'checked';
result.bias_restored = compare_arrays(bias_before, bias_after);
result.m_b_restored = compare_arrays(state_before.m_b, state_after.m_b);
result.v_b_restored = compare_arrays(state_before.v_b, state_after.v_b);
result.vhat_b_restored = compare_arrays(state_before.vhat_b, state_after.vhat_b);
result.t_adam_before = state_before.t_adam;
result.t_adam_after = state_after.t_adam;
assert_check_passed(result.bias_restored, opts.check_tolerance, 'GPU validation restored bias');
assert_check_passed(result.m_b_restored, opts.check_tolerance, 'GPU validation restored Adam m');
assert_check_passed(result.v_b_restored, opts.check_tolerance, 'GPU validation restored Adam v');
assert_check_passed(result.vhat_b_restored, opts.check_tolerance, 'GPU validation restored AMSGrad vhat');
if state_before.t_adam ~= state_after.t_adam
    error('snn_primary_api:gpuValidationStateNotRestored', ...
        'GPU validation changed t_adam from %g to %g.', state_before.t_adam, state_after.t_adam);
end
end

