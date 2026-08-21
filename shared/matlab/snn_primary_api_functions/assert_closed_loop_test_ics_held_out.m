% assert_closed_loop_test_ics_held_out.m
function assert_closed_loop_test_ics_held_out(test_eval_set, training_opts)
%ASSERT_CLOSED_LOOP_TEST_ICS_HELD_OUT Reject validation/test IC overlap.
previous_rng = rng;
cleanup_rng = onCleanup(@() rng(previous_rng)); %#ok<NASGU>
sys = make_dynamics_system_for_api(test_eval_set.opts.system_name);
validation_opts = training_opts;
validation_opts.closed_loop_ic_include_reference = logical(get_opt( ...
    training_opts, 'closed_loop_ic_include_reference', true));
validation_x0 = closed_loop_validation_initial_conditions(sys, validation_opts);
test_x0 = single(test_eval_set.x0_list);
overlap = false(size(validation_x0, 2), size(test_x0, 2));
for vv = 1:size(validation_x0, 2)
    for tt = 1:size(test_x0, 2)
        overlap(vv, tt) = isequal(validation_x0(:, vv), test_x0(:, tt));
    end
end
if any(overlap, 'all')
    [validation_index, test_index] = find(overlap, 1, 'first');
    error('snn_primary_api:sharedValidationTestIC', ...
        ['Test initial condition %d is identical to validation initial condition %d. ', ...
         'Use a separate test seed/list and keep closed_loop_test_include_reference=false.'], ...
        test_index, validation_index);
end
end

