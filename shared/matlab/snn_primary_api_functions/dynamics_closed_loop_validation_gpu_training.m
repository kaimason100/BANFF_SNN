% dynamics_closed_loop_validation_gpu_training.m
function [closed, Pg] = dynamics_closed_loop_validation_gpu_training(x_train, Pg, opts, eval_set)
%DYNAMICS_CLOSED_LOOP_VALIDATION_GPU_TRAINING Validate without losing optimizer state.
%   The dynamics MEX has a fixed sequence-length capacity. Training snippets
%   and long validation rollouts therefore use different MEX initialisations.
%   Preserve the bias optimizer buffers, run lightweight GPU closed-loop
%   validation, then restore the training-sized MEX and optimizer state.
optim_state = dynamics_get_optimizer_state_gpu();
Pg.B = dynamics_get_bias_gpu();
P_eval = Pg;
closed = dynamics_closed_loop_evaluation_gpu(P_eval, opts, eval_set);
Pg.B = P_eval.B;
Pg = init_dynamics_gpu(Pg, opts, max_sequence_steps(x_train));
dynamics_update_bias_gpu(Pg.B);
dynamics_set_optimizer_state_gpu(optim_state);
end
