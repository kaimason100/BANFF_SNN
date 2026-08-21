% dynamics_eval_cpu.m
function [summary, Z] = dynamics_eval_cpu(x, lambda, P, opts)
[loss_sum, ~, Z] = dynamics_epoch_cpu(x, lambda, P, opts, false);
Z = valid_dynamics_predictions(Z, max_sequence_steps(x));
summary = struct('loss', single(loss_sum / max(1,supervised_step_count(x))), ...
    'Z', Z, 'num_valid_prediction_columns', size(Z,2));
end

