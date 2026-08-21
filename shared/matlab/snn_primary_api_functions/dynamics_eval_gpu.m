% dynamics_eval_gpu.m
function [summary, Pg] = dynamics_eval_gpu(x, lambda, P, opts)
Pg = init_dynamics_gpu(P, opts, max_sequence_steps(x));
[summary, Pg] = dynamics_eval_gpu_current(x, lambda, Pg, opts);
end

