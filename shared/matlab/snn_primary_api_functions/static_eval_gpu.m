% static_eval_gpu.m
function [summary, Pg] = static_eval_gpu(domain, data, P, opts, split)
Pg = init_static_gpu(domain, data, P, opts);
static_update_bias_gpu(domain, P.B, P.W_out);
summary = static_predict_gpu(domain, data, split, opts);
Pg.B = static_get_bias_gpu(domain);
end

