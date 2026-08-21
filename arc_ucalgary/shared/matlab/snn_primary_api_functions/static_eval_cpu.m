% static_eval_cpu.m
function [summary, Z] = static_eval_cpu(domain, X, Y, P, opts)
[loss_sum, metric, ~, Z] = static_epoch_cpu(domain, X, Y, P, opts, int32(1:size(X,2)), false);
summary = struct('loss', single(loss_sum / max(1,size(X,2))), 'metric', single(metric), 'Z', Z, 'Y', Y);
end

