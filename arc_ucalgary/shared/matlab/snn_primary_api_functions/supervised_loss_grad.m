% supervised_loss_grad.m
function [loss, gZ, metric_ok] = supervised_loss_grad(domain, z, y)
%SUPERVISED_LOSS_GRAD Loss and readout-gradient for static tasks.
%   gZ is dL/dz, the only global learning signal needed by e-prop. The
%   hidden-neuron credit assignment is handled by multiplying W_out' * gZ by
%   local eligibility traces in static_epoch_cpu.
if domain == "classification"
    % Cross-entropy with softmax. For one-hot y, dL/dz = softmax(z) - y.
    p = softmax_single(z);
    gZ = p - y;
    loss = -sum(y .* log(max(p, realmin('single'))));
    [~, pred] = max(p);
    [~, truth] = max(y);
    metric_ok = double(pred == truth);
else
    % Standard half squared error. The half factor makes dL/dz = z - y.
    err = z - y;
    gZ = err;
    loss = single(0.5) .* sum(err .* err);
    metric_ok = 0;
end
end
