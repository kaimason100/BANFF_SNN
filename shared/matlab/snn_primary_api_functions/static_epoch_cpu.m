% static_epoch_cpu.m
function [loss_sum, metric, gB, Zall] = static_epoch_cpu(domain, X, Y, P, opts, order, need_grad)
%STATIC_EPOCH_CPU CPU reference for one static-task epoch/evaluation pass.
%   This is the clearest implementation of the supervised bias-learning rule:
%   output loss gradient is projected through the fixed readout and multiplied
%   elementwise by each neuron's e-prop eligibility for the bias.

n = numel(order);
gB = zeros(P.N_hidden,1,'single');
Zall = zeros(P.N_out, size(X,2), 'single');
loss_sum = single(0);
correct = 0;
for jj = 1:n
    idx = double(order(jj));
    % Forward pass for one sample: averaged output and averaged bias
    % eligibility over the readout window.
    [z, ebar_sum] = static_sample_cpu(P, X(:,idx), opts);
    % Supervised loss derivative with respect to the task output z.
    [loss, gZ, ok] = supervised_loss_grad(domain, z, Y(:,idx));
    loss_sum = loss_sum + loss;
    correct = correct + ok;
    Zall(:,idx) = z;
    if need_grad
        % e-prop bias rule:
        %   dL/dB_i ~= sum_d W_out(d,i) dL/dz_d * E_i
        % where E_i is the readout-window average eligibility of neuron i.
        gB = gB + (P.W_out.' * gZ) .* (ebar_sum ./ single(opts.steps_avg));
    end
end
if domain == "classification"
    metric = single(100 * correct / max(1,n));
else
    metric = pearson_r(Zall(:,order), Y(:,order));
end
end
