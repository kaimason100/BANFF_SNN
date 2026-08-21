% adam_bias_update.m
function P = adam_bias_update(P, gB, lr, n_average)
%ADAM_BIAS_UPDATE Adam/AMSGrad update for the hidden bias vector only.
%   All structural weights and readouts remain fixed. gB is the accumulated
%   e-prop gradient over samples/timesteps and is averaged before applying the
%   optimizer step.
P.t_adam = P.t_adam + 1;
b1 = single(P.adam.b1); b2 = single(P.adam.b2); eps_adam = single(P.adam.eps);
% Normalize the accumulated gradient by the number of samples or transitions.
g = single(gB) ./ single(max(1,n_average));
% Adam first and second moments.
P.m_b = b1 .* P.m_b + (single(1)-b1) .* g;
P.v_b = b2 .* P.v_b + (single(1)-b2) .* (g .* g);
% Bias correction.
mhat = P.m_b ./ max(single(1)-b1.^single(P.t_adam), realmin('single'));
vhat = P.v_b ./ max(single(1)-b2.^single(P.t_adam), realmin('single'));
% AMSGrad keeps a nondecreasing second-moment denominator.
P.vhat_b = max(P.vhat_b, vhat);
P.B = P.B - lr .* (mhat ./ (sqrt(P.vhat_b) + eps_adam));
end
