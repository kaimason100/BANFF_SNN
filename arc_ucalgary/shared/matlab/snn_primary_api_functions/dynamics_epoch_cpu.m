% dynamics_epoch_cpu.m
function [loss_sum, gB, Z] = dynamics_epoch_cpu(x, lambda, P, opts, need_grad)
%DYNAMICS_EPOCH_CPU CPU reference for dynamical-system sequence learning.
%   The network predicts the next state in normalized state coordinates. For
%   training pools, each epoch samples contiguous snippets from one long true
%   trajectory. lambda controls teacher forcing: true means feed the true
%   current state; false means feed the network's previous output.

if isstruct(x) && isfield(x, 'pool')
    % x.pool is a long normalized true trajectory. Each block is a contiguous
    % snippet sampled uniformly by its start index.
    loss_sum = single(0);
    gB = zeros(P.N_hidden,1,'single');
    Z = [];
    for bb = 1:x.train_blocks
        start_idx = randi(x.max_start_idx, 1, 'uint32');
        xb = x.pool(:, double(start_idx):double(start_idx)+x.steps-1);
        [block_loss, block_gB, block_Z] = dynamics_epoch_cpu(xb, lambda, P, opts, need_grad);
        loss_sum = loss_sum + block_loss;
        gB = gB + block_gB;
        if bb == 1, Z = block_Z; end
    end
    return;
end
if iscell(x)
    loss_sum = single(0);
    gB = zeros(P.N_hidden,1,'single');
    Z = cell(size(x));
    for bb = 1:numel(x)
        [block_loss, block_gB, block_Z] = dynamics_epoch_cpu(x{bb}, lambda{bb}, P, opts, need_grad);
        loss_sum = loss_sum + block_loss;
        gB = gB + block_gB;
        Z{bb} = block_Z;
    end
    return;
end
steps = size(x,2);
% Hidden state is reset at the start of each training/evaluation snippet.
u = single(zeros(P.N_hidden,1) + P.E_L);
w = zeros(P.N_hidden,1,'single');
x_syn = zeros(P.N_hidden,1,'single');
r = zeros(P.N_hidden,1,'single');
elig = zero_elig(P.N_hidden);
Z = zeros(P.N_out, max(1, steps-1), 'single');
Zprev = zeros(P.N_out,1,'single');
gB = zeros(P.N_hidden,1,'single');
loss_sum = single(0);
for k = 1:steps-1
    % Teacher forcing / closed-loop selection. The first step is always true
    % input. Thereafter lambda(k)=true uses x(:,k); lambda(k)=false feeds the
    % network's previous prediction Zprev back as the next input.
    if k == 1 || lambda(k)
        x_in = x(:,k);
    else
        x_in = Zprev;
    end
    % One recurrent spiking timestep driven by the chosen state input.
    I_in = P.W_in * (P.INPUT_SCALE * single(x_in));
    [u, w, rho, spike, surr, x_syn, r] = primary_step(P, I_in, u, w, x_syn, r);
    % Primary decoder predicts the next normalized dynamical-system state.
    z = P.W_out * r;
    Z(:,k) = z;
    Zprev = z;
    % One-step prediction loss against the next true state.
    err = z - x(:,k+1);
    loss_sum = loss_sum + sum(err .* err);
    if need_grad
        elig = elig_step(P, elig, rho, spike, surr);
        % Dynamical-system loss is sum(err^2), so dL/dz = 2*err. As in static
        % tasks, e-prop multiplies readout-projected error by local eligibility.
        gB = gB + (P.W_out.' * (single(2).*err)) .* elig.Ebar_f;
    end
end
end
