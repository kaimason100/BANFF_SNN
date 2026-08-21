% dynamics_spike_diagnostics_cpu.m
function [S, U, IinLog, IrecLog, Wlog] = dynamics_spike_diagnostics_cpu(P, x, lambda, opts)
steps = size(x,2);
u = single(zeros(P.N_hidden,1) + P.E_L);
w = zeros(P.N_hidden,1,'single');
x_syn = zeros(P.N_hidden,1,'single');
r = zeros(P.N_hidden,1,'single');
Zprev = zeros(P.N_out,1,'single');
S = false(P.N_hidden, max(1,steps-1));
U = nan(P.N_hidden, max(1,steps-1), 'single');
IinLog = nan(P.N_hidden, max(1,steps-1), 'single');
IrecLog = nan(P.N_hidden, max(1,steps-1), 'single');
Wlog = nan(P.N_hidden, max(1,steps-1), 'single');
for k = 1:steps-1
    if k == 1 || lambda(k)
        x_in = x(:,k);
    else
        x_in = Zprev;
    end
    I_in = P.W_in * (P.INPUT_SCALE * single(x_in));
    I_rec = recurrent_current(P, r);
    [u, w, rho, spike, ~] = advance_u_w(P, u, w, I_in + I_rec + P.B);
    [x_syn, r] = cascade_advance(P, rho, x_syn, r);
    x_syn(spike) = x_syn(spike) + P.spike_jump_sr;
    [x_syn, r] = cascade_advance(P, single(1)-rho, x_syn, r);
    Zprev = P.W_out * r;
    S(:,k) = spike;
    U(:,k) = u;
    IinLog(:,k) = I_in;
    IrecLog(:,k) = I_rec;
    Wlog(:,k) = w;
end
end
