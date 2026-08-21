% static_spike_diagnostics_cpu.m
function [S, U, IinLog, IrecLog, Wlog] = static_spike_diagnostics_cpu(P, X, opts)
n = size(X,2);
S = false(P.N_hidden, opts.steps_present, n);
U = [];
IinLog = [];
IrecLog = [];
Wlog = [];
for sample = 1:n
    u = single(zeros(P.N_hidden,1) + P.E_L);
    w = zeros(P.N_hidden,1,'single');
    x_syn = zeros(P.N_hidden,1,'single');
    r = zeros(P.N_hidden,1,'single');
    I_in = P.W_in * (P.INPUT_SCALE * single(X(:,sample)));
    if sample == 1
        U = nan(P.N_hidden, opts.steps_present, 'single');
        IinLog = nan(P.N_hidden, opts.steps_present, 'single');
        IrecLog = nan(P.N_hidden, opts.steps_present, 'single');
        Wlog = nan(P.N_hidden, opts.steps_present, 'single');
    end
    for k = 1:opts.steps_present
        I_rec = recurrent_current(P, r);
        [u, w, rho, spike, ~] = advance_u_w(P, u, w, I_in + I_rec + P.B);
        [x_syn, r] = cascade_advance(P, rho, x_syn, r);
        x_syn(spike) = x_syn(spike) + P.spike_jump_sr;
        [x_syn, r] = cascade_advance(P, single(1)-rho, x_syn, r);
        S(:,k,sample) = spike;
        if sample == 1
            U(:,k) = u;
            IinLog(:,k) = I_in;
            IrecLog(:,k) = I_rec;
            Wlog(:,k) = w;
        end
    end
end
end
