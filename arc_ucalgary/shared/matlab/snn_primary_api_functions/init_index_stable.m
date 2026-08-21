% init_index_stable.m
function [W_in, W_out_base_rec, W_out, Eta_rec, dself, dale_sign] = init_index_stable( ...
    N_in, N_hidden, N_out_task, N_rec, dale, SCALE, p_rec, do_varcorr, seed)
%INIT_INDEX_STABLE Fixed random network initialization.
%   Every random value is generated from its semantic index and seed rather
%   than the mutable RNG stream. This means extending a matrix does not change
%   the already-existing submatrix, which is useful for comparing tasks/seeds.
precision = 'single';
uU = @(mid,i,j) u01_indexstable(mid,i,j,seed);
uN = @(mid,i,j) sqrt(2)*erfinv(2*uU(mid,i,j)-1);
% Dale sign per hidden neuron. If enabled, outgoing decoder rows obey this
% sign convention by making each hidden unit effectively excitatory/inhibitory.
p_exc = get_opt(dale, 'p_exc', 0.5);
dale_sign = -ones(1,N_hidden,precision);
dale_sign(uU(5,1:N_hidden,0) <= p_exc) = 1;

% Dense fixed input encoder. SCALE.enc controls the input current amplitude
% before the global 1/sqrt(N_in) factor applied at runtime.
a_in = sqrt(3) * SCALE.enc;
W_in = zeros(N_hidden, N_in, precision);
for h = 1:N_hidden
    u = uU(1,h,1:N_in);
    W_in(h,:) = a_in * (2*single(u)-1);
end

% Fixed decoder basis from hidden synaptic state r to a low-dimensional
% recurrent basis. A provisional shared task decoder uses its first N_out
% rows; make_primary_model replaces this in independent signed-decoder mode.
a_dec = sqrt(3) / sqrt(max(1,N_hidden));
W_out_base_rec = zeros(N_rec, N_hidden, precision);
for r = 1:N_rec
    u = uU(20, r, 1:N_hidden);
    row = a_dec * (2*single(u)-1);
    if isfield(dale,'enable') && dale.enable
        row = abs(row) .* single(dale_sign(:)).';
    end
    W_out_base_rec(r,:) = row;
end
W_out = single(SCALE.dec) * W_out_base_rec(1:N_out_task,:);

% Fixed encoder from low-dimensional decoded recurrent state back to hidden
% recurrent current. Sparsity is controlled by p_rec. Variance correction
% divides by sqrt(p_rec) so current variance is roughly preserved as sparsity
% changes.
Eta_rec = zeros(N_hidden, N_rec, precision);
scale_eta = single(1) / sqrt(single(max(1,double(N_rec))));
for r = 1:N_rec
    z = single(uN(30,(1:N_hidden).', r));
    m = single(uU(40,(1:N_hidden).', r) < p_rec);
    col = scale_eta * z .* m;
    if do_varcorr && p_rec > 0, col = col ./ sqrt(single(p_rec)); end
    if isfield(dale,'enable') && dale.enable, col = abs(col); end
    Eta_rec(:,r) = col;
end
% Self-feedback correction removes each neuron's direct contribution through
% the decoder-encoder loop, preventing an artificial instantaneous self-term.
dself = SCALE.rec * sum(Eta_rec .* W_out_base_rec.', 2);
end
