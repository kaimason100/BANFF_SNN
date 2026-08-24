% Package orientation: Plotting/figure script. It consumes saved result structs or generated arrays and formats diagnostics or publication figures without changing model training.

% make_FI_nonconstant_figure.m
% Publication-ready F–I: steady-state curves WITH vs WITHOUT adaptation (SE-adLIF)
% Dynamics use Option B (advance–reset–advance) with LSTI fractional timing.
% We run a warmup to steady state, then average rate over >=10 s.
% Kai Mason — 26 Sep 2025

clear; clc; close all;
rng(42,'twister');

%% ---------------- Neuron & sim params (mirrors your script) ----------------
dt        = single(1e-3);             % [s]
tau_u     = single(50e-3);
tau_w     = single(500e-3);
E_L       = single(-70);
V_th      = single(-50);
V_reset   = single(-65);

% Purely spike-triggered adaptation, matching the BANFF model.
b_param   = single(0.5);              % spike-triggered jump (mV)

alpha     = exp(-dt/tau_u);
beta      = exp(-dt/tau_w);
oneMinusAlpha = single(1) - alpha;

% Bias (constant offset) — keep at 0 mV for clarity
B = single(0);

% Rate estimator (causal exponential smoother), dt-invariant with LSTI
tau_rate   = single(50e-3);
gamma_r    = exp(-dt/tau_rate);
spike_jump_r = -log(max(gamma_r, realmin('single'))) / dt;   % == 1/tau_rate

% DC input sweep [mV]; include sub- and supra-threshold
I_min = single(0);  I_max = single(50);  nI = 100;
I_list = single(linspace(I_min, I_max, nI));

% ---------- Simulation schedule ----------
% Warmup long enough for adaptation to settle; then measure >=10 s.
T_warmup = single(max(8*tau_w, 8*tau_u));   % conservative steady-state warmup
T_meas   = single(10.0);                    % >= 10 s measurement window
steps_warm = round(T_warmup/dt);
steps_meas = round(T_meas/dt);
steps_total = steps_warm + steps_meas;

%% ---------------- Output arrays ----------------
F_noAdapt = zeros(nI,1,'single');     % steady-state rate without adaptation
F_adapt   = zeros(nI,1,'single');     % steady-state rate with adaptation

%% ---------------- Main loop over DC inputs ----------------
for ii = 1:nI
    I_dc = I_list(ii);

    % ---------- Case 1: WITHOUT adaptation ----------
    u = single(E_L); w = single(0);          % w held at 0 throughout
    r_inst = single(0);

    sum_rate = single(0); cnt = 0;

    for k = 1:steps_total
        I_tot = I_dc + B;

        % Predict u_hat for spike test (w == 0 here)
        u_pre = u;
        u_hat = E_L + alpha*(u_pre - E_L) + oneMinusAlpha*(I_tot);   % no w

        spike = (u_hat >= V_th);

        % Fractional crossing time rho
        u_diff = u_hat - u_pre;
        den    = max(u_diff, realmin('single'));
        rho    = single(0);
        if spike && u_diff > 0
            rho = (V_th - u_pre) ./ den;
            rho = min(max(single(rho), single(0)), single(1));
            if ~isfinite(rho), rho = single(0); end
        end

        % Split-step decays (w is unused here; keep for symmetry)
        alpha_pre  = exp( rho    .* log(max(alpha, realmin('single'))) );
        alpha_post = exp( (1-rho).* log(max(alpha, realmin('single'))) );
        oneMinusAlpha_pre  = single(1) - alpha_pre;
        oneMinusAlpha_post = single(1) - alpha_post;

        % (1) pre to t*
        u_star = E_L + alpha_pre .* (u_pre - E_L) + oneMinusAlpha_pre .* (I_tot);
        % (2) reset at t*
        if spike, u_star = V_reset; end
        % (3) post to t_{k+1}
        u = E_L + alpha_post .* (u_star - E_L) + oneMinusAlpha_post .* (I_tot);

        % LSTI rate filter
        f_pre  = rho;
        f_post = single(1) - rho;
        r_inst = advance_single_exp_frac(r_inst, f_pre,  gamma_r, 0);
        if spike, r_inst = r_inst + spike_jump_r; end
        r_inst = advance_single_exp_frac(r_inst, f_post, gamma_r, 0);

        % Accumulate only in measurement window
        if k > steps_warm
            sum_rate = sum_rate + r_inst;
            cnt = cnt + 1;
        end
    end
    F_noAdapt(ii) = sum_rate / max(1,cnt);

    % ---------- Case 2: WITH adaptation ----------
    u = single(E_L); w = single(0);
    r_inst = single(0);
    sum_rate = single(0); cnt = 0;

    for k = 1:steps_total
        I_tot = I_dc + B;

        % Predict u_hat for spike test (uses current w)
        u_pre = u;
        u_hat = E_L + alpha*(u_pre - E_L) + oneMinusAlpha*(I_tot - w);
        spike = (u_hat >= V_th);

        % Fractional crossing time rho
        u_diff = u_hat - u_pre;
        den    = max(u_diff, realmin('single'));
        rho    = single(0);
        if spike && u_diff > 0
            rho = (V_th - u_pre) ./ den;
            rho = min(max(single(rho), single(0)), single(1));
            if ~isfinite(rho), rho = single(0); end
        end

        % Split-step decays
        alpha_pre  = exp( rho    .* log(max(alpha, realmin('single'))) );
        beta_pre   = exp( rho    .* log(max(beta,  realmin('single'))) );
        alpha_post = exp( (1-rho).* log(max(alpha, realmin('single'))) );
        beta_post  = exp( (1-rho).* log(max(beta,  realmin('single'))) );

        oneMinusAlpha_pre  = single(1) - alpha_pre;
        oneMinusAlpha_post = single(1) - alpha_post;

        % (1) pre to t*
        u_star = E_L + alpha_pre .* (u_pre - E_L) + oneMinusAlpha_pre .* (I_tot - w);
        w      = beta_pre .* w;

        % (2) instantaneous spike/reset at t*
        if spike
            u_star = V_reset;
            w      = w + b_param;
        end

        % (3) post to t_{k+1}
        u = E_L + alpha_post .* (u_star - E_L) + oneMinusAlpha_post .* (I_tot - w);
        w = beta_post .* w;

        % LSTI rate filter
        f_pre  = rho;
        f_post = single(1) - rho;
        r_inst = advance_single_exp_frac(r_inst, f_pre,  gamma_r, 0);
        if spike, r_inst = r_inst + spike_jump_r; end
        r_inst = advance_single_exp_frac(r_inst, f_post, gamma_r, 0);

        % Accumulate only in measurement window
        if k > steps_warm
            sum_rate = sum_rate + r_inst;
            cnt = cnt + 1;
        end
    end
    F_adapt(ii) = sum_rate / max(1,cnt);
end

%% ---------------- Figure (publication-ready) ----------------
fig = figure('Color','w','Units','centimeters','Position',[2 2 18 10]); %#ok<NASGU>
plot(double(I_list), double(F_noAdapt), '-', 'LineWidth', 2.5); hold on;
plot(double(I_list), double(F_adapt),   '-', 'LineWidth', 2.5);
hold off
xlabel('DC Input (mV)');
ylabel('Firing Rate (Hz)');
title(sprintf('SE-adLIF F–I (steady-state): warmup %.2fs, measure %.2fs', double(T_warmup), double(T_meas)));

set(gca,'FontName','Helvetica','LineWidth',1,'TickDir','out');
% Threshold (E_L→V_th) reference at ~20 mV
xline(20,'k:','LineWidth',3);
legend('No adaptation (w\equiv0)', 'With adaptation', 'Threshold', ...
       'Location','northwest','Box','off');
set(gca,'fontsize',15)

%% ---------------- Helpers ----------------
function y = advance_single_exp_frac(y, f, gamma, z_const)
% Fractional advance for a single-stage causal exponential smoother:
% y <- gamma^f * y + (1 - gamma^f) * z_const
% z_const = 0 (no continuous input); spikes handled as instantaneous impulses.
    f = single(f);
    g_f = exp( f .* log(max(gamma, realmin('single'))) );
    y = g_f .* y + (1 - g_f) .* single(z_const);
end
