%% Autonomous near-threshold recurrence and transient-cascade diagnostic
% This evaluation-only script initializes the publication network randomly
% around spike threshold, supplies no encoder input, and lets recurrent and
% intrinsic dynamics evolve. It resolves the initial cascade explicitly,
% rather than examining only the late-time rate. The measurements distinguish
% an isolated initial volley, a finite recurrent cascade, persistent regular
% activity, and sensitive persistent activity.
%
% Nearby-trajectory separation is an operational finite-network diagnostic;
% it is not a formal Lyapunov exponent or proof of chaos.

clearvars;
close all;
clc;
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);

%% Editable settings
task = 'breast_cancer';
N_hidden = 32000;
N_recurrent = 10;
recurrent_gains = single([0.005 0.01 0.025 0.05 0.075 0.10 0.15 ...
    0.25 0.40 0.60 1.0 1.5 2.0 3.0 5.0]);
simulation_time_s = 2.0;
initial_voltage_half_width_mV = single(1.0);
trajectory_perturbation_mV = single(1e-3);
diagnostic_stride = 5;
extinction_quiet_time_s = 0.100;
random_seed = 731;
example_gain = single(0.10);
run_mechanism_controls = true;
control_bias_increment_mV = single(0.25);
control_simulation_time_s = 5.0;
run_selected_gain_controls = true;
selected_control_gains = single([0.05 0.10 0.40 1.0 5.0]);
run_initial_fraction_controls = true;
initial_spiking_fractions = [0.001 0.01 0.10 0.50];
estimate_recurrent_spectrum = true;
spectrum_eigenvalue_count = 6;

if ~canUseGPU
    error('banff:edgeDiagnosticGPU', ...
        'This production-step diagnostic requires a supported MATLAB GPU.');
end
if N_hidden<1 || N_recurrent<1 || simulation_time_s<=0 || ...
        diagnostic_stride<1 || extinction_quiet_time_s<0 || ...
        any(~isfinite(recurrent_gains)) || any(recurrent_gains<0) || ...
        initial_voltage_half_width_mV<=0 || trajectory_perturbation_mV<=0 || ...
        control_simulation_time_s<=0 || any(~isfinite(selected_control_gains)) || ...
        any(selected_control_gains<0) || any(~isfinite(initial_spiking_fractions)) || ...
        any(initial_spiking_fractions<=0 | initial_spiking_fractions>=1) || ...
        ~isfinite(control_bias_increment_mV) || ~isfinite(spectrum_eigenvalue_count) || ...
        spectrum_eigenvalue_count<1
    error('banff:edgeDiagnosticSettings', ...
        'Network size, durations, gains, and perturbations must be finite and valid.');
end

%% Construct one deterministic unit-gain scaffold
% Low-rank matrices are independent of recurrent gain. Both the factorized
% operator and its exact diagonal self-coupling correction are scaled together
% for each condition, so the sweep changes only recurrent gain.
overrides = struct('N_hidden',N_hidden,'N_recurrent',N_recurrent, ...
    'recurrent_mode','low_rank','recurrent_gain',single(1));
cfg = banff('config',task,overrides);
Punit = banff_model('create',1,1,cfg);
unit_self_coupling = Punit.self_coupling;

unit_operator_spectral_radius = NaN;
if estimate_recurrent_spectrum
    [unit_operator_spectral_radius,dominant_operator_eigenvalues,spectrum_flag] = ...
        recurrent_operator_spectrum(Punit,spectrum_eigenvalue_count);
    fprintf(['Unit-gain recurrent-current operator spectral radius: %.6g ', ...
        '(eigs flag %d)\n'],unit_operator_spectral_radius,spectrum_flag);
    disp(table(dominant_operator_eigenvalues,'VariableNames',{'DominantEigenvalue'}));
end

settings = struct();
settings.step_count = round(simulation_time_s/double(cfg.dt));
settings.sample_steps = 1:diagnostic_stride:settings.step_count;
settings.diagnostic_stride = diagnostic_stride;
settings.tail_start = max(1,round(0.75*settings.step_count));
settings.quiet_steps = round(extinction_quiet_time_s/double(cfg.dt));
settings.initial_voltage_half_width_mV = initial_voltage_half_width_mV;
settings.trajectory_perturbation_mV = trajectory_perturbation_mV;
settings.random_seed = random_seed;
settings.dt = double(cfg.dt);
settings.threshold_voltage = single(cfg.threshold_voltage);
settings.initialization_mode = "uniform_around_threshold";
settings.initial_spiking_fraction = 0.5;

gain_count = numel(recurrent_gains);
stored = cell(gain_count,1);
fprintf('\nAutonomous near-threshold recurrence diagnostic\n');
fprintf(['N = %d, rank = %d, duration = %.3g s, no external input; ', ...
    'initial voltages uniform within +/-%.3g mV of threshold\n\n'], ...
    N_hidden,N_recurrent,simulation_time_s,initial_voltage_half_width_mV);

for gain_index = 1:gain_count
    stored{gain_index} = simulate_condition(Punit,unit_self_coupling, ...
        recurrent_gains(gain_index),single(cfg.adaptation_jump),single(0),settings);
    S = stored{gain_index};
    fprintf(['gain %.4g: initial %.2f%% spiking, %.4g subsequent spikes/neuron, ', ...
        'last spike %.4g s, tail %.4g Hz, tail-active neurons %.2f%%\n'], ...
        recurrent_gains(gain_index),S.initial_spiking_percent, ...
        S.subsequent_spikes_per_neuron,S.last_spike_time_s,S.tail_rate_hz, ...
        S.tail_active_neuron_percent);
end

%% Assemble a reviewer-readable summary table
initial_spiking_percent = cellfun(@(S)S.initial_spiking_percent,stored);
subsequent_spikes_per_neuron = cellfun(@(S)S.subsequent_spikes_per_neuron,stored);
mean_rate_first_50ms_hz = cellfun(@(S)S.mean_rate_first_50ms_hz,stored);
mean_rate_first_250ms_hz = cellfun(@(S)S.mean_rate_first_250ms_hz,stored);
peak_rate_hz = cellfun(@(S)S.peak_rate_hz,stored);
last_spike_time_s = cellfun(@(S)S.last_spike_time_s,stored);
extinguished = cellfun(@(S)S.extinguished,stored);
tail_rate_hz = cellfun(@(S)S.tail_rate_hz,stored);
tail_active_neuron_percent = cellfun(@(S)S.tail_active_neuron_percent,stored);
peak_net_recurrent_mV = cellfun(@(S)S.peak_net_recurrent_mV,stored);
peak_gross_recurrent_mV = cellfun(@(S)S.peak_gross_recurrent_mV,stored);
peak_adaptation_mV = cellfun(@(S)S.peak_adaptation_mV,stored);
mean_recurrent_cancellation_ratio = ...
    cellfun(@(S)S.mean_recurrent_cancellation_ratio,stored);
early_log_voltage_separation_slope = ...
    cellfun(@(S)S.early_log_voltage_separation_slope,stored);
separation_fit_end_s = cellfun(@(S)S.separation_fit_end_s,stored);
numerically_finite = cellfun(@(S)S.numerically_finite,stored);
linear_operator_spectral_radius = ...
    double(recurrent_gains(:))*unit_operator_spectral_radius;

results = table(double(recurrent_gains(:)),linear_operator_spectral_radius, ...
    initial_spiking_percent, ...
    subsequent_spikes_per_neuron,mean_rate_first_50ms_hz, ...
    mean_rate_first_250ms_hz,peak_rate_hz,last_spike_time_s,extinguished, ...
    tail_rate_hz,tail_active_neuron_percent,peak_net_recurrent_mV, ...
    peak_gross_recurrent_mV,peak_adaptation_mV, ...
    mean_recurrent_cancellation_ratio,early_log_voltage_separation_slope, ...
    separation_fit_end_s,numerically_finite,'VariableNames',{'RecurrentGain', ...
    'LinearOperatorSpectralRadius','InitialSpikingPercent', ...
    'SubsequentSpikesPerNeuron','MeanRateFirst50msHz', ...
    'MeanRateFirst250msHz','PeakRateHz','LastSpikeTimeSeconds','Extinguished', ...
    'TailRateHz','TailActiveNeuronPercent','PeakNetRecurrentMv', ...
    'PeakGrossRecurrentMv','PeakAdaptationMv', ...
    'MeanNetToGrossRecurrentRatio','ActiveCascadeLogSeparationSlopePerSecond', ...
    'SeparationFitEndSeconds','NumericallyFinite'});
disp(results);

%% Gain-sweep figures: show the transient rather than only the tail
colors = parula(gain_count);
figure('Color','w');
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');
nexttile; hold on;
for gain_index = 1:gain_count
    plot(stored{gain_index}.time_s,stored{gain_index}.rate_hz(:,1), ...
        'Color',colors(gain_index,:),'LineWidth',0.9, ...
        'DisplayName',sprintf('g=%.4g',recurrent_gains(gain_index)));
end
hold off; grid on; xlim([0 min(simulation_time_s,0.5)]);
xlabel('Time (s)'); ylabel('Population rate (Hz)');
title('Resolved initial recurrent cascades'); legend('Location','eastoutside');

nexttile;
semilogx(double(recurrent_gains),subsequent_spikes_per_neuron,'o-','LineWidth',1.2);
grid on; xlabel('Recurrent gain'); ylabel('Spikes/neuron after first step');
title('Recruitment beyond imposed initial volley');

nexttile;
semilogx(double(recurrent_gains),last_spike_time_s,'o-','LineWidth',1.2);
yline(simulation_time_s-extinction_quiet_time_s,'k--','Quiet-window boundary');
grid on; xlabel('Recurrent gain'); ylabel('Time of final spike (s)');
title('Cascade lifetime');

nexttile; hold on;
semilogx(double(recurrent_gains),peak_net_recurrent_mV,'o-','LineWidth',1.2);
semilogx(double(recurrent_gains),peak_gross_recurrent_mV,'s-','LineWidth',1.2);
semilogx(double(recurrent_gains),peak_adaptation_mV,'^-','LineWidth',1.2);
hold off; grid on; xlabel('Recurrent gain'); ylabel('Mean magnitude (mV)');
title('Peak recurrent drive and adaptation');
legend({'Net recurrent','Gross recurrent','Adaptation'},'Location','best');

nexttile;
semilogx(double(recurrent_gains),mean_recurrent_cancellation_ratio,'o-','LineWidth',1.2);
grid on; ylim([0 1]); xlabel('Recurrent gain');
ylabel('Mean |net| / gross recurrent magnitude');
title('Excitatory/inhibitory cancellation');

nexttile; yyaxis left;
semilogx(double(recurrent_gains),tail_rate_hz,'o-','LineWidth',1.2);
ylabel('Tail population rate (Hz)');
yyaxis right;
semilogx(double(recurrent_gains),tail_active_neuron_percent,'s-','LineWidth',1.2);
ylabel('Tail-active neurons (%)'); grid on; xlabel('Recurrent gain');
title('True late-time persistence');
sgtitle(sprintf('BANFF autonomous gain sweep: N=%d, no external input',N_hidden));

%% Detailed mechanism trace at one selected gain
[~,example_index] = min(abs(double(recurrent_gains)-double(example_gain)));
example = stored{example_index};
figure('Color','w');
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');
nexttile;
plot(example.time_s,example.rate_hz(:,1),'LineWidth',1.0); grid on;
xlabel('Time (s)'); ylabel('Population rate (Hz)');
title(sprintf('Activity at gain %.4g',recurrent_gains(example_index)));
nexttile; hold on;
plot(example.diagnostic_time_s,example.net_recurrent_mV,'LineWidth',1.1);
plot(example.diagnostic_time_s,example.gross_recurrent_mV,'LineWidth',1.1);
hold off; grid on; xlabel('Time (s)'); ylabel('Mean magnitude (mV)');
title('Recurrent drive'); legend({'|net|','gross before cancellation'});
nexttile;
plot(example.diagnostic_time_s,example.adaptation_mV,'LineWidth',1.1); grid on;
xlabel('Time (s)'); ylabel('Mean |adaptation| (mV)'); title('Adaptation buildup');
nexttile;
plot(example.diagnostic_time_s,example.filtered_spike_state,'LineWidth',1.1); grid on;
xlabel('Time (s)'); ylabel('Mean filtered-spike state'); title('Synaptic trace decay');
nexttile;
semilogy(example.diagnostic_time_s,max(example.voltage_separation_mV,realmin), ...
    'LineWidth',1.1); grid on;
xlabel('Time (s)'); ylabel('RMS voltage separation (mV)');
title('Nearby trajectories');
nexttile;
plot(example.time_s,100*example.spike_disagreement,'LineWidth',1.0); grid on;
xlabel('Time (s)'); ylabel('Spike disagreement (% neurons)');
title('Instantaneous trajectory disagreement');
sgtitle('Mechanism trace: recurrence, adaptation, and extinction');

%% Full factorial mechanism controls at the selected gain
% Recurrence, adaptation and the small suprathreshold bias are crossed in a
% 2-by-2-by-2 design. In particular, bias-plus-zero-recurrence separates
% intrinsically generated firing from recurrently maintained firing. The
% longer horizon resolves adaptation-driven silent intervals or bursting.
if run_mechanism_controls
    long_settings = settings_with_duration(settings,control_simulation_time_s);
    factor_recurrence = [false true];
    factor_adaptation = [false true];
    factor_bias = [false true];
    control_count = numel(factor_recurrence)*numel(factor_adaptation)*numel(factor_bias);
    controls = cell(control_count,1);
    recurrence_on = false(control_count,1);
    adaptation_on = false(control_count,1);
    bias_increment_mV = zeros(control_count,1);
    control_label = cell(control_count,1);
    control_index = 0;
    for recurrence_index = 1:numel(factor_recurrence)
        for adaptation_index = 1:numel(factor_adaptation)
            for bias_index = 1:numel(factor_bias)
                control_index = control_index+1;
                recurrence_on(control_index) = factor_recurrence(recurrence_index);
                adaptation_on(control_index) = factor_adaptation(adaptation_index);
                bias_increment_mV(control_index) = ...
                    double(factor_bias(bias_index))*double(control_bias_increment_mV);
                local_gain = single(recurrence_on(control_index))* ...
                    recurrent_gains(example_index);
                local_adaptation = single(adaptation_on(control_index))* ...
                    single(cfg.adaptation_jump);
                controls{control_index} = simulate_condition(Punit,unit_self_coupling, ...
                    local_gain,local_adaptation,single(bias_increment_mV(control_index)), ...
                    long_settings);
                control_label{control_index} = sprintf('rec=%d, adapt=%d, bias=%+.3g mV', ...
                    recurrence_on(control_index),adaptation_on(control_index), ...
                    bias_increment_mV(control_index));
            end
        end
    end
    control_results = condition_table(controls);
    control_results = addvars(control_results,recurrence_on,adaptation_on, ...
        bias_increment_mV,'Before',1,'NewVariableNames', ...
        {'RecurrenceOn','AdaptationOn','BiasIncrementMv'});
    fprintf('\n%.3g-second 2 x 2 x 2 mechanism controls at gain %.4g\n', ...
        control_simulation_time_s,recurrent_gains(example_index));
    disp(control_results);

    figure('Color','w'); hold on;
    for control_index = 1:control_count
        plot(controls{control_index}.time_s,controls{control_index}.rate_hz(:,1), ...
            'LineWidth',1.0,'DisplayName',control_label{control_index});
    end
    hold off; grid on;
    xlabel('Time (s)'); ylabel('Population rate (Hz)');
    title(sprintf('Factorial mechanism controls at recurrent gain %.4g', ...
        recurrent_gains(example_index)));
    legend('Location','eastoutside');
end

%% Selected-gain controls without adaptation and with positive bias
% These tests determine whether a persistent regime exists elsewhere in the
% gain sweep but is hidden by adaptation or exact-rheobase operation.
if run_selected_gain_controls
    selected_count = numel(selected_control_gains);
    gain_controls = cell(2*selected_count,1);
    control_gain = zeros(2*selected_count,1);
    control_type = strings(2*selected_count,1);
    for gain_index = 1:selected_count
        row = 2*gain_index-1;
        control_gain(row:row+1) = double(selected_control_gains(gain_index));
        control_type(row) = "No adaptation";
        control_type(row+1) = "Bias increment";
        gain_controls{row} = simulate_condition(Punit,unit_self_coupling, ...
            selected_control_gains(gain_index),single(0),single(0),settings);
        gain_controls{row+1} = simulate_condition(Punit,unit_self_coupling, ...
            selected_control_gains(gain_index),single(cfg.adaptation_jump), ...
            control_bias_increment_mV,settings);
    end
    selected_gain_results = condition_table(gain_controls);
    selected_gain_results = addvars(selected_gain_results,control_gain,control_type, ...
        'Before',1,'NewVariableNames',{'RecurrentGain','Control'});
    fprintf('\nSelected-gain mechanism controls\n');
    disp(selected_gain_results);

    figure('Color','w');
    tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
    nexttile; hold on;
    for gain_index = 1:selected_count
        plot(gain_controls{2*gain_index-1}.time_s, ...
            gain_controls{2*gain_index-1}.rate_hz(:,1),'LineWidth',1.0, ...
            'DisplayName',sprintf('g=%.3g',selected_control_gains(gain_index)));
    end
    hold off; grid on; xlabel('Time (s)'); ylabel('Population rate (Hz)');
    title('No adaptation jump'); legend('Location','best');
    nexttile; hold on;
    for gain_index = 1:selected_count
        plot(gain_controls{2*gain_index}.time_s, ...
            gain_controls{2*gain_index}.rate_hz(:,1),'LineWidth',1.0, ...
            'DisplayName',sprintf('g=%.3g',selected_control_gains(gain_index)));
    end
    hold off; grid on; xlabel('Time (s)'); ylabel('Population rate (Hz)');
    title(sprintf('Bias increment %+.3g mV',control_bias_increment_mV));
    legend('Location','best');
    sgtitle('Does a persistent regime exist at another recurrent gain?');
end

%% Initialization-density controls
% A uniform distribution around threshold imposes an unusually dense 50%%
% first volley. Sparse fractions test whether that synchronization itself
% drives reset/adaptation and masks a longer branching-like cascade.
if run_initial_fraction_controls
    fraction_count = numel(initial_spiking_fractions);
    fraction_controls = cell(fraction_count,1);
    for fraction_index = 1:fraction_count
        fraction_settings = settings;
        fraction_settings.initialization_mode = "specified_fraction";
        fraction_settings.initial_spiking_fraction = ...
            initial_spiking_fractions(fraction_index);
        fraction_controls{fraction_index} = simulate_condition( ...
            Punit,unit_self_coupling,recurrent_gains(example_index), ...
            single(cfg.adaptation_jump),single(0),fraction_settings);
    end
    fraction_results = condition_table(fraction_controls);
    fraction_results = addvars(fraction_results,initial_spiking_fractions(:), ...
        'Before',1,'NewVariableNames','RequestedInitialSpikingFraction');
    fprintf('\nInitialization-density controls at gain %.4g\n', ...
        recurrent_gains(example_index));
    disp(fraction_results);

    figure('Color','w'); hold on;
    for fraction_index = 1:fraction_count
        plot(fraction_controls{fraction_index}.time_s, ...
            fraction_controls{fraction_index}.rate_hz(:,1),'LineWidth',1.1, ...
            'DisplayName',sprintf('initial %.3g%%', ...
            100*initial_spiking_fractions(fraction_index)));
    end
    hold off; grid on; xlim([0 min(simulation_time_s,0.75)]);
    xlabel('Time (s)'); ylabel('Population rate (Hz)');
    title(sprintf('Initial-volley density at recurrent gain %.4g', ...
        recurrent_gains(example_index)));
    legend('Location','best');
end

%% Local simulation and measurement functions
function S = simulate_condition(Punit,unitSelfCoupling,gain,adaptationJump, ...
        biasIncrement,settings)
P = Punit;
P.recurrentGain = single(gain);
P.self_coupling = single(gain).*unitSelfCoupling;
P.adaptationJump = single(adaptationJump);
P.B = P.B+single(biasIncrement);
P = banff_model('gpu',P);

rng(settings.random_seed,'twister');
neuron_count = P.N_hidden;
if settings.initialization_mode=="uniform_around_threshold"
    base_voltage = settings.threshold_voltage + ...
        settings.initial_voltage_half_width_mV.*single(2*rand(neuron_count,1)-1);
elseif settings.initialization_mode=="specified_fraction"
    % Start all neurons below threshold, then place an exact randomly chosen
    % fraction above it. Distances remain randomized within the same voltage
    % band used by the dense initialization.
    base_voltage = settings.threshold_voltage- ...
        settings.initial_voltage_half_width_mV.*single(0.1+0.9*rand(neuron_count,1));
    initially_spiking = max(1,min(neuron_count-1, ...
        round(settings.initial_spiking_fraction*neuron_count)));
    selected = randperm(neuron_count,initially_spiking);
    base_voltage(selected) = settings.threshold_voltage + ...
        settings.initial_voltage_half_width_mV.* ...
        single(0.1+0.9*rand(initially_spiking,1));
else
    error('banff:edgeInitializationMode','Unknown initialization mode.');
end
perturbation = settings.trajectory_perturbation_mV.*single(randn(neuron_count,1));
state = make_diagnostic_state(base_voltage,base_voltage+perturbation);
% Keep the constant zero input resident on the GPU rather than transferring
% it afresh during every diagnostic timestep.
input_current = gpuArray.zeros(neuron_count,2,'single');

rate_gpu = gpuArray.zeros(settings.step_count,2,'single');
disagreement_gpu = gpuArray.zeros(settings.step_count,1,'single');
sample_count = numel(settings.sample_steps);
net_recurrent_gpu = gpuArray.zeros(sample_count,1,'single');
gross_recurrent_gpu = gpuArray.zeros(sample_count,1,'single');
adaptation_gpu = gpuArray.zeros(sample_count,1,'single');
filtered_spike_gpu = gpuArray.zeros(sample_count,1,'single');
separation_gpu = gpuArray.zeros(sample_count,1,'single');
active_tail_gpu = gpuArray.false(neuron_count,1);
sample_index = 0;

for step = 1:settings.step_count
    if mod(step-1,settings.diagnostic_stride)==0
        sample_index = sample_index+1;
        [net_recurrent,gross_recurrent] = recurrent_components(P,state.r);
        net_recurrent_gpu(sample_index) = mean(abs(net_recurrent(:,1)),1);
        gross_recurrent_gpu(sample_index) = mean(gross_recurrent(:,1),1);
        adaptation_gpu(sample_index) = mean(abs(state.w(:,1)),1);
        filtered_spike_gpu(sample_index) = mean(state.r(:,1),1);
        % Currents and separation are all sampled from the same pre-update
        % state and therefore share the timestamp (step-1)*dt. In particular,
        % the first value is the deliberately imposed initial perturbation.
        du = state.u(:,2)-state.u(:,1);
        separation_gpu(sample_index) = sqrt(mean(du.*du,1));
    end

    [state,spike] = banff_model('gpu_step',P,state,input_current,false);
    rate_gpu(step,:) = sum(single(spike),1)./single(neuron_count*settings.dt);
    disagreement_gpu(step) = mean(single(xor(spike(:,1),spike(:,2))),1);
    if step>=settings.tail_start
        active_tail_gpu = active_tail_gpu|spike(:,1);
    end
end

S = struct();
S.time_s = (1:settings.step_count).'*settings.dt;
S.diagnostic_time_s = (double(settings.sample_steps(:))-1).*settings.dt;
S.rate_hz = double(gather(rate_gpu));
S.spike_disagreement = double(gather(disagreement_gpu));
S.net_recurrent_mV = double(gather(net_recurrent_gpu));
S.gross_recurrent_mV = double(gather(gross_recurrent_gpu));
S.adaptation_mV = double(gather(adaptation_gpu));
S.filtered_spike_state = double(gather(filtered_spike_gpu));
S.voltage_separation_mV = double(gather(separation_gpu));

spike_count = round(S.rate_hz(:,1).*neuron_count.*settings.dt);
S.initial_spiking_percent = 100*spike_count(1)/neuron_count;
S.subsequent_spikes_per_neuron = sum(spike_count(2:end))/neuron_count;
S.mean_rate_first_50ms_hz = window_mean(S.rate_hz(:,1),0.050,settings.dt);
S.mean_rate_first_250ms_hz = window_mean(S.rate_hz(:,1),0.250,settings.dt);
S.peak_rate_hz = max(S.rate_hz(:,1));
last_step = find(spike_count>0,1,'last');
if isempty(last_step), last_step = 0; end
S.last_spike_time_s = last_step*settings.dt;
S.extinguished = last_step<=settings.step_count-settings.quiet_steps;
S.tail_rate_hz = mean(S.rate_hz(settings.tail_start:end,1));
S.tail_active_neuron_percent = 100*mean(gather(active_tail_gpu));
S.peak_net_recurrent_mV = max(S.net_recurrent_mV);
S.peak_gross_recurrent_mV = max(S.gross_recurrent_mV);
S.peak_adaptation_mV = max(S.adaptation_mV);
gross_scale = max(S.gross_recurrent_mV);
valid_gross = isfinite(S.gross_recurrent_mV) & gross_scale>0 & ...
    S.gross_recurrent_mV>gross_scale*1e-6;
if any(valid_gross)
    S.mean_recurrent_cancellation_ratio = mean( ...
        S.net_recurrent_mV(valid_gross)./S.gross_recurrent_mV(valid_gross));
else
    S.mean_recurrent_cancellation_ratio = NaN;
end
relative_separation = S.voltage_separation_mV./ ...
    max(S.voltage_separation_mV(1),realmin);
% Stop the fit when the actual cascade ends. Including hundreds of
% milliseconds of subsequent silent relaxation would force a negative slope
% even if the active cascade initially separated nearby trajectories.
active_fit_end_s = min(0.5,max(S.last_spike_time_s, ...
    3*settings.diagnostic_stride*settings.dt));
early = S.diagnostic_time_s<=active_fit_end_s & isfinite(relative_separation) & ...
    relative_separation>0;
if nnz(early)>=3
    coefficient = polyfit(S.diagnostic_time_s(early),log(relative_separation(early)),1);
    S.early_log_voltage_separation_slope = coefficient(1);
else
    S.early_log_voltage_separation_slope = NaN;
end
S.separation_fit_end_s = active_fit_end_s;
numeric_values = [S.rate_hz(:);S.net_recurrent_mV;S.gross_recurrent_mV; ...
    S.adaptation_mV;S.filtered_spike_state;S.voltage_separation_mV];
S.numerically_finite = all(isfinite(numeric_values));
end

function output = settings_with_duration(input,duration)
output = input;
output.step_count = round(duration/output.dt);
output.sample_steps = 1:output.diagnostic_stride:output.step_count;
output.tail_start = max(1,round(0.75*output.step_count));
end

function T = condition_table(conditions)
tail_rate_hz = cellfun(@(S)S.tail_rate_hz,conditions);
tail_active_neuron_percent = cellfun(@(S)S.tail_active_neuron_percent,conditions);
last_spike_time_s = cellfun(@(S)S.last_spike_time_s,conditions);
extinguished = cellfun(@(S)S.extinguished,conditions);
subsequent_spikes_per_neuron = ...
    cellfun(@(S)S.subsequent_spikes_per_neuron,conditions);
active_cascade_separation_slope = ...
    cellfun(@(S)S.early_log_voltage_separation_slope,conditions);
separation_fit_end_s = cellfun(@(S)S.separation_fit_end_s,conditions);
numerically_finite = cellfun(@(S)S.numerically_finite,conditions);
T = table(tail_rate_hz,tail_active_neuron_percent,last_spike_time_s, ...
    extinguished,subsequent_spikes_per_neuron, ...
    active_cascade_separation_slope,separation_fit_end_s,numerically_finite, ...
    'VariableNames',{'TailRateHz','TailActiveNeuronPercent', ...
    'LastSpikeTimeSeconds','Extinguished','SubsequentSpikesPerNeuron', ...
    'ActiveCascadeLogSeparationSlopePerSecond','SeparationFitEndSeconds', ...
    'NumericallyFinite'});
end

function [radius,eigenvalues,flag] = recurrent_operator_spectrum(P,count)
% Estimate the spectrum of the exact unit-gain current operator
% A=E*F-diag(self) without materializing its N-by-N dense matrix. This is a
% useful scale diagnostic, but it is not the Jacobian of the complete spiking,
% adaptive and synaptically filtered dynamical system.
count = max(1,min(P.N_hidden-2,round(count)));
options = struct('isreal',true,'issym',false,'tol',1e-5,'maxit',500, ...
    'disp',0,'v0',ones(P.N_hidden,1)/sqrt(P.N_hidden));
try
    [~,D,flag] = eigs(@(x)apply_recurrent_operator(P,x),P.N_hidden, ...
        count,'largestabs',options);
    eigenvalues = diag(D);
    radius = max(abs(eigenvalues));
catch exception
    warning('banff:edgeSpectrum', ...
        'Recurrent spectrum estimation was skipped: %s',exception.message);
    eigenvalues = complex(nan(count,1));
    radius = NaN;
    flag = -1;
end
end

function output = apply_recurrent_operator(P,input)
input_single = single(input);
output = double(P.recurrent_expansion*(P.W_feedback*input_single) ...
    -P.self_coupling.*input_single);
end

function [netCurrent,grossCurrent] = recurrent_components(P,filteredSpikes)
% For the Dale-signed low-rank operator, gross incoming magnitude has an
% exact factorized form. The subtracted diagonal term reflects the explicitly
% omitted self-connection and prevents it being counted as an afferent.
latent = P.W_feedback*filteredSpikes;
netCurrent = P.recurrentGain.*(P.recurrent_expansion*latent) ...
    -P.self_coupling.*filteredSpikes;
grossCurrent = P.recurrentGain.*(P.recurrent_expansion* ...
    (abs(P.W_feedback)*filteredSpikes))-abs(P.self_coupling).*filteredSpikes;
grossCurrent = max(grossCurrent,single(0));
end

function value = window_mean(trace,duration,dt)
step_count = min(numel(trace),max(1,round(duration/dt)));
value = mean(trace(1:step_count));
end

function state = make_diagnostic_state(voltageA,voltageB)
voltage = [voltageA voltageB];
[neuron_count,trajectory_count] = size(voltage);
state = struct();
state.u = voltage;
state.w = zeros(neuron_count,trajectory_count,'single');
state.x = zeros(neuron_count,trajectory_count,'single');
state.r = zeros(neuron_count,trajectory_count,'single');
state.epsilonVoltage = zeros(neuron_count,trajectory_count,'single');
state.epsilonAdaptation = zeros(neuron_count,trajectory_count,'single');
state.eligibilityRise = zeros(neuron_count,trajectory_count,'single');
state.eligibilityDecay = zeros(neuron_count,trajectory_count,'single');
end
