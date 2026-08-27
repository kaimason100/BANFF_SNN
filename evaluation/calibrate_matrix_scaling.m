%% Training-data-only calibration of encoder, recurrent and decoder scaling
% This evaluation-only script recommends gains for the fixed BANFF operators.
% It never trains a model and never accesses the held-out test split. Analytic
% fan-in factors provide width/dimension invariance; empirical gain calibration
% then matches actual task-driven currents and output scale.
%
% The selected numerical targets are modelling choices, not biological laws.
% Report them explicitly and perform sensitivity analyses around them.

clearvars;
close all;
clc;
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);

%% Editable calibration choices
task = 'breast_cancer';
% Leave empty to select the publication architecture default: 32,000 for
% low-rank calibration or 6,000 for the sparse full-rank control. A deliberate
% numeric override is still accepted.
N_hidden = [];
N_recurrent = 10;
network_seed = 1;
recurrent_mode = 'low_rank'; % also supports 'full_rank'

if isempty(N_hidden)
    if string(recurrent_mode)=="full_rank"
        N_hidden = 6000;
    else
        N_hidden = 32000;
    end
end
if string(recurrent_mode)=="full_rank" && N_hidden>6000
    estimated_connections = double(N_hidden)^2*0.10;
    warning('banff:scalingCalibrationFullRankSize', ...
        ['Full-rank calibration at N=%d and the default 10%% connectivity ', ...
        'constructs approximately %.3g recurrent connections. This may ', ...
        'exhaust host or GPU memory.'],N_hidden,estimated_connections);
end

% Current targets are expressed in the mV-equivalent current convention used
% by BANFF's membrane equation. A 2 mV encoder RMS is 10% of the 20 mV
% resting-to-threshold interval. Recurrent target is relative to encoder RMS.
target_encoder_rms_mV = 2.0;
target_recurrent_to_encoder_rms = 0.50;
target_unit_output_sd = 0.25; % normalized-logit/target units

calibration_training_samples = 256;
calibration_batch_size = 64;
dynamics_calibration_time_s = 2.0;
dynamics_warmup_time_s = 0.5;
recurrent_calibration_iterations = 4;
recurrent_update_damping = 0.50;

% Diagnostic physiological envelopes. These are warnings, not optimization
% objectives, and should be changed if the study declares another regime.
preferred_mean_rate_hz = [1 20];
preferred_active_neuron_percent = [10 100];

if ~canUseGPU
    error('banff:scalingCalibrationGPU', ...
        'Matrix-scaling calibration requires the production MATLAB GPU step.');
end
numeric_settings = [N_hidden N_recurrent network_seed target_encoder_rms_mV ...
    target_recurrent_to_encoder_rms target_unit_output_sd ...
    calibration_training_samples calibration_batch_size ...
    dynamics_calibration_time_s dynamics_warmup_time_s ...
    recurrent_calibration_iterations recurrent_update_damping];
if any(~isfinite(numeric_settings)) || N_hidden<1 || N_recurrent<1 || ...
        target_encoder_rms_mV<=0 || target_recurrent_to_encoder_rms<=0 || ...
        target_unit_output_sd<=0 || calibration_training_samples<1 || ...
        calibration_batch_size<1 || dynamics_calibration_time_s<=0 || ...
        dynamics_warmup_time_s<0 || recurrent_calibration_iterations<1 || ...
        recurrent_update_damping<=0 || recurrent_update_damping>1
    error('banff:scalingCalibrationSettings','Calibration settings are invalid.');
end

%% Load only training data and construct unit-gain operators
base_cfg = banff('config',task,struct('N_hidden',N_hidden, ...
    'N_recurrent',N_recurrent,'seed',network_seed, ...
    'recurrent_mode',recurrent_mode));
unit_cfg = banff('config',task,struct('N_hidden',N_hidden, ...
    'N_recurrent',N_recurrent,'seed',network_seed, ...
    'recurrent_mode',recurrent_mode, ...
    'encoder_gain',single(1),'recurrent_gain',single(1), ...
    'decoder_gain',single(1),'batch_size',calibration_batch_size));

if unit_cfg.kind=="dynamics"
    [pool,~] = banff_data('dynamics',unit_cfg);
    transition_count = min(size(pool.states,2)-1, ...
        round(dynamics_calibration_time_s/double(unit_cfg.dt)));
    calibration_data = struct('kind',"dynamics", ...
        'sequence',pool.states(:,1:transition_count+1));
    n_input = size(pool.states,1);
    n_output = n_input;
else
    [data,~] = banff_data('static',unit_cfg);
    sample_count = min(size(data.X_train,2),round(calibration_training_samples));
    sample_index = unique(round(linspace(1,size(data.X_train,2),sample_count)));
    calibration_data = struct('kind',"static", ...
        'X',data.X_train(:,sample_index));
    n_input = size(data.X_train,1);
    n_output = size(data.Y_train,1);
end
Punit = banff_model('create',n_input,n_output,unit_cfg);
if Punit.recurrent_mode=="low_rank"
    feedback_width_factor = 1/sqrt(N_hidden);
    expansion_rank_factor = 1/sqrt(N_recurrent);
    full_rank_fanin_factor = NaN;
else
    mean_fanin = max(1,nnz(Punit.W_recurrent)/N_hidden);
    feedback_width_factor = NaN;
    expansion_rank_factor = NaN;
    full_rank_fanin_factor = 1/sqrt(mean_fanin);
end

fprintf('\nBANFF matrix-scaling calibration: %s\n',task);
fprintf('Training data only | N=%d | rank=%d | input D=%d | output K=%d\n', ...
    N_hidden,N_recurrent,n_input,n_output);
fprintf('Analytic factors: input 1/sqrt(D)=%.8g, decoder 1/sqrt(N)=%.8g\n', ...
    1/sqrt(n_input),1/sqrt(N_hidden));
if Punit.recurrent_mode=="low_rank"
    fprintf('Low-rank factors: feedback RMS 1/sqrt(N)=%.8g, expansion 1/sqrt(rank)=%.8g\n\n', ...
        feedback_width_factor,expansion_rank_factor);
else
    fprintf('Full-rank factor: 1/sqrt(mean fan-in)=%.8g\n\n', ...
        full_rank_fanin_factor);
end

%% 1. Encoder calibration: direct and independent of network activity
unit_encoder_rms = direct_encoder_rms(Punit,calibration_data,unit_cfg);
recommended_encoder_gain = target_encoder_rms_mV/unit_encoder_rms;
fprintf('Unit encoder RMS: %.6g mV -> recommended encoder gain %.6g\n', ...
    unit_encoder_rms,recommended_encoder_gain);

%% 2. Recurrent calibration: iterate because activity depends on recurrence
recommended_recurrent_gain = double(base_cfg.recurrent_gain);
target_recurrent_rms_mV = ...
    target_recurrent_to_encoder_rms*target_encoder_rms_mV;
iteration = (1:recurrent_calibration_iterations).';
gain_used = nan(size(iteration));
unit_recurrent_rms = nan(size(iteration));
raw_recommendation = nan(size(iteration));
mean_rate_hz = nan(size(iteration));
active_neuron_percent = nan(size(iteration));
for index = 1:recurrent_calibration_iterations
    gain_used(index) = recommended_recurrent_gain;
    probe = run_task_probe(Punit,calibration_data,unit_cfg, ...
        recommended_encoder_gain,recommended_recurrent_gain, ...
        dynamics_warmup_time_s);
    unit_recurrent_rms(index) = probe.unit_recurrent_rms_mV;
    mean_rate_hz(index) = probe.mean_rate_hz;
    active_neuron_percent(index) = probe.active_neuron_percent;
    if ~isfinite(probe.unit_recurrent_rms_mV) || probe.unit_recurrent_rms_mV<=0
        error('banff:scalingNoRecurrence', ...
            ['Task-driven activity produced no measurable recurrent state. ', ...
             'Increase the encoder target or bias before calibrating recurrence.']);
    end
    raw_recommendation(index) = ...
        target_recurrent_rms_mV/probe.unit_recurrent_rms_mV;
    % Geometric damping respects the multiplicative role of a gain and avoids
    % oscillation when changing recurrence also changes firing statistics.
    recommended_recurrent_gain = exp( ...
        (1-recurrent_update_damping)*log(max(recommended_recurrent_gain,realmin)) + ...
        recurrent_update_damping*log(max(raw_recommendation(index),realmin)));
end
iteration_table = table(iteration,gain_used,unit_recurrent_rms, ...
    raw_recommendation,mean_rate_hz,active_neuron_percent, ...
    'VariableNames',{'Iteration','GainUsed','UnitGainRecurrentRmsMv', ...
    'RawRecommendedGain','MeanRateHz','ActiveNeuronPercent'});
disp(iteration_table);

%% 3. Final activity probe and decoder calibration
final_probe = run_task_probe(Punit,calibration_data,unit_cfg, ...
    recommended_encoder_gain,recommended_recurrent_gain, ...
    dynamics_warmup_time_s);
if ~isfinite(final_probe.unit_decoder_output_sd) || ...
        final_probe.unit_decoder_output_sd<=0
    error('banff:scalingNoDecoderState', ...
        'The calibration run produced no variable decoder state.');
end
recommended_decoder_gain = ...
    target_unit_output_sd/final_probe.unit_decoder_output_sd;

actual_encoder_rms_mV = final_probe.encoder_rms_mV;
actual_recurrent_rms_mV = recommended_recurrent_gain* ...
    final_probe.unit_recurrent_rms_mV;
actual_decoder_output_sd = recommended_decoder_gain* ...
    final_probe.unit_decoder_output_sd;

recommendations = table( ...
    recommended_encoder_gain,recommended_recurrent_gain,recommended_decoder_gain, ...
    1/sqrt(n_input),1/sqrt(N_hidden),feedback_width_factor, ...
    expansion_rank_factor,full_rank_fanin_factor, ...
    'VariableNames',{'EncoderGain','RecurrentGain','DecoderGain', ...
    'InputDimensionFactor','DecoderWidthFactor', ...
    'FeedbackWidthFactor','ExpansionRankFactor','FullRankFaninFactor'});
fprintf('\nRecommended scaling\n');
disp(recommendations);

achieved = table(actual_encoder_rms_mV,actual_recurrent_rms_mV, ...
    actual_recurrent_rms_mV/actual_encoder_rms_mV, ...
    final_probe.gross_recurrent_rms_mV*recommended_recurrent_gain, ...
    final_probe.recurrent_net_to_gross_rms_ratio, ...
    final_probe.adaptation_rms_mV,final_probe.mean_rate_hz, ...
    final_probe.active_neuron_percent,actual_decoder_output_sd, ...
    'VariableNames',{'EncoderRmsMv','NetRecurrentRmsMv', ...
    'RecurrentToEncoderRms','GrossRecurrentRmsMv', ...
    'NetToGrossRecurrentRms','AdaptationRmsMv','MeanRateHz', ...
    'ActiveNeuronPercent','DecoderOutputSd'});
fprintf('Achieved task-driven regime\n');
disp(achieved);

relative_recurrent_target_error = abs(actual_recurrent_rms_mV- ...
    target_recurrent_rms_mV)/target_recurrent_rms_mV;
if relative_recurrent_target_error>0.10
    warning('banff:scalingIterationConvergence', ...
        ['Recurrent RMS remains %.2f%% from its target. Increase ', ...
         'recurrent_calibration_iterations before adopting the gain.'], ...
        100*relative_recurrent_target_error);
end

if final_probe.mean_rate_hz<preferred_mean_rate_hz(1) || ...
        final_probe.mean_rate_hz>preferred_mean_rate_hz(2)
    warning('banff:scalingRateEnvelope', ...
        'Mean firing rate %.4g Hz is outside the declared [%g,%g] Hz envelope.', ...
        final_probe.mean_rate_hz,preferred_mean_rate_hz(1),preferred_mean_rate_hz(2));
end
if final_probe.active_neuron_percent<preferred_active_neuron_percent(1) || ...
        final_probe.active_neuron_percent>preferred_active_neuron_percent(2)
    warning('banff:scalingActivityEnvelope', ...
        'Active-neuron fraction %.4g%% is outside the declared [%g,%g]%% envelope.', ...
        final_probe.active_neuron_percent,preferred_active_neuron_percent(1), ...
        preferred_active_neuron_percent(2));
end

%% Sensitivity recommendations around the declared targets
encoder_target_grid_mV = [1 2 4].';
recurrent_ratio_grid = [0.25 0.50 1.00].';
decoder_target_grid = [0.10 0.25 0.50].';
sensitivity = table(encoder_target_grid_mV, ...
    encoder_target_grid_mV/unit_encoder_rms, ...
    recurrent_ratio_grid, ...
    recurrent_ratio_grid.*encoder_target_grid_mV/ ...
        final_probe.unit_recurrent_rms_mV, ...
    decoder_target_grid,decoder_target_grid/final_probe.unit_decoder_output_sd, ...
    'VariableNames',{'EncoderTargetRmsMv','EncoderGain', ...
    'RecurrentToEncoderTarget','RecurrentGain', ...
    'DecoderTargetSd','DecoderGain'});
fprintf('Conservative / central / strong sensitivity settings\n');
disp(sensitivity);

figure('Color','w');
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
nexttile;
bar([actual_encoder_rms_mV actual_recurrent_rms_mV ...
    final_probe.adaptation_rms_mV]);
set(gca,'XTickLabel',{'Encoder','Net recurrence','Adaptation'});
ylabel('RMS contribution (mV)'); grid on;
title('Recommended task-driven current scale');
nexttile;
bar([recommended_encoder_gain recommended_recurrent_gain recommended_decoder_gain]);
set(gca,'XTickLabel',{'Encoder','Recurrent','Decoder'});
ylabel('Dimensionless gain'); grid on; set(gca,'YScale','log');
title('Recommended operator gains');

fprintf(['\nUse these values as scientific overrides only after checking multiple ', ...
    'network seeds and validation performance. Do not select them using test results.\n']);

%% Local calibration functions
function rmsValue = direct_encoder_rms(P,data,cfg)
if data.kind=="static"
    X = data.X;
else
    X = data.sequence(:,1:end-1);
end
P = banff_model('gpu',P);
square_sum = gpuArray.zeros(1,1,'single');
value_count = 0;
batch_size = double(cfg.batch_size);
for first = 1:batch_size:size(X,2)
    indices = first:min(size(X,2),first+batch_size-1);
    current = P.W_in*(P.inputScale.*gpuArray(single(X(:,indices))));
    square_sum = square_sum+sum(current.*current,'all');
    value_count = value_count+numel(current);
end
rmsValue = sqrt(double(gather(square_sum))/value_count);
end

function probe = run_task_probe(Punit,data,cfg,encoderGain,recurrentGain,warmupTime)
P = Punit;
P.W_in = single(encoderGain).*P.W_in;
unit_recurrent = struct('mode',string(P.recurrent_mode));
if P.recurrent_mode=="low_rank"
    unit_recurrent.self_coupling = P.self_coupling;
    P.recurrentGain = single(recurrentGain);
    P.self_coupling = single(recurrentGain).*unit_recurrent.self_coupling;
else
    unit_recurrent.W_recurrent = P.W_recurrent;
    P.W_recurrent = double(recurrentGain).*P.W_recurrent;
end
P = banff_model('gpu',P);
if unit_recurrent.mode=="low_rank"
    unit_recurrent.self_coupling = ...
        gpuArray(single(unit_recurrent.self_coupling));
else
    unit_recurrent.W_recurrent = gpuArray(unit_recurrent.W_recurrent);
end
if data.kind=="static"
    probe = probe_static(P,unit_recurrent,data.X,cfg);
else
    warmup_steps = round(warmupTime/double(cfg.dt));
    probe = probe_dynamics(P,unit_recurrent,data.sequence,cfg,warmup_steps);
end
end

function probe = probe_static(P,unitRecurrent,X,cfg)
encoder_square = gpuArray.zeros(1,1,'single');
unit_recurrent_square = gpuArray.zeros(1,1,'single');
gross_recurrent_square = gpuArray.zeros(1,1,'single');
adaptation_square = gpuArray.zeros(1,1,'single');
spike_total = gpuArray.zeros(1,1,'single');
active = gpuArray.false(P.N_hidden,1);
output_sum = zeros(P.N_output,1);
output_square = zeros(P.N_output,1);
output_samples = 0;
observation_count = 0;

for first = 1:double(cfg.batch_size):size(X,2)
    indices = first:min(size(X,2),first+double(cfg.batch_size)-1);
    local_batch = numel(indices);
    input_current = P.W_in*(P.inputScale.*gpuArray(single(X(:,indices))));
    state = empty_state(P,local_batch);
    decoder_state_sum = gpuArray.zeros(P.N_hidden,local_batch,'single');
    for step = 1:P.presentationSteps
        [unit_recurrent,gross_recurrent] = ...
            unit_recurrent_components(P,unitRecurrent,state.r);
        encoder_square = encoder_square+sum(input_current.*input_current,'all');
        unit_recurrent_square = unit_recurrent_square+ ...
            sum(unit_recurrent.*unit_recurrent,'all');
        gross_recurrent_square = gross_recurrent_square+ ...
            sum(gross_recurrent.*gross_recurrent,'all');
        adaptation_square = adaptation_square+sum(state.w.*state.w,'all');
        [state,spike] = banff_model('gpu_step',P,state,input_current,false);
        spike_total = spike_total+sum(single(spike),'all');
        active = active|any(spike,2);
        if step>=P.averageStartStep
            decoder_state_sum = decoder_state_sum+state.r;
        end
    end
    unit_output = gather(P.W_out*(decoder_state_sum/single(P.averageSteps)));
    output_sum = output_sum+sum(double(unit_output),2);
    output_square = output_square+sum(double(unit_output).^2,2);
    output_samples = output_samples+local_batch;
    observation_count = observation_count+P.N_hidden*local_batch*P.presentationSteps;
end
duration = double(P.presentationSteps)*double(P.dt);
probe = finish_probe(encoder_square,unit_recurrent_square, ...
    gross_recurrent_square,adaptation_square,observation_count, ...
    spike_total,active,size(X,2)*duration,output_sum,output_square,output_samples);
end

function probe = probe_dynamics(P,unitRecurrent,sequence,cfg,warmupSteps)
transition_count = size(sequence,2)-1;
warmupSteps = min(max(0,warmupSteps),max(0,transition_count-1));
state = empty_state(P,1);
encoder_square = gpuArray.zeros(1,1,'single');
unit_recurrent_square = gpuArray.zeros(1,1,'single');
gross_recurrent_square = gpuArray.zeros(1,1,'single');
adaptation_square = gpuArray.zeros(1,1,'single');
spike_total = gpuArray.zeros(1,1,'single');
active = gpuArray.false(P.N_hidden,1);
output_sum = zeros(P.N_output,1);
output_square = zeros(P.N_output,1);
output_samples = 0;
observation_count = 0;
for step = 1:transition_count
    input_current = P.W_in*(P.inputScale.*gpuArray(single(sequence(:,step))));
    [unit_recurrent,gross_recurrent] = ...
        unit_recurrent_components(P,unitRecurrent,state.r);
    adaptation_before_step = state.w;
    [state,spike] = banff_model('gpu_step',P,state,input_current,false);
    if step>warmupSteps
        encoder_square = encoder_square+sum(input_current.*input_current,'all');
        unit_recurrent_square = unit_recurrent_square+ ...
            sum(unit_recurrent.*unit_recurrent,'all');
        gross_recurrent_square = gross_recurrent_square+ ...
            sum(gross_recurrent.*gross_recurrent,'all');
        adaptation_square = adaptation_square+ ...
            sum(adaptation_before_step.*adaptation_before_step,'all');
        spike_total = spike_total+sum(single(spike),'all');
        active = active|spike;
        unit_output = gather(P.W_out*state.r);
        output_sum = output_sum+double(unit_output);
        output_square = output_square+double(unit_output).^2;
        output_samples = output_samples+1;
        observation_count = observation_count+P.N_hidden;
    end
end
scored_duration = output_samples*double(P.dt);
probe = finish_probe(encoder_square,unit_recurrent_square, ...
    gross_recurrent_square,adaptation_square,observation_count, ...
    spike_total,active,scored_duration,output_sum,output_square,output_samples);
end

function probe = finish_probe(encoderSquare,recurrentSquare,grossSquare, ...
        adaptationSquare,observationCount,spikeTotal,active,totalDuration, ...
        outputSum,outputSquare,outputSamples)
probe.encoder_rms_mV = sqrt(double(gather(encoderSquare))/observationCount);
probe.unit_recurrent_rms_mV = ...
    sqrt(double(gather(recurrentSquare))/observationCount);
probe.gross_recurrent_rms_mV = ...
    sqrt(double(gather(grossSquare))/observationCount);
probe.adaptation_rms_mV = ...
    sqrt(double(gather(adaptationSquare))/observationCount);
probe.mean_rate_hz = double(gather(spikeTotal))/numel(active)/totalDuration;
probe.active_neuron_percent = 100*mean(gather(active));
variance = max(0,outputSquare/max(1,outputSamples) ...
    -(outputSum/max(1,outputSamples)).^2);
probe.unit_decoder_output_sd = sqrt(mean(variance));
probe.recurrent_net_to_gross_rms_ratio = probe.unit_recurrent_rms_mV/ ...
    max(probe.gross_recurrent_rms_mV,realmin);
end

function [netCurrent,grossCurrent] = ...
        unit_recurrent_components(P,unitRecurrent,filteredSpikes)
if unitRecurrent.mode=="full_rank"
    netCurrent = single(unitRecurrent.W_recurrent*double(filteredSpikes));
    grossCurrent = single(abs(unitRecurrent.W_recurrent)*double(filteredSpikes));
else
    latent = P.W_feedback*filteredSpikes;
    netCurrent = P.recurrent_expansion*latent ...
        -unitRecurrent.self_coupling.*filteredSpikes;
    grossCurrent = P.recurrent_expansion* ...
        (abs(P.W_feedback)*filteredSpikes) ...
        -abs(unitRecurrent.self_coupling).*filteredSpikes;
end
grossCurrent = max(grossCurrent,single(0));
end

function state = empty_state(P,batchSize)
state = struct('u',repmat(P.restingVoltage,P.N_hidden,batchSize), ...
    'w',gpuArray.zeros(P.N_hidden,batchSize,'single'), ...
    'x',gpuArray.zeros(P.N_hidden,batchSize,'single'), ...
    'r',gpuArray.zeros(P.N_hidden,batchSize,'single'), ...
    'epsilonVoltage',gpuArray.zeros(P.N_hidden,batchSize,'single'), ...
    'epsilonAdaptation',gpuArray.zeros(P.N_hidden,batchSize,'single'), ...
    'eligibilityRise',gpuArray.zeros(P.N_hidden,batchSize,'single'), ...
    'eligibilityDecay',gpuArray.zeros(P.N_hidden,batchSize,'single'));
end
