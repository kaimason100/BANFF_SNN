%% Train Lorenz from a recurrent-mode-aligned bias initialization
% This controlled experiment keeps every fixed operator and gain unchanged
% while selecting a heterogeneous initial neuronal bias vector that places
% the untrained network in a stronger recurrent-current regime.
%
% For each network seed, the script:
%   1. constructs the ordinary fixed Lorenz network;
%   2. extracts phase B of its dominant complex recurrent eigenmode;
%   3. varies only the amplitude of that bias pattern until the initial
%      signed-net recurrent/encoder RMS ratio is approximately one;
%   4. trains biases using the normal BANFF Lorenz training path;
%   5. performs final held-out testing only after training is complete; and
%   6. invokes the shared Lorenz evaluator for the complete standard plots,
%      including untrained/trained currents, gross afferents, recurrence
%      ablation, phase portraits, firing rates and full-test inverse ISIs.
%
% Bias calibration uses only supervised inputs drawn from the normalized
% Lorenz training trajectory. Ground-truth state vectors drive the encoder at
% every calibration timestep; validation and test trajectories do not
% influence the selected initialization.

clearvars;
close all;
clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(projectRoot);
addpath(fullfile(projectRoot, 'evaluation'));

%% Editable settings
networkSeeds = 1;                  % use 1:3 for a multi-seed experiment
epochs = 5000;                    % current principal DS training budget
checkpointHours = inf;             % set below the ARC wall time if required
reuseCompletedRuns = true;

% The fixed operators retain the ordinary Lorenz gains. Only the initial
% neuronal bias distribution is calibrated.
encoderGain = single(2);
recurrentGain = single(0.05);
decoderGain = single(0.1);

% B_i(0) = B_default + meanOffset + patternScale * phaseB_i.
phaseBMeanOffsetMv = single(1);
phaseBPatternScaleInitialGuessMv = 2;
phaseBPatternScaleBoundsMv = [0.25 8];
targetInitialRecurrentToEncoderRatio = 1;
biasCalibrationTolerance = 0.03;
biasCalibrationRefinementIterations = 4;

% Use several separated, fully teacher-forced training-trajectory segments.
% The initial transient of each independently reset segment is excluded.
calibrationSequenceCount = 2;
calibrationDurationSeconds = 0.50;
calibrationWarmupSeconds = 0.10;
calibrationMeanRateBoundsHz = [0.5 30];

experimentOutputDirectory = fullfile(projectRoot, 'outputs', ...
    'lorenz_initialization_experiment');

commonOverrides = struct();
commonOverrides.N_hidden = 32000;
commonOverrides.N_recurrent = 10;
commonOverrides.recurrent_mode = 'low_rank';
commonOverrides.encoder_gain = encoderGain;
commonOverrides.recurrent_gain = recurrentGain;
commonOverrides.decoder_gain = decoderGain;
commonOverrides.epochs = epochs;
commonOverrides.checkpoint_hours = checkpointHours;
commonOverrides.output_directory = experimentOutputDirectory;

if ~canUseGPU
    error('banff:lorenzBiasInitializationGPU', ...
        'This 32,000-neuron training experiment requires a supported GPU.');
end
if calibrationWarmupSeconds < 0 || ...
        calibrationDurationSeconds <= calibrationWarmupSeconds
    error('banff:lorenzBiasCalibrationTime', ...
        'Calibration duration must be positive and exceed its warmup.');
end

seedCount = numel(networkSeeds);
trainedRuns = cell(seedCount, 1);
testedRuns = cell(seedCount, 1);
calibrationTraces = cell(seedCount, 1);
selectedBiases = cell(seedCount, 1);
selectedPatternScaleMv = nan(seedCount, 1);
selectedInitialRatio = nan(seedCount, 1);
selectedInitialEncoderRmsMv = nan(seedCount, 1);
selectedInitialRecurrentRmsMv = nan(seedCount, 1);
selectedInitialGrossEncoderRmsMv = nan(seedCount, 1);
selectedInitialGrossRecurrentRmsMv = nan(seedCount, 1);
selectedInitialMeanRateHz = nan(seedCount, 1);
trainingSecondsThisInvocation = nan(seedCount, 1);

fprintf('Lorenz recurrent-mode bias-initialization experiment\n');
fprintf('Fixed encoder/recurrent/decoder gains: %.6g / %.6g / %.6g\n', ...
    encoderGain, recurrentGain, decoderGain);
fprintf('Epochs: %d | seeds: %s | target initial ratio: %.4g\n\n', ...
    epochs, mat2str(networkSeeds), targetInitialRecurrentToEncoderRatio);

%% Bias-only calibration and training
for seedIndex = 1:seedCount
    seed = networkSeeds(seedIndex);
    seedOverrides = commonOverrides;
    seedOverrides.seed = seed;
    baseCfg = banff('config', 'lorenz', seedOverrides);
    [trainingPool, dataInformation] = banff_data('dynamics', baseCfg);
    dimension = size(trainingPool.states, 1);
    Pinitial = banff_model('create', dimension, dimension, baseCfg);

    [dominantVector, dominantEigenvalue, eigsFlag] = ...
        dominant_recurrent_mode(Pinitial);
    if std(imag(dominantVector)) <= 1e-8
        error('banff:lorenzBiasInitializationRealMode', ...
            ['The dominant recurrent eigenmode is real for seed %d, so the ' ...
            'requested phase-B initialization is unavailable.'], seed);
    end
    phaseB = standardize_pattern(single(imag(dominantVector)));
    calibrationSteps = round(calibrationDurationSeconds/double(baseCfg.dt));
    trainingSequences = select_training_sequences(trainingPool.states, ...
        calibrationSequenceCount, calibrationSteps);

    [selectedBias, selectedScale, selectedSummary, trace] = ...
        calibrate_phase_bias_scale(seedOverrides, baseCfg.initial_bias, ...
        phaseB, phaseBMeanOffsetMv, phaseBPatternScaleInitialGuessMv, ...
        trainingSequences, targetInitialRecurrentToEncoderRatio, ...
        phaseBPatternScaleBoundsMv, biasCalibrationTolerance, ...
        biasCalibrationRefinementIterations, calibrationDurationSeconds, ...
        calibrationWarmupSeconds, calibrationMeanRateBoundsHz);

    selectedBiases{seedIndex} = selectedBias;
    selectedPatternScaleMv(seedIndex) = selectedScale;
    selectedInitialRatio(seedIndex) = ...
        selectedSummary.recurrent_to_encoder_rms;
    selectedInitialEncoderRmsMv(seedIndex) = selectedSummary.encoder_rms_mV;
    selectedInitialRecurrentRmsMv(seedIndex) = ...
        selectedSummary.net_recurrent_rms_mV;
    selectedInitialGrossEncoderRmsMv(seedIndex) = ...
        selectedSummary.gross_encoder_rms_mV;
    selectedInitialGrossRecurrentRmsMv(seedIndex) = ...
        selectedSummary.gross_recurrent_rms_mV;
    selectedInitialMeanRateHz(seedIndex) = selectedSummary.mean_rate_hz;
    calibrationTraces{seedIndex} = trace;

    fprintf('Seed %d dominant eigenvalue: %.6g %+.6gi (flag %d)\n', ...
        seed, real(dominantEigenvalue), imag(dominantEigenvalue), eigsFlag);
    fprintf(['Selected bias distribution: mean %.6g mV, SD %.6g mV; ' ...
        'phase-B scale %.6g mV.\n'], mean(double(selectedBias)), ...
        std(double(selectedBias)), selectedScale);
    fprintf(['Initial supervised-input currents: net encoder %.6g mV, net ' ...
        'recurrent %.6g mV, ratio %.6g; gross encoder %.6g mV, gross ' ...
        'recurrent %.6g mV; mean rate %.6g Hz.\n'], ...
        selectedSummary.encoder_rms_mV, ...
        selectedSummary.net_recurrent_rms_mV, ...
        selectedSummary.recurrent_to_encoder_rms, ...
        selectedSummary.gross_encoder_rms_mV, ...
        selectedSummary.gross_recurrent_rms_mV, ...
        selectedSummary.mean_rate_hz);
    disp(trace);

    trainingOverrides = seedOverrides;
    trainingOverrides.initial_bias = selectedBias;
    resolvedCfg = banff('config', 'lorenz', trainingOverrides);
    timer = tic;
    trainedRuns{seedIndex} = train_or_reuse( ...
        resolvedCfg, trainingOverrides, reuseCompletedRuns);
    trainingSecondsThisInvocation(seedIndex) = toc(timer);
    if ~trainedRuns{seedIndex}.complete
        fprintf(['Seed %d checkpointed before completion. Rerun this script ' ...
            'with identical settings to resume; no test data were used.\n'], seed);
        return;
    end

    % Retain the information object used during calibration for a direct
    % consistency check against the packaged training result.
    if ~isequaln(dataInformation, trainedRuns{seedIndex}.data_information)
        error('banff:lorenzBiasDataIdentity', ...
            'Training data information changed between calibration and training.');
    end
end

%% Final held-out testing only after all requested training is complete
fprintf('\nAll requested runs completed. Beginning held-out Lorenz testing.\n');
for seedIndex = 1:seedCount
    testOverrides = commonOverrides;
    testOverrides.seed = networkSeeds(seedIndex);
    testOverrides.initial_bias = selectedBiases{seedIndex};
    testedRuns{seedIndex} = banff('test', 'lorenz', testOverrides);
end

calibrationSummary = table(double(networkSeeds(:)), ...
    selectedPatternScaleMv, cellfun(@(B)mean(double(B)),selectedBiases), ...
    cellfun(@(B)std(double(B)),selectedBiases), ...
    selectedInitialEncoderRmsMv, selectedInitialRecurrentRmsMv, ...
    selectedInitialRatio, selectedInitialGrossEncoderRmsMv, ...
    selectedInitialGrossRecurrentRmsMv, selectedInitialMeanRateHz, ...
    trainingSecondsThisInvocation, ...
    'VariableNames', {'Seed','PhaseBPatternScaleMv','InitialBiasMeanMv', ...
    'InitialBiasStdMv','InitialEncoderRmsMv','InitialRecurrentRmsMv', ...
    'InitialRecurrentToEncoderRms','InitialGrossEncoderRmsMv', ...
    'InitialGrossRecurrentRmsMv','InitialMeanRateHz', ...
    'TrainingSecondsThisInvocation'});
fprintf('\nTraining-only initialization calibration summary\n');
disp(calibrationSummary);

%% Calibration figures
figure('Color', 'w');
tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
colors = lines(seedCount);

nexttile;
hold on;
for seedIndex = 1:seedCount
    trace = calibrationTraces{seedIndex};
    [scaleForPlot,order] = sort(trace.PatternScaleMv);
    plot(scaleForPlot, trace.RecurrentToEncoderRms(order), 'o-', ...
        'Color', colors(seedIndex, :), 'DisplayName', ...
        sprintf('Seed %d', networkSeeds(seedIndex)));
end
yline(targetInitialRecurrentToEncoderRatio, 'k--');
hold off; grid on;
xlabel('Phase-B bias-pattern scale (mV)');
ylabel('Net recurrent / encoder RMS');
title('Bias-only current-balance calibration');
legend('Location', 'best');

nexttile;
hold on;
for seedIndex = 1:seedCount
    trace = calibrationTraces{seedIndex};
    [scaleForPlot,order] = sort(trace.PatternScaleMv);
    plot(scaleForPlot, trace.MeanRateHz(order), 'o-', ...
        'Color', colors(seedIndex, :), 'DisplayName', ...
        sprintf('Seed %d', networkSeeds(seedIndex)));
end
hold off; grid on;
yline(calibrationMeanRateBoundsHz(1), 'k:');
yline(calibrationMeanRateBoundsHz(2), 'k:');
xlabel('Phase-B bias-pattern scale (mV)');
ylabel('Mean firing rate (Hz)');
title('Calibration operating rate');
legend('Location', 'best');

nexttile;
bar(categorical(string(networkSeeds)), ...
    [selectedInitialGrossEncoderRmsMv selectedInitialGrossRecurrentRmsMv]);
grid on;
xlabel('Network seed');
ylabel('Gross afferent RMS magnitude (mV)');
title('Selected absolute afferent drive');
legend({'Encoder gross','Recurrent gross'}, 'Location', 'best');

%% Reuse the complete standard Lorenz evaluation and plotting architecture
displayOptions = struct();
displayOptions.preloaded_results = [testedRuns{:}];
displayOptions.assessment_split = "test";
displayOptions.run_recurrent_ablation = true;
displayOptions.figure_visibility = "on";
displayOptions.save_figures = false;
banff_evaluate_task('lorenz', networkSeeds, "main", struct(), displayOptions);

if exist(experimentOutputDirectory, 'dir') ~= 7
    mkdir(experimentOutputDirectory);
end
summaryFile = fullfile(experimentOutputDirectory, ...
    'lorenz_bias_initialization_summary.mat');
save(summaryFile, 'calibrationSummary', 'calibrationTraces', ...
    'selectedBiases', '-v7.3');
fprintf('Saved initialization summary: %s\n', summaryFile);

%% Local functions
function result = train_or_reuse(cfg, overrides, reuseCompleted)
%TRAIN_OR_REUSE Reuse only a complete result with the exact configuration.
if reuseCompleted && exist(cfg.model_file, 'file') == 2
    loaded = load(cfg.model_file, 'result');
    if isfield(loaded, 'result') && isfield(loaded.result, 'complete') && ...
            loaded.result.complete && ...
            strcmpi(char(loaded.result.config.scientific_config_sha256), ...
            char(cfg.scientific_config_sha256)) && ...
            loaded.result.config.seed == cfg.seed
        fprintf('Reusing complete matching run: %s\n', cfg.model_file);
        result = loaded.result;
        return;
    end
end
result = banff('train', 'lorenz', overrides);
end

function sequences = select_training_sequences(states,count,steps)
%SELECT_TRAINING_SEQUENCES Choose separated supervised training segments.
availableStarts = size(states,2)-steps+1;
if availableStarts < 1
    error('banff:lorenzBiasCalibrationPool', ...
        'The training pool is shorter than the requested calibration segment.');
end
count = max(1,min(availableStarts,round(count)));
starts = unique(round(linspace(1,availableStarts,count)));
sequences = zeros(size(states,1),steps,numel(starts),'single');
for index = 1:numel(starts)
    sequences(:,:,index) = single(states(:, ...
        starts(index):(starts(index)+steps-1)));
end
end

function [bias, scale, summary, trace] = calibrate_phase_bias_scale( ...
        baseOverrides, baseBias, phasePattern, meanOffset, initialScale, ...
        trainingSequences, target, bounds, tolerance, refinementIterations, ...
        durationSeconds, warmupSeconds, rateBoundsHz)
%CALIBRATE_PHASE_BIAS_SCALE Search a nonlinear bias-only operating regime.
if numel(bounds) ~= 2 || any(~isfinite(bounds)) || bounds(1) < 0 || ...
        bounds(2) <= bounds(1)
    error('banff:lorenzBiasScaleBounds', ...
        'Bias-pattern scale bounds must be finite and strictly increasing.');
end
coarseScales = linspace(bounds(1), bounds(2), 7);
candidateScales = unique([coarseScales initialScale], 'sorted');
maximumEvaluations = numel(candidateScales) + 2 * refinementIterations;
scaleUsed = nan(maximumEvaluations, 1);
encoderRms = nan(maximumEvaluations, 1);
recurrentRms = nan(maximumEvaluations, 1);
ratioMeasured = nan(maximumEvaluations, 1);
grossEncoderRms = nan(maximumEvaluations, 1);
grossRecurrentRms = nan(maximumEvaluations, 1);
meanRateHz = nan(maximumEvaluations, 1);
relativeTargetError = inf(maximumEvaluations, 1);
summaries = cell(maximumEvaluations, 1);
evaluationCount = 0;

for index = 1:numel(candidateScales)
    [scaleUsed,encoderRms,recurrentRms,ratioMeasured,grossEncoderRms, ...
        grossRecurrentRms,meanRateHz,relativeTargetError,summaries, ...
        evaluationCount] = evaluate_scale(candidateScales(index), ...
        scaleUsed,encoderRms,recurrentRms,ratioMeasured,grossEncoderRms, ...
        grossRecurrentRms,meanRateHz,relativeTargetError,summaries, ...
        evaluationCount,baseOverrides,baseBias,phasePattern,meanOffset, ...
        trainingSequences,target,durationSeconds,warmupSeconds,rateBoundsHz);
end

for refinement = 1:refinementIterations
    [~, order] = sort(scaleUsed(1:evaluationCount));
    sortedScales = scaleUsed(order);
    sortedErrors = relativeTargetError(order);
    [bestError, bestPosition] = min(sortedErrors);
    if bestError <= tolerance, break; end
    newScales = zeros(0, 1);
    if bestPosition > 1
        newScales(end+1,1) = mean(sortedScales(bestPosition+[-1 0])); %#ok<AGROW>
    end
    if bestPosition < numel(sortedScales)
        newScales(end+1,1) = mean(sortedScales(bestPosition+[0 1])); %#ok<AGROW>
    end
    if isempty(newScales), break; end
    for index = 1:numel(newScales)
        [scaleUsed,encoderRms,recurrentRms,ratioMeasured,grossEncoderRms, ...
            grossRecurrentRms,meanRateHz,relativeTargetError,summaries, ...
            evaluationCount] = evaluate_scale(newScales(index), ...
            scaleUsed,encoderRms,recurrentRms,ratioMeasured,grossEncoderRms, ...
            grossRecurrentRms,meanRateHz,relativeTargetError,summaries, ...
            evaluationCount,baseOverrides,baseBias,phasePattern,meanOffset, ...
            trainingSequences,target,durationSeconds,warmupSeconds,rateBoundsHz);
    end
end

[bestError, bestIndex] = min(relativeTargetError(1:evaluationCount));
scale = scaleUsed(bestIndex);
summary = summaries{bestIndex};
bias = single(baseBias) + single(meanOffset) + ...
    single(scale) .* single(phasePattern);
trace = table((1:evaluationCount).', scaleUsed(1:evaluationCount), ...
    encoderRms(1:evaluationCount), recurrentRms(1:evaluationCount), ...
    ratioMeasured(1:evaluationCount), grossEncoderRms(1:evaluationCount), ...
    grossRecurrentRms(1:evaluationCount), meanRateHz(1:evaluationCount), ...
    meanRateHz(1:evaluationCount)>=rateBoundsHz(1) & ...
    meanRateHz(1:evaluationCount)<=rateBoundsHz(2), ...
    relativeTargetError(1:evaluationCount), ...
    'VariableNames', {'Evaluation','PatternScaleMv','EncoderRmsMv', ...
    'RecurrentRmsMv','RecurrentToEncoderRms','GrossEncoderRmsMv', ...
    'GrossRecurrentRmsMv','MeanRateHz','RateWithinBounds', ...
    'RelativeTargetError'});
if isempty(summary) || ~isfinite(bestError) || bestError > tolerance
    error('banff:lorenzBiasScaleCalibrationTarget', ...
        ['Could not reach recurrent/encoder RMS %.4g within %.2f%% by ' ...
        'varying the phase-B bias scale over [%.4g, %.4g] mV. Best ratio ' ...
        'was %.6g at %.6g mV among candidates within the firing-rate ' ...
        'bounds. Expand the bounds or inspect the trace.'], ...
        target,100*tolerance,bounds(1),bounds(2), ...
        ratioMeasured(bestIndex),scale);
end
end

function [scaleUsed,encoderRms,recurrentRms,ratioMeasured, ...
        grossEncoderRms,grossRecurrentRms,meanRateHz,relativeTargetError, ...
        summaries,count] = evaluate_scale(scale,scaleUsed,encoderRms, ...
        recurrentRms,ratioMeasured,grossEncoderRms,grossRecurrentRms, ...
        meanRateHz,relativeTargetError,summaries,count,baseOverrides, ...
        baseBias,phasePattern,meanOffset,trainingSequences,target, ...
        durationSeconds,warmupSeconds,rateBoundsHz)
%EVALUATE_SCALE Simulate one structured initial bias without training.
bias = single(baseBias) + single(meanOffset) + ...
    single(scale) .* single(phasePattern);
localOverrides = baseOverrides;
localOverrides.initial_bias = bias;
summary = training_supervised_current_summary(localOverrides, ...
    trainingSequences,durationSeconds,warmupSeconds);
count = count + 1;
scaleUsed(count) = scale;
encoderRms(count) = summary.encoder_rms_mV;
recurrentRms(count) = summary.net_recurrent_rms_mV;
ratioMeasured(count) = summary.recurrent_to_encoder_rms;
grossEncoderRms(count) = summary.gross_encoder_rms_mV;
grossRecurrentRms(count) = summary.gross_recurrent_rms_mV;
meanRateHz(count) = summary.mean_rate_hz;
if summary.finite && summary.recurrent_to_encoder_rms >= 0 && ...
        summary.mean_rate_hz >= rateBoundsHz(1) && ...
        summary.mean_rate_hz <= rateBoundsHz(2)
    relativeTargetError(count) = ...
        abs(summary.recurrent_to_encoder_rms-target)/target;
end
summaries{count} = summary;
end

function summary = training_supervised_current_summary( ...
        overrides,trainingSequences,durationSeconds,warmupSeconds)
%TRAINING_SUPERVISED_CURRENT_SUMMARY Probe fully teacher-forced dynamics
% using normalized training trajectories and the exact production GPU step.
cfg = banff('config', 'lorenz', overrides);
dimension = size(trainingSequences,1);
P = banff_model('gpu', banff_model('create', dimension, dimension, cfg));
trialCount = size(trainingSequences,3);
totalSteps = round(durationSeconds/double(P.dt));
warmupSteps = round(warmupSeconds/double(P.dt));
if totalSteps <= warmupSteps
    error('banff:lorenzBiasProbeSteps', ...
        'The calibration probe has no scored timesteps.');
end

state = diagnostic_state(P,trialCount);
trainingSequences = gpuArray(single(trainingSequences));
encoderSquare = gpuArray.zeros(1,1,'single');
recurrentSquare = gpuArray.zeros(1,1,'single');
grossEncoderSquare = gpuArray.zeros(1,1,'single');
grossRecurrentSquare = gpuArray.zeros(1,1,'single');
adaptationSquare = gpuArray.zeros(1,1,'single');
spikeCount = gpuArray.zeros(1,1,'single');

for step = 1:totalSteps
    inputSignal = reshape(trainingSequences(:,step,:),dimension,trialCount);
    inputCurrent = P.W_in*(P.inputScale.*inputSignal);
    grossInputCurrent = abs(P.W_in)*(P.inputScale.*abs(inputSignal));
    recurrentCurrent = recurrent_current(P,state.r);
    grossRecurrentCurrent = gross_recurrent_current(P,state.r);
    adaptationBeforeStep = state.w;
    [state,spike] = banff_model('gpu_step',P,state,inputCurrent,false);
    if step > warmupSteps
        encoderSquare = encoderSquare+sum(inputCurrent.^2,'all');
        recurrentSquare = recurrentSquare+sum(recurrentCurrent.^2,'all');
        grossEncoderSquare = grossEncoderSquare+ ...
            sum(grossInputCurrent.^2,'all');
        grossRecurrentSquare = grossRecurrentSquare+ ...
            sum(grossRecurrentCurrent.^2,'all');
        adaptationSquare = adaptationSquare+sum(adaptationBeforeStep.^2,'all');
        spikeCount = spikeCount+sum(single(spike),'all');
    end
end

scoredSteps = totalSteps-warmupSteps;
observationCount = double(P.N_hidden)*trialCount*scoredSteps;
scoredDuration = scoredSteps*double(P.dt);
summary = struct();
summary.encoder_rms_mV = sqrt(double(gather(encoderSquare))/observationCount);
summary.net_recurrent_rms_mV = ...
    sqrt(double(gather(recurrentSquare))/observationCount);
summary.gross_encoder_rms_mV = ...
    sqrt(double(gather(grossEncoderSquare))/observationCount);
summary.gross_recurrent_rms_mV = ...
    sqrt(double(gather(grossRecurrentSquare))/observationCount);
summary.adaptation_rms_mV = ...
    sqrt(double(gather(adaptationSquare))/observationCount);
summary.recurrent_to_encoder_rms = summary.net_recurrent_rms_mV/ ...
    max(summary.encoder_rms_mV,realmin);
summary.net_to_gross_encoder_rms = summary.encoder_rms_mV/ ...
    max(summary.gross_encoder_rms_mV,realmin);
summary.net_to_gross_recurrent_rms = summary.net_recurrent_rms_mV/ ...
    max(summary.gross_recurrent_rms_mV,realmin);
summary.mean_rate_hz = double(gather(spikeCount))/ ...
    (double(P.N_hidden)*trialCount*scoredDuration);
numericValues = [summary.encoder_rms_mV summary.net_recurrent_rms_mV ...
    summary.gross_encoder_rms_mV summary.gross_recurrent_rms_mV ...
    summary.adaptation_rms_mV summary.recurrent_to_encoder_rms ...
    summary.net_to_gross_encoder_rms summary.net_to_gross_recurrent_rms ...
    summary.mean_rate_hz];
summary.finite = all(isfinite(numericValues));
end

function state = diagnostic_state(P,batchSize)
%DIAGNOSTIC_STATE Match the production simulator's complete reset state.
state = struct('u',repmat(P.restingVoltage,P.N_hidden,batchSize), ...
    'w',gpuArray.zeros(P.N_hidden,batchSize,'single'), ...
    'x',gpuArray.zeros(P.N_hidden,batchSize,'single'), ...
    'r',gpuArray.zeros(P.N_hidden,batchSize,'single'), ...
    'epsilonVoltage',gpuArray.zeros(P.N_hidden,batchSize,'single'), ...
    'epsilonAdaptation',gpuArray.zeros(P.N_hidden,batchSize,'single'), ...
    'eligibilityRise',gpuArray.zeros(P.N_hidden,batchSize,'single'), ...
    'eligibilityDecay',gpuArray.zeros(P.N_hidden,batchSize,'single'));
end

function current = recurrent_current(P,filteredSpikes)
%RECURRENT_CURRENT Reproduce the signed production recurrent operator.
if P.recurrent_mode == "full_rank"
    current = single(P.W_recurrent*double(filteredSpikes));
else
    current = P.recurrentGain.*(P.recurrent_expansion* ...
        (P.W_feedback*filteredSpikes))-P.self_coupling.*filteredSpikes;
end
end

function gross = gross_recurrent_current(P,filteredSpikes)
%GROSS_RECURRENT_CURRENT Sum absolute presynaptic recurrent contributions.
if P.recurrent_mode == "full_rank"
    gross = single(abs(P.W_recurrent)*double(filteredSpikes));
else
    gross = P.recurrentGain.*(P.recurrent_expansion* ...
        (abs(P.W_feedback)*filteredSpikes)) ...
        -abs(P.self_coupling).*filteredSpikes;
    gross = max(gross,single(0));
end
end

function output = standardize_pattern(input)
%STANDARDIZE_PATTERN Produce a bounded zero-mean, unit-SD spatial pattern.
output = single(input(:));
output = output-mean(output);
scale = std(output);
if ~isfinite(scale) || scale <= 0
    error('banff:lorenzBiasPattern', ...
        'The recurrent-mode bias pattern has invalid variance.');
end
output = output./scale;
output = min(max(output,single(-3)),single(3));
output = output-mean(output);
output = output./std(output);
end

function [vector,eigenvalue,flag] = dominant_recurrent_mode(P)
%DOMINANT_RECURRENT_MODE Find the largest-magnitude effective-current mode.
if P.recurrent_mode == "low_rank"
    apply = @(x) double(P.recurrentGain).*(double(P.recurrent_expansion)* ...
        (double(P.W_feedback)*x))-double(P.self_coupling).*x;
else
    matrix = double(P.W_recurrent);
    apply = @(x) matrix*x;
end
indices = (1:P.N_hidden).';
start = sin(indices.*sqrt(2))+cos(indices.*sqrt(3));
start = start./norm(start);
options = struct('isreal',true,'issym',false,'tol',1e-7, ...
    'maxit',1000,'disp',0,'v0',start);
[vector,matrix,flag] = eigs(apply,P.N_hidden,1,'largestabs',options);
eigenvalue = matrix(1,1);
end
