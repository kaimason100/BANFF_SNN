%% Does training retain a recurrent-mode-aligned bias initialization?
% This controlled experiment trains breast-cancer networks that differ only
% in their initial neuronal bias vector:
%   1. the ordinary uniform BANFF initialization; and
%   2. a heterogeneous vector aligned with phase B of the dominant recurrent
%      eigenmode, using the configuration that produced strong recurrent
%      currents in the preceding operating-regime diagnostic.
%
% Training, validation model selection, fixed weights, data split, optimizer,
% sample order and learning-rate schedule are otherwise identical. Test data
% are evaluated only after every requested run has completed.

clear;
close all;
clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(projectRoot);

%% Editable settings
networkSeeds = 2;             % use 1:3 for a stronger multi-seed comparison
epochs = 200;
phaseBMeanOffsetMv = single(1);
checkpointHours = inf;
reuseCompletedRuns = true;

% Keep all fixed-weight gains at their ordinary values. Calibrate only the
% amplitude of the phase-B bias pattern so that this initialization begins
% with the requested recurrent/encoder RMS ratio on training inputs. The
% uniform control retains the ordinary scalar initial bias. No validation or
% test observations enter this calibration.
targetInitialRecurrentToEncoderRatio = 1;
phaseBPatternScaleInitialGuessMv = 2;
phaseBPatternScaleBoundsMv = [0.25 8];
biasCalibrationTolerance = 0.03;
biasCalibrationRefinementIterations = 5;

experimentOutputDirectory = fullfile(projectRoot, 'outputs', ...
    'initialization_experiment');

commonOverrides = struct();
commonOverrides.N_hidden = 32000;
commonOverrides.N_recurrent = 10;
commonOverrides.recurrent_mode = 'low_rank';
commonOverrides.encoder_gain = single(2);
commonOverrides.recurrent_gain = single(0.05);
commonOverrides.decoder_gain = single(0.1);
commonOverrides.epochs = epochs;
commonOverrides.checkpoint_hours = checkpointHours;
commonOverrides.output_directory = experimentOutputDirectory;

if ~canUseGPU
    error('banff:biasInitializationGPU', ...
        'This training experiment requires a supported MATLAB GPU.');
end

conditionNames = ["Uniform initialization"; "Dominant phase-B initialization"];
conditionCount = numel(conditionNames);
runCount = conditionCount * numel(networkSeeds);
trainedRuns = cell(runCount, 1);
testedRuns = cell(runCount, 1);
runCondition = strings(runCount, 1);
runSeed = zeros(runCount, 1);
trainingSecondsThisInvocation = nan(runCount, 1);
initialEncoderRmsMv = nan(runCount, 1);
initialRecurrentRmsMv = nan(runCount, 1);
initialRecurrentToEncoderRms = nan(runCount, 1);
initialGrossEncoderRmsMv = nan(runCount, 1);
initialGrossRecurrentRmsMv = nan(runCount, 1);
initialNetToGrossEncoderRms = nan(runCount, 1);
initialNetToGrossRecurrentRms = nan(runCount, 1);
phaseBPatternScaleUsedMv = nan(runCount, 1);
runIndex = 0;

fprintf('Controlled breast-cancer initialization experiment\n');
fprintf('Conditions: ordinary uniform versus dominant phase-B bias\n');
fprintf('Epochs: %d | seeds: %s\n\n', epochs, mat2str(networkSeeds));

%% Train every condition using validation data only for model selection
for seedIndex = 1:numel(networkSeeds)
    seed = networkSeeds(seedIndex);
    seedOverrides = commonOverrides;
    seedOverrides.seed = seed;
    baseCfg = banff('config', 'breast_cancer', seedOverrides);
    [seedData, ~] = banff_data('static', baseCfg);
    Pinitial = banff_model('create', size(seedData.X_train, 1), ...
        size(seedData.Y_train, 1), baseCfg);

    [dominantVector, dominantEigenvalue, eigsFlag] = ...
        dominant_recurrent_mode(Pinitial);
    if std(imag(dominantVector)) <= 1e-8
        error('banff:biasInitializationRealMode', ...
            ['The dominant recurrent eigenmode is real for seed %d, so the ' ...
            'requested phase-B initialization is unavailable.'], seed);
    end
    phaseB = standardize_pattern(single(imag(dominantVector)));
    [phaseBBias, phaseBPatternScaleMv, achievedRatio, calibrationTrace] = ...
        calibrate_phase_bias_scale(seedOverrides, baseCfg.initial_bias, ...
        phaseB, phaseBMeanOffsetMv, phaseBPatternScaleInitialGuessMv, ...
        seedData.X_train, size(seedData.Y_train, 1), ...
        targetInitialRecurrentToEncoderRatio, ...
        phaseBPatternScaleBoundsMv, biasCalibrationTolerance, ...
        biasCalibrationRefinementIterations);

    fprintf('Seed %d dominant eigenvalue: %.6g %+.6gi (flag %d)\n', ...
        seed, real(dominantEigenvalue), imag(dominantEigenvalue), eigsFlag);
    fprintf(['Fixed gains: encoder %.6g, recurrent %.6g, decoder %.6g.\n'], ...
        double(seedOverrides.encoder_gain), ...
        double(seedOverrides.recurrent_gain), ...
        double(seedOverrides.decoder_gain));
    fprintf(['Calibrated phase-B bias-pattern scale %.6g mV gives ' ...
        'training-input ' ...
        'recurrent/encoder RMS %.6g (target %.6g).\n'], ...
        phaseBPatternScaleMv, achievedRatio, ...
        targetInitialRecurrentToEncoderRatio);
    disp(calibrationTrace);

    initializations = {single(baseCfg.initial_bias); phaseBBias};
    for conditionIndex = 1:conditionCount
        runIndex = runIndex + 1;
        runCondition(runIndex) = conditionNames(conditionIndex);
        runSeed(runIndex) = seed;
        if conditionIndex == 2
            phaseBPatternScaleUsedMv(runIndex) = phaseBPatternScaleMv;
        end

        overrides = seedOverrides;
        overrides.initial_bias = initializations{conditionIndex};
        resolvedCfg = banff('config', 'breast_cancer', overrides);

        initialCurrent = initial_current_summary( ...
            overrides, initializations{conditionIndex}, seedData.X_train, ...
            size(seedData.Y_train, 1));
        initialEncoderRmsMv(runIndex) = initialCurrent.encoder_rms_mV;
        initialRecurrentRmsMv(runIndex) = ...
            initialCurrent.net_recurrent_rms_mV;
        initialRecurrentToEncoderRms(runIndex) = ...
            initialCurrent.recurrent_to_encoder_rms;
        initialGrossEncoderRmsMv(runIndex) = ...
            initialCurrent.gross_encoder_rms_mV;
        initialGrossRecurrentRmsMv(runIndex) = ...
            initialCurrent.gross_recurrent_rms_mV;
        initialNetToGrossEncoderRms(runIndex) = ...
            initialCurrent.net_to_gross_encoder_rms;
        initialNetToGrossRecurrentRms(runIndex) = ...
            initialCurrent.net_to_gross_recurrent_rms;

        fprintf('\nRun %d/%d: %s, seed %d\n', runIndex, runCount, ...
            char(conditionNames(conditionIndex)), seed);
        fprintf(['Initial training-input currents: encoder RMS %.6g mV, ' ...
            'recurrent RMS %.6g mV, recurrent/encoder %.6g; gross ' ...
            'encoder %.6g mV, gross recurrent %.6g mV.\n'], ...
            initialEncoderRmsMv(runIndex), ...
            initialRecurrentRmsMv(runIndex), ...
            initialRecurrentToEncoderRms(runIndex), ...
            initialGrossEncoderRmsMv(runIndex), ...
            initialGrossRecurrentRmsMv(runIndex));
        timer = tic;
        trainedRuns{runIndex} = train_or_reuse( ...
            resolvedCfg, overrides, reuseCompletedRuns);
        trainingSecondsThisInvocation(runIndex) = toc(timer);

        if ~trainedRuns{runIndex}.complete
            fprintf(['Run checkpointed before completion. Rerun this script ' ...
                'with identical settings to resume it. Test data have not ' ...
                'been evaluated.\n']);
            return;
        end
    end
end

%% Final held-out testing after all validation-selected models are complete
fprintf('\nAll training runs completed. Beginning final held-out testing.\n');
for runIndex = 1:runCount
    result = trainedRuns{runIndex};
    overrides = commonOverrides;
    overrides.seed = runSeed(runIndex);
    overrides.encoder_gain = result.config.encoder_gain;
    overrides.initial_bias = result.config.initial_bias;
    testedRuns{runIndex} = banff('test', 'breast_cancer', overrides);
end

%% Measure currents and recurrence dependence on the held-out test set
validationAccuracyPercent = zeros(runCount, 1);
bestEpoch = zeros(runCount, 1);
testAccuracyPercent = zeros(runCount, 1);
ablatedAccuracyPercent = zeros(runCount, 1);
ablationDropPercentagePoints = zeros(runCount, 1);
predictionsChangedPercent = zeros(runCount, 1);
encoderRmsMv = zeros(runCount, 1);
recurrentRmsMv = zeros(runCount, 1);
recurrentToEncoderRms = zeros(runCount, 1);
grossRecurrentRmsMv = zeros(runCount, 1);
grossEncoderRmsMv = zeros(runCount, 1);
netToGrossEncoderRms = zeros(runCount, 1);
netToGrossRecurrentRms = zeros(runCount, 1);
adaptationRmsMv = zeros(runCount, 1);
meanRateHz = zeros(runCount, 1);
activeNeuronPercent = zeros(runCount, 1);
inverseIsiCount = zeros(runCount, 1);
inverseIsiMedianHz = nan(runCount, 1);
inverseIsiP10Hz = nan(runCount, 1);
inverseIsiP90Hz = nan(runCount, 1);
finalBiasMeanMv = zeros(runCount, 1);
finalBiasStdMv = zeros(runCount, 1);
initialBiasMeanMv = zeros(runCount, 1);
initialBiasStdMv = zeros(runCount, 1);
initializationDisplacementRmsMv = zeros(runCount, 1);
currentSummaries = cell(runCount, 1);
firingRateDistributionsHz = cell(runCount, 1);
inverseIsiRateDistributionsHz = cell(runCount, 1);

for runIndex = 1:runCount
    result = testedRuns{runIndex};
    cfg = result.config;
    [data, ~] = banff_data('static', cfg, result.data_information);
    P = banff_model('create', size(data.X_train, 1), ...
        size(data.Y_train, 1), cfg);
    initialBias = single(P.B(:));
    P.B = single(result.best.B(:));

    currentSummaries{runIndex} = banff_plot( ...
        'static_current_magnitudes', P, data.X_test, cfg);
    aggregate = currentSummaries{runIndex}.aggregate;

    Pablated = remove_recurrence(P);
    ablated = banff_eval('static', banff_model('gpu', Pablated), ...
        data.X_test, data.Y_test, cfg, true);
    ablatedStatistics = banff_metrics('classification', ...
        ablated.output, data.Y_test);

    validationAccuracyPercent(runIndex) = double(result.best.metric);
    bestEpoch(runIndex) = result.best.epoch;
    testAccuracyPercent(runIndex) = double(result.test.statistics.accuracy_percent);
    ablatedAccuracyPercent(runIndex) = ...
        double(ablatedStatistics.accuracy_percent);
    ablationDropPercentagePoints(runIndex) = ...
        testAccuracyPercent(runIndex) - ablatedAccuracyPercent(runIndex);
    [~, fullPrediction] = max(result.test.output, [], 1);
    [~, ablatedPrediction] = max(ablated.output, [], 1);
    predictionsChangedPercent(runIndex) = ...
        100 * mean(fullPrediction ~= ablatedPrediction);

    encoderRmsMv(runIndex) = aggregate.encoder_rms_mV;
    recurrentRmsMv(runIndex) = aggregate.net_recurrent_rms_mV;
    recurrentToEncoderRms(runIndex) = aggregate.recurrent_to_encoder_rms;
    grossEncoderRmsMv(runIndex) = aggregate.gross_encoder_rms_mV;
    grossRecurrentRmsMv(runIndex) = aggregate.gross_recurrent_rms_mV;
    netToGrossEncoderRms(runIndex) = aggregate.net_to_gross_encoder_rms;
    netToGrossRecurrentRms(runIndex) = aggregate.net_to_gross_recurrent_rms;
    adaptationRmsMv(runIndex) = aggregate.adaptation_rms_mV;
    rates = double(result.test.neural_activity.mean_firing_rate_by_neuron_hz(:));
    firingRateDistributionsHz{runIndex} = rates;
    inverseIsiRateDistributionsHz{runIndex} = banff_plot( ...
        'static_inverse_isi_rates', P, data.X_test, cfg);
    inverseRates = inverseIsiRateDistributionsHz{runIndex};
    inverseIsiCount(runIndex) = numel(inverseRates);
    if ~isempty(inverseRates)
        inverseIsiP10Hz(runIndex) = linear_percentile(inverseRates, 0.10);
        inverseIsiMedianHz(runIndex) = linear_percentile(inverseRates, 0.50);
        inverseIsiP90Hz(runIndex) = linear_percentile(inverseRates, 0.90);
    end
    meanRateHz(runIndex) = mean(rates);
    activeNeuronPercent(runIndex) = ...
        result.test.neural_activity.active_fraction_percent;

    finalBias = double(result.best.B(:));
    initialBiasDouble = double(initialBias);
    finalBiasMeanMv(runIndex) = mean(finalBias);
    finalBiasStdMv(runIndex) = std(finalBias);
    initialBiasMeanMv(runIndex) = mean(initialBiasDouble);
    initialBiasStdMv(runIndex) = std(initialBiasDouble);
    initializationDisplacementRmsMv(runIndex) = sqrt(mean( ...
        (finalBias - initialBiasDouble).^2));
end

results = table(runCondition, runSeed, phaseBPatternScaleUsedMv, ...
    initialEncoderRmsMv, initialRecurrentRmsMv, ...
    initialRecurrentToEncoderRms, initialGrossEncoderRmsMv, ...
    initialGrossRecurrentRmsMv, initialNetToGrossEncoderRms, ...
    initialNetToGrossRecurrentRms, initialBiasMeanMv, initialBiasStdMv, ...
    finalBiasMeanMv, finalBiasStdMv, initializationDisplacementRmsMv, ...
    bestEpoch, validationAccuracyPercent, testAccuracyPercent, ...
    ablatedAccuracyPercent, ablationDropPercentagePoints, ...
    predictionsChangedPercent, encoderRmsMv, recurrentRmsMv, ...
    recurrentToEncoderRms, grossEncoderRmsMv, grossRecurrentRmsMv, ...
    netToGrossEncoderRms, netToGrossRecurrentRms, adaptationRmsMv, meanRateHz, ...
    activeNeuronPercent, inverseIsiCount, inverseIsiP10Hz, ...
    inverseIsiMedianHz, inverseIsiP90Hz, trainingSecondsThisInvocation, ...
    'VariableNames', {'Condition','Seed','PhaseBPatternScaleUsedMv', ...
    'InitialEncoderRmsMv','InitialRecurrentRmsMv', ...
    'InitialRecurrentToEncoderRms','InitialGrossEncoderRmsMv', ...
    'InitialGrossRecurrentRmsMv','InitialNetToGrossEncoderRms', ...
    'InitialNetToGrossRecurrentRms','InitialBiasMeanMv','InitialBiasStdMv', ...
    'FinalBiasMeanMv','FinalBiasStdMv','BiasDisplacementRmsMv','BestEpoch', ...
    'ValidationAccuracyPercent','TestAccuracyPercent', ...
    'AblatedAccuracyPercent','AblationDropPercentagePoints', ...
    'PredictionsChangedPercent','EncoderRmsMv','RecurrentRmsMv', ...
    'RecurrentToEncoderRms','GrossEncoderRmsMv','GrossRecurrentRmsMv', ...
    'NetToGrossEncoderRms','NetToGrossRecurrentRms', ...
    'AdaptationRmsMv','MeanRateHz', ...
    'ActiveNeuronPercent','InverseIsiCount','InverseIsiP10Hz', ...
    'InverseIsiMedianHz','InverseIsiP90Hz', ...
    'TrainingSecondsThisInvocation'});

fprintf('\nControlled initialization experiment results\n');
disp(results);

%% Learning curves and final functional comparison
figure('Color', 'w');
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
hold on;
for runIndex = 1:runCount
    validation = double(trainedRuns{runIndex}.history.validation_metric(:));
    epochsRecorded = find(isfinite(validation));
    plot(epochsRecorded, validation(epochsRecorded), 'LineWidth', 1.1, ...
        'DisplayName', sprintf('%s, seed %d', ...
        char(runCondition(runIndex)), runSeed(runIndex)));
end
hold off;
grid on;
xlabel('Epoch');
ylabel('Validation accuracy (%)');
title('Validation-selected learning trajectories');
legend('Location', 'best');

nexttile;
runLabels = runCondition + ", seed " + string(runSeed);
conditionAxis = categorical(runLabels);
conditionAxis = reordercats(conditionAxis, cellstr(runLabels));
bar(conditionAxis, [testAccuracyPercent ablatedAccuracyPercent]);
grid on;
ylabel('Held-out test accuracy (%)');
title('Final recurrence ablation');
legend({'Full network','Recurrence removed'}, 'Location', 'best');

nexttile;
bar(conditionAxis, [encoderRmsMv recurrentRmsMv adaptationRmsMv]);
grid on;
ylabel('Population RMS magnitude (mV)');
title('Final test-set current magnitudes');
legend({'Encoder','Recurrent','Adaptation'}, 'Location', 'best');

nexttile;
bar(conditionAxis, [grossEncoderRmsMv grossRecurrentRmsMv]);
grid on;
ylabel('Population RMS gross afferent magnitude (mV)');
title('Absolute afferents before cancellation');
legend({'Encoder gross','Recurrent gross'}, 'Location', 'best');

nexttile;
scatter(recurrentToEncoderRms, ablationDropPercentagePoints, 80, ...
    testAccuracyPercent, 'filled');
xline(1, 'k--');
yline(0, ':');
grid on;
xlabel('Recurrent / encoder RMS');
ylabel('Ablation loss (percentage points)');
title('Did training retain functional recurrence?');
colorbar;

nexttile;
bar(conditionAxis, [netToGrossEncoderRms netToGrossRecurrentRms]);
grid on;
ylabel('Net / gross RMS');
title('Afferent cancellation');
legend({'Encoder','Recurrent'}, 'Location', 'best');

%% Full-test-set firing-rate and inverse-ISI distributions
figure('Color', 'w');
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
distributionColors = lines(runCount);

nexttile;
hold on;
allRates = vertcat(firingRateDistributionsHz{:});
rateMaximum = max([1; allRates]);
rateEdges = linspace(0, rateMaximum, 51);
for index = 1:runCount
    histogram(firingRateDistributionsHz{index}, rateEdges, ...
        'Normalization', 'probability', 'DisplayStyle', 'stairs', ...
        'LineWidth', 1.5, 'EdgeColor', distributionColors(index, :), ...
        'DisplayName', char(runLabels(index)));
end
hold off;
grid on;
xlabel('Mean firing rate over complete test set (Hz)');
ylabel('Fraction of neurons');
title('Per-neuron mean firing-rate distribution');
legend('Location', 'best');

nexttile;
hold on;
allInverseIsiRates = vertcat(inverseIsiRateDistributionsHz{:});
if isempty(allInverseIsiRates)
    axis off;
    text(0.5, 0.5, 'No repeated-neuron spikes in the test set', ...
        'HorizontalAlignment', 'center');
else
    inverseMinimum = min(allInverseIsiRates);
    inverseMaximum = max(allInverseIsiRates);
    if inverseMinimum == inverseMaximum
        inverseEdges = inverseMinimum + [-0.5 0.5];
    else
        inverseEdges = linspace(inverseMinimum, inverseMaximum, 51);
    end
    for index = 1:runCount
        values = inverseIsiRateDistributionsHz{index};
        if ~isempty(values)
            histogram(values, inverseEdges, 'Normalization', 'probability', ...
                'DisplayStyle', 'stairs', 'LineWidth', 1.5, ...
                'EdgeColor', distributionColors(index, :), ...
                'DisplayName', char(runLabels(index)));
        end
    end
    grid on;
    xlabel('Inverse inter-spike interval, 1/ISI (Hz)');
    ylabel('Fraction of within-sample intervals');
    title('Full-test instantaneous-rate distribution');
    legend('Location', 'best');
end
hold off;

if exist(experimentOutputDirectory, 'dir') ~= 7
    mkdir(experimentOutputDirectory);
end
comparisonFile = fullfile(experimentOutputDirectory, ...
    'breast_cancer_initialization_comparison.mat');
save(comparisonFile, 'results', 'trainedRuns', 'testedRuns', ...
    'currentSummaries', 'firingRateDistributionsHz', ...
    'inverseIsiRateDistributionsHz', '-v7.3');
fprintf('Saved comparison: %s\n', comparisonFile);

%% Local functions
function result = train_or_reuse(cfg, overrides, reuseCompleted)
%TRAIN_OR_REUSE Reuse only a complete result with the exact resolved identity.
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
result = banff('train', 'breast_cancer', overrides);
end

function [bias, scale, achievedRatio, trace] = calibrate_phase_bias_scale( ...
        baseOverrides, baseBias, phasePattern, meanOffset, initialScale, ...
        trainingInputs, outputDimension, target, bounds, tolerance, ...
        refinementIterations)
%CALIBRATE_PHASE_BIAS_SCALE Match current balance by changing biases only.
% The encoder, recurrent and decoder gains remain exactly fixed. A coarse
% bounded search is followed by local interval refinement because spike
% thresholds make the ratio a nonlinear and potentially non-monotonic
% function of the bias-pattern amplitude.
if numel(bounds) ~= 2 || any(~isfinite(bounds)) || bounds(1) < 0 || ...
        bounds(2) <= bounds(1)
    error('banff:biasScaleBounds', ...
        'phaseBPatternScaleBoundsMv must be finite increasing bounds.');
end

coarseScales = linspace(bounds(1), bounds(2), 9);
candidateScales = unique([coarseScales initialScale], 'sorted');
maximumEvaluations = numel(candidateScales) + 2 * refinementIterations;
scaleUsed = nan(maximumEvaluations, 1);
ratioMeasured = nan(maximumEvaluations, 1);
relativeTargetError = nan(maximumEvaluations, 1);
evaluationCount = 0;

for index = 1:numel(candidateScales)
    [scaleUsed, ratioMeasured, relativeTargetError, evaluationCount] = ...
        evaluate_scale(candidateScales(index), scaleUsed, ratioMeasured, ...
        relativeTargetError, evaluationCount, baseOverrides, baseBias, ...
        phasePattern, meanOffset, trainingInputs, outputDimension, target);
end

for refinement = 1:refinementIterations
    [~, order] = sort(scaleUsed(1:evaluationCount));
    sortedScales = scaleUsed(order);
    sortedErrors = relativeTargetError(order);
    [bestError, bestPosition] = min(sortedErrors);
    if bestError <= tolerance
        break;
    end

    newScales = zeros(0, 1);
    if bestPosition > 1
        newScales(end + 1, 1) = mean( ...
            sortedScales(bestPosition + [-1 0])); %#ok<AGROW>
    end
    if bestPosition < numel(sortedScales)
        newScales(end + 1, 1) = mean( ...
            sortedScales(bestPosition + [0 1])); %#ok<AGROW>
    end
    if isempty(newScales)
        break;
    end
    for index = 1:numel(newScales)
        [scaleUsed, ratioMeasured, relativeTargetError, evaluationCount] = ...
            evaluate_scale(newScales(index), scaleUsed, ratioMeasured, ...
            relativeTargetError, evaluationCount, baseOverrides, baseBias, ...
            phasePattern, meanOffset, trainingInputs, outputDimension, target);
    end
end

[bestError, bestIndex] = min(relativeTargetError(1:evaluationCount));
scale = scaleUsed(bestIndex);
achievedRatio = ratioMeasured(bestIndex);
bias = single(baseBias) + single(meanOffset) + ...
    single(scale) .* single(phasePattern);
trace = table((1:evaluationCount).', scaleUsed(1:evaluationCount), ...
    ratioMeasured(1:evaluationCount), ...
    relativeTargetError(1:evaluationCount), ...
    'VariableNames', {'Evaluation','PhaseBPatternScaleMv', ...
    'RecurrentToEncoderRms','RelativeTargetError'});
if ~isfinite(achievedRatio) || bestError > tolerance
    error('banff:biasScaleCalibrationTarget', ...
        ['Could not reach recurrent/encoder RMS %.4g within %.2f%% by ' ...
        'varying the phase-B bias-pattern scale from %.4g to %.4g mV. ' ...
        'Best ratio %.6g at scale %.6g mV. Expand the scale bounds or ' ...
        'inspect the calibration trace.'], ...
        target, 100 * tolerance, bounds(1), bounds(2), ...
        achievedRatio, scale);
end
end

function [scaleUsed, ratioMeasured, relativeTargetError, count] = ...
        evaluate_scale(scale, scaleUsed, ratioMeasured, ...
        relativeTargetError, count, baseOverrides, baseBias, phasePattern, ...
        meanOffset, trainingInputs, outputDimension, target)
%EVALUATE_SCALE Simulate one bias-only calibration candidate.
bias = single(baseBias) + single(meanOffset) + ...
    single(scale) .* single(phasePattern);
summary = initial_current_summary(baseOverrides, bias, trainingInputs, ...
    outputDimension);
ratio = double(summary.recurrent_to_encoder_rms);
if ~isfinite(ratio) || ratio < 0
    error('banff:biasScaleCalibrationFinite', ...
        'Bias-scale calibration produced an invalid current ratio.');
end
count = count + 1;
scaleUsed(count) = scale;
ratioMeasured(count) = ratio;
relativeTargetError(count) = abs(ratio - target) / target;
end

function value = linear_percentile(values, probability)
%LINEAR_PERCENTILE Toolbox-independent linearly interpolated percentile.
values = sort(double(values(:)));
if isempty(values) || ~isscalar(probability) || probability < 0 || ...
        probability > 1
    error('banff:percentileInput', ...
        'Percentiles require nonempty values and a probability in [0,1].');
end
position = 1 + probability * (numel(values) - 1);
lowerIndex = floor(position);
upperIndex = ceil(position);
fraction = position - lowerIndex;
value = (1 - fraction) * values(lowerIndex) + ...
    fraction * values(upperIndex);
end

function aggregate = initial_current_summary( ...
        overrides, initialBias, inputs, outputDimension)
%INITIAL_CURRENT_SUMMARY Evaluate one initialization without training.
localOverrides = overrides;
localOverrides.initial_bias = single(initialBias(:));
cfg = banff('config', 'breast_cancer', localOverrides);
P = banff_model('create', size(inputs, 1), outputDimension, cfg);
summary = banff_plot('static_current_magnitudes', P, inputs, cfg);
aggregate = summary.aggregate;
end

function P = remove_recurrence(P)
%REMOVE_RECURRENCE Zero the exact production recurrent-current operator.
if P.recurrent_mode == "low_rank"
    P.recurrentGain = single(0);
    P.self_coupling(:) = single(0);
else
    P.W_recurrent = sparse([], [], [], P.N_hidden, P.N_hidden);
end
end

function output = standardize_pattern(input)
%STANDARDIZE_PATTERN Return a bounded zero-mean, unit-SD real pattern.
output = single(input(:));
output = output - mean(output);
scale = std(output);
if ~isfinite(scale) || scale <= 0
    error('banff:biasInitializationPattern', ...
        'The recurrent-mode bias pattern has invalid variance.');
end
output = output ./ scale;
output = min(max(output, single(-3)), single(3));
output = output - mean(output);
output = output ./ std(output);
end

function [vector, eigenvalue, flag] = dominant_recurrent_mode(P)
%DOMINANT_RECURRENT_MODE Calculate the effective operator's dominant mode.
if P.recurrent_mode == "low_rank"
    apply = @(x) double(P.recurrentGain) .* ...
        (double(P.recurrent_expansion) * (double(P.W_feedback) * x)) ...
        - double(P.self_coupling) .* x;
else
    matrix = double(P.W_recurrent);
    apply = @(x) matrix * x;
end
indices = (1:P.N_hidden).';
start = sin(indices .* sqrt(2)) + cos(indices .* sqrt(3));
start = start ./ norm(start);
options = struct('isreal', true, 'issym', false, 'tol', 1e-7, ...
    'maxit', 1000, 'disp', 0, 'v0', start);
[vector, matrix, flag] = eigs(apply, P.N_hidden, 1, ...
    'largestabs', options);
eigenvalue = matrix(1, 1);
end
