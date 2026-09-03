%% Breast-cancer inputs under untrained, trained and mode-aligned biases
% This diagnostic compares several bias configurations while holding the
% encoder, recurrent scaffold, decoder, dataset split and preprocessing fixed.
% Every condition is evaluated on every held-out breast-cancer test sample for
% the complete presentation interval.
%
% Breast cancer is a static classification task: its 30-dimensional input and
% two-dimensional decoder output cannot be connected as a dimension-preserving
% autonomous loop. The scientifically relevant comparison therefore uses the
% actual normalized test inputs and BANFF's standard 300 ms presentation.

clearvars;
close all;
clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(projectRoot);

%% Editable settings
seed = 1;

% Leave empty to use the model file expected by the current default
% breast-cancer configuration. Otherwise provide a complete path to any
% compatible saved breast-cancer result.
modelFile = '';

% Configurations analogous to the recurrent-mode regimes found in the Lorenz
% exploratory sweep. Mode phase A and B are the real and imaginary components
% of one complex dominant eigenvector. Their individual orientation is phase
% conventional; together they sample the dominant two-dimensional eigenspace.
includeUniformPlusTwoMv = true;
includeDominantPhaseA = true;
includeDominantPhaseB = true;
dominantPatternScaleMv = single(2);
phaseAMeanOffsetMv = single(0); % analogous to Lorenz candidate 27
phaseBMeanOffsetMv = single(1); % analogous to Lorenz candidate 40

maximumSwarmNeurons = 3000;
swarmSeed = 619;

if ~canUseGPU
    error('banff:breastBiasComparisonGPU', ...
        'This full-test-set diagnostic requires a supported MATLAB GPU.');
end

%% Locate and load the trained model
if isempty(modelFile)
    requested = struct('seed', seed);
    expectedCfg = banff('config', 'breast_cancer', requested);
    modelFile = expectedCfg.model_file;
end
if exist(modelFile, 'file') ~= 2
    error('banff:breastBiasComparisonModel', ...
        ['No trained model was found at:\n%s\nSet modelFile at the top of ' ...
        'this script to the required saved breast-cancer MAT file.'], modelFile);
end

loaded = load(modelFile, 'result');
if ~isfield(loaded, 'result') || ~isfield(loaded.result, 'config') || ...
        ~isfield(loaded.result, 'best') || ~isfield(loaded.result.best, 'B') || ...
        ~isfield(loaded.result, 'data_information')
    error('banff:breastBiasComparisonFormat', ...
        'The selected file is not a complete BANFF trained-result file.');
end
trained = loaded.result;
cfg = trained.config;
if string(cfg.task) ~= "breast_cancer"
    error('banff:breastBiasComparisonTask', ...
        'The selected trained result is for %s, not breast cancer.', cfg.task);
end

[data, ~] = banff_data('static', cfg, trained.data_information);
X = data.X_test;
Y = data.Y_test;
Pbase = banff_model('create', size(data.X_train, 1), ...
    size(data.Y_train, 1), cfg);
untrainedBias = single(Pbase.B(:));
trainedBias = single(trained.best.B(:));
if numel(trainedBias) ~= Pbase.N_hidden
    error('banff:breastBiasComparisonBias', ...
        'The saved best bias does not match the reconstructed network size.');
end

fprintf('Breast-cancer bias-regime comparison\n');
fprintf('Model: %s\n', modelFile);
fprintf('N: %d | rank: %d | seed: %d | held-out test samples: %d\n', ...
    Pbase.N_hidden, Pbase.N_recurrent, cfg.seed, size(X, 2));
fprintf('Presentation: %d steps (%.3g s) per sample\n\n', ...
    cfg.presentation_steps, double(cfg.presentation_steps) * double(cfg.dt));

%% Obtain directions spanning the dominant recurrent mode
[dominantVector, dominantEigenvalue, eigsFlag] = dominant_recurrent_mode(Pbase);
phaseA = standardize_pattern(single(real(dominantVector)));
phaseBAvailable = std(imag(dominantVector)) > 1e-8;
if phaseBAvailable
    phaseB = standardize_pattern(single(imag(dominantVector)));
else
    phaseB = single(zeros(Pbase.N_hidden, 1));
    if includeDominantPhaseB
        warning('banff:breastBiasRealMode', ...
            ['The dominant eigenmode is real, so a second phase direction ' ...
            'is unavailable and will be skipped.']);
    end
end
fprintf('Dominant effective-current eigenvalue: %.6g %+.6gi (flag %d)\n\n', ...
    real(dominantEigenvalue), imag(dominantEigenvalue), eigsFlag);

%% Define the bias conditions
conditionName = ["Untrained"; "Trained"];
conditionBias = {untrainedBias; trainedBias};

if includeUniformPlusTwoMv
    conditionName(end + 1, 1) = "Initial +2 mV uniform";
    conditionBias{end + 1, 1} = untrainedBias + single(2);
end
if includeDominantPhaseA
    conditionName(end + 1, 1) = "Initial + dominant phase A";
    conditionBias{end + 1, 1} = untrainedBias + phaseAMeanOffsetMv + ...
        dominantPatternScaleMv .* phaseA;
end
if includeDominantPhaseB && phaseBAvailable
    conditionName(end + 1, 1) = "Initial + dominant phase B";
    conditionBias{end + 1, 1} = untrainedBias + phaseBMeanOffsetMv + ...
        dominantPatternScaleMv .* phaseB;
end

conditionCount = numel(conditionName);
currentSummary = cell(conditionCount, 1);
evaluation = cell(conditionCount, 1);
ablatedEvaluation = cell(conditionCount, 1);

%% Evaluate all conditions and their recurrence-off controls
for conditionIndex = 1:conditionCount
    P = Pbase;
    P.B = single(conditionBias{conditionIndex});

    currentSummary{conditionIndex} = banff_plot( ...
        'static_current_magnitudes', P, X, cfg);
    evaluation{conditionIndex} = banff_eval('static', ...
        banff_model('gpu', P), X, Y, cfg, true);

    Pablated = remove_recurrence(P);
    ablatedEvaluation{conditionIndex} = banff_eval('static', ...
        banff_model('gpu', Pablated), X, Y, cfg, true);

    fprintf('%d/%d completed: %s\n', conditionIndex, conditionCount, ...
        char(conditionName(conditionIndex)));
end

%% Assemble directly comparable statistics
biasMeanMv = zeros(conditionCount, 1);
biasStdMv = zeros(conditionCount, 1);
encoderRmsMv = zeros(conditionCount, 1);
recurrentRmsMv = zeros(conditionCount, 1);
recurrentToEncoderRms = zeros(conditionCount, 1);
grossEncoderRmsMv = zeros(conditionCount, 1);
grossRecurrentRmsMv = zeros(conditionCount, 1);
netToGrossEncoderRms = zeros(conditionCount, 1);
netToGrossRecurrentRms = zeros(conditionCount, 1);
adaptationRmsMv = zeros(conditionCount, 1);
accuracyPercent = zeros(conditionCount, 1);
ablatedAccuracyPercent = zeros(conditionCount, 1);
ablationDropPercentagePoints = zeros(conditionCount, 1);
predictionChangePercent = zeros(conditionCount, 1);
meanRateHz = zeros(conditionCount, 1);
activeNeuronPercent = zeros(conditionCount, 1);

for conditionIndex = 1:conditionCount
    bias = double(conditionBias{conditionIndex});
    aggregate = currentSummary{conditionIndex}.aggregate;
    biasMeanMv(conditionIndex) = mean(bias);
    biasStdMv(conditionIndex) = std(bias);
    encoderRmsMv(conditionIndex) = aggregate.encoder_rms_mV;
    recurrentRmsMv(conditionIndex) = aggregate.net_recurrent_rms_mV;
    recurrentToEncoderRms(conditionIndex) = ...
        aggregate.recurrent_to_encoder_rms;
    grossEncoderRmsMv(conditionIndex) = aggregate.gross_encoder_rms_mV;
    grossRecurrentRmsMv(conditionIndex) = ...
        aggregate.gross_recurrent_rms_mV;
    netToGrossEncoderRms(conditionIndex) = ...
        aggregate.net_to_gross_encoder_rms;
    netToGrossRecurrentRms(conditionIndex) = ...
        aggregate.net_to_gross_recurrent_rms;
    adaptationRmsMv(conditionIndex) = aggregate.adaptation_rms_mV;

    fullOutput = evaluation{conditionIndex}.output;
    ablatedOutput = ablatedEvaluation{conditionIndex}.output;
    fullStatistics = banff_metrics('classification', fullOutput, Y);
    ablatedStatistics = banff_metrics('classification', ablatedOutput, Y);
    accuracyPercent(conditionIndex) = fullStatistics.accuracy_percent;
    ablatedAccuracyPercent(conditionIndex) = ...
        ablatedStatistics.accuracy_percent;
    ablationDropPercentagePoints(conditionIndex) = ...
        accuracyPercent(conditionIndex) - ablatedAccuracyPercent(conditionIndex);
    [~, fullPrediction] = max(fullOutput, [], 1);
    [~, ablatedPrediction] = max(ablatedOutput, [], 1);
    predictionChangePercent(conditionIndex) = ...
        100 * mean(fullPrediction ~= ablatedPrediction);

    rates = double(evaluation{conditionIndex}.neural_activity. ...
        mean_firing_rate_by_neuron_hz(:));
    meanRateHz(conditionIndex) = mean(rates);
    activeNeuronPercent(conditionIndex) = ...
        evaluation{conditionIndex}.neural_activity.active_fraction_percent;
end

results = table(conditionName, biasMeanMv, biasStdMv, encoderRmsMv, ...
    recurrentRmsMv, recurrentToEncoderRms, grossEncoderRmsMv, ...
    grossRecurrentRmsMv, netToGrossEncoderRms, ...
    netToGrossRecurrentRms, adaptationRmsMv, meanRateHz, ...
    activeNeuronPercent, accuracyPercent, ablatedAccuracyPercent, ...
    ablationDropPercentagePoints, predictionChangePercent, ...
    'VariableNames', {'Condition','BiasMeanMv','BiasStdMv','EncoderRmsMv', ...
    'RecurrentRmsMv','RecurrentToEncoderRms','GrossEncoderRmsMv', ...
    'GrossRecurrentRmsMv','NetToGrossEncoderRms', ...
    'NetToGrossRecurrentRms','MeanAdaptationRmsMv','MeanRateHz', ...
    'ActiveNeuronPercent','AccuracyPercent','AblatedAccuracyPercent', ...
    'AblationDropPercentagePoints','PredictionsChangedPercent'});

fprintf('\nFull held-out-test comparison\n');
disp(results);
fprintf(['Each current magnitude is the RMS over all %d neurons, %d test ' ...
    'samples and %d presentation timesteps. Signed afferents are summed ' ...
    'before the net-current RMS is calculated.\n'], ...
    Pbase.N_hidden, size(X, 2), cfg.presentation_steps);

%% Aggregate comparison plots
figure('Color', 'w');
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
conditionAxis = categorical(conditionName);
conditionAxis = reordercats(conditionAxis, cellstr(conditionName));

nexttile;
bar(conditionAxis, [encoderRmsMv recurrentRmsMv adaptationRmsMv]);
grid on;
ylabel('Population RMS magnitude (mV)');
title('Full test-set current magnitudes');
legend({'Encoder','Recurrent','Adaptation'}, 'Location', 'best');
xtickangle(20);

nexttile;
bar(conditionAxis, [grossEncoderRmsMv grossRecurrentRmsMv]);
grid on;
ylabel('Population RMS gross afferent magnitude (mV)');
title('Absolute afferents before cancellation');
legend({'Encoder gross','Recurrent gross'}, 'Location', 'best');
xtickangle(20);

nexttile;
bar(conditionAxis, recurrentToEncoderRms);
yline(1, 'k--');
grid on;
ylabel('Recurrent / encoder RMS');
title('Signed-net current balance');
xtickangle(20);

nexttile;
bar(conditionAxis, ...
    [accuracyPercent ablatedAccuracyPercent]);
grid on;
ylabel('Held-out accuracy (%)');
title('Functional recurrence ablation');
legend({'Full network','Recurrence removed'}, 'Location', 'best');
xtickangle(20);

nexttile;
scatter(recurrentToEncoderRms, ablationDropPercentagePoints, 75, ...
    meanRateHz, 'filled');
xline(1, 'k--');
yline(0, ':');
grid on;
xlabel('Recurrent / encoder RMS');
ylabel('Accuracy loss after ablation (percentage points)');
title('Current magnitude versus functional dependence');
colorbar;

nexttile;
bar(conditionAxis, [netToGrossEncoderRms netToGrossRecurrentRms]);
grid on;
ylabel('Net / gross RMS');
title('Afferent cancellation');
legend({'Encoder','Recurrent'}, 'Location', 'best');
xtickangle(20);

%% Per-neuron current distributions
% Plot a deterministic neuron subset for legibility. Every neuron still
% contributes to the numerical summaries and aggregate bars above.
rng(swarmSeed, 'twister');
neuronCount = Pbase.N_hidden;
keptNeurons = randperm(neuronCount, min(maximumSwarmNeurons, neuronCount));

figure('Color', 'w');
columns = min(3, conditionCount);
rows = ceil(conditionCount / columns);
tiledlayout(rows, columns, 'TileSpacing', 'compact', 'Padding', 'compact');
for conditionIndex = 1:conditionCount
    summary = currentSummary{conditionIndex};
    encoderValues = double(summary.encoder_net_rms(keptNeurons));
    recurrentValues = double(summary.recurrent_net_rms(keptNeurons));
    grossEncoderValues = double( ...
        summary.encoder_gross_afferent_rms(keptNeurons));
    grossRecurrentValues = double( ...
        summary.recurrent_gross_afferent_rms(keptNeurons));
    adaptationValues = double(summary.adaptation_rms(keptNeurons));
    plottedValues = [encoderValues(:); recurrentValues(:); ...
        grossEncoderValues(:); grossRecurrentValues(:); adaptationValues(:)];
    plottedGroups = categorical([ ...
        repmat({'Encoder net'}, numel(encoderValues), 1); ...
        repmat({'Recurrent net'}, numel(recurrentValues), 1); ...
        repmat({'Encoder gross'}, numel(grossEncoderValues), 1); ...
        repmat({'Recurrent gross'}, numel(grossRecurrentValues), 1); ...
        repmat({'Adaptation'}, numel(adaptationValues), 1)], ...
        {'Encoder net','Recurrent net','Encoder gross', ...
        'Recurrent gross','Adaptation'});

    nexttile;
    swarmchart(plottedGroups, plottedValues, 5, 'filled', ...
        'MarkerFaceAlpha', 0.18, 'MarkerEdgeAlpha', 0.18);
    hold on;
    boxchart(plottedGroups, plottedValues, 'BoxFaceAlpha', 0.15, ...
        'MarkerStyle', 'none');
    hold off;
    grid on;
    ylabel('Per-neuron RMS magnitude (mV)');
    title(conditionName(conditionIndex));
end

fprintf(['\nInterpretation: mode-aligned configurations show what the fixed ' ...
    'network can express through bias changes. Only the trained row measures ' ...
    'what the current breast-cancer objective actually selected. A large ' ...
    'recurrent current is scientifically useful only if held-out accuracy ' ...
    'also depends on recurrence.\n']);

%% Local functions
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
    error('banff:breastBiasPattern', ...
        'A recurrent-mode bias pattern has invalid variance.');
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
