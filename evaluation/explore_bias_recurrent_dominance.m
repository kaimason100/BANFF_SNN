%% Can bias reconfiguration produce a recurrent-dominant operating regime?
% This diagnostic asks whether changing only neuronal biases can move a fixed
% BANFF network into an operating regime in which the signed-net recurrent
% current is comparable to, or larger than, the closed-loop encoder current.
%
% Each candidate bias vector receives the same reproducible, temporally
% correlated random input prefix. The external drive is then removed and the
% decoder output is fed back through the encoder exactly as in BANFF dynamics
% evaluation. Currents are measured during this autonomous closed-loop phase.
%
% Promising candidates are subsequently simulated with recurrence removed.
% This distinguishes a genuinely recurrence-dependent trajectory from a
% bias-driven high-rate state that merely happens to contain a large current.
%
% The experiment demonstrates that a regime is dynamically accessible; it
% does not demonstrate that the present learning rule will discover that
% regime or that the resulting trajectory solves the target task.

clearvars;
close all;
clc;

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(projectRoot);

%% Editable scientific settings
task = 'lorenz';
networkSeed = 1;

% Leave modelFile empty to start from the configured initial bias. To explore
% the neighbourhood of an already trained model, supply its complete MAT-file
% path. Its saved configuration and validation-selected best bias are used.
modelFile = '';

overrides = struct();
overrides.N_hidden = 32000;
overrides.N_recurrent = 10;
overrides.recurrent_mode = 'low_rank';
overrides.encoder_gain = single(2);
overrides.recurrent_gain = single(0.05);
overrides.decoder_gain = single(0.1);
overrides.seed = networkSeed;

% Bias candidates are B_candidate = B_baseline + mean offset + s*p, where p
% is a zero-mean, unit-standard-deviation spatial pattern. Uniform candidates
% use s=0. Dominant-mode patterns test whether biases can preferentially
% recruit the strongest recurrent subspace; the random pattern is a neutral
% control. Extend either vector for a denser search.
biasMeanOffsetsMv = [-3 -1 0 1 2 4];
biasPatternScalesMv = [1 2];
includeRandomPattern = true;
includeDominantModePatterns = true;

randomInputTrials = 3;
randomDriveTimeSeconds = 0.50;
closedLoopTimeSeconds = 1.50;
closedLoopWarmupSeconds = 0.25;
randomInputRms = single(1);       % normalized state-coordinate units
randomInputCorrelationTime = 0.050;
randomSeed = 8173;

% Plausibility filters do not assert biological truth. They prevent a silent
% state or an extreme rate state from being recommended solely because a
% current ratio is numerically large.
minimumAcceptedRateHz = 0.5;
maximumAcceptedRateHz = 50;
minimumEncoderRmsMv = 0.05;
recurrentDominanceThreshold = 1.0;
minimumAblationOutputDifference = 0.25;
maximumAblationCandidates = 12;

if ~canUseGPU
    error('banff:biasRegimeGPU', ...
        'This production-step diagnostic requires a supported MATLAB GPU.');
end

%% Construct the fixed model and choose the baseline bias
if isempty(modelFile)
    cfg = banff('config', task, overrides);
    if cfg.kind ~= "dynamics"
        error('banff:biasRegimeTask', ...
            'Closed decoder-to-encoder feedback requires a dynamics task.');
    end
    system = banff_data('system', cfg.task);
    stateDimension = numel(system.initial_state);
    Pcpu = banff_model('create', stateDimension, stateDimension, cfg);
    baselineBias = single(Pcpu.B);
    baselineDescription = 'configured initial bias';
else
    if exist(modelFile, 'file') ~= 2
        error('banff:biasRegimeModelMissing', ...
            'The specified model file does not exist: %s', modelFile);
    end
    loaded = load(modelFile, 'result');
    if ~isfield(loaded, 'result') || ~isfield(loaded.result, 'config') || ...
            ~isfield(loaded.result, 'best') || ~isfield(loaded.result.best, 'B')
        error('banff:biasRegimeModelFormat', ...
            'The model file must contain result.config and result.best.B.');
    end
    cfg = loaded.result.config;
    if cfg.kind ~= "dynamics"
        error('banff:biasRegimeTask', ...
            'Closed decoder-to-encoder feedback requires a dynamics task.');
    end
    stateDimension = numel(loaded.result.data_information.mean);
    Pcpu = banff_model('create', stateDimension, stateDimension, cfg);
    baselineBias = single(loaded.result.best.B(:));
    if numel(baselineBias) ~= Pcpu.N_hidden
        error('banff:biasRegimeBiasSize', ...
            'The saved best bias does not match the reconstructed network.');
    end
    baselineDescription = sprintf('saved validation-selected bias: %s', modelFile);
end

if Pcpu.N_input ~= Pcpu.N_output
    error('banff:biasRegimeDimensions', ...
        'Closed-loop feedback requires equal input and output dimensions.');
end
if randomInputTrials < 1 || randomDriveTimeSeconds <= 0 || ...
        closedLoopTimeSeconds <= closedLoopWarmupSeconds || ...
        randomInputCorrelationTime <= 0
    error('banff:biasRegimeSettings', ...
        'Trial counts, durations, and correlation time must be positive.');
end

fprintf('Bias-only recurrent-regime exploration\n');
fprintf('Task: %s | N: %d | rank: %d | seed: %d\n', ...
    char(cfg.task), Pcpu.N_hidden, Pcpu.N_recurrent, cfg.seed);
fprintf('Encoder/recurrent/decoder gains: %.6g / %.6g / %.6g\n', ...
    cfg.encoder_gain, cfg.recurrent_gain, cfg.decoder_gain);
fprintf('Baseline: %s\n\n', baselineDescription);

%% Build reproducible spatial bias patterns
rng(randomSeed, 'twister');
patternNames = strings(0, 1);
patterns = zeros(Pcpu.N_hidden, 0, 'single');

if includeRandomPattern
    patternNames(end + 1, 1) = "random";
    patterns(:, end + 1) = standardize_pattern(single(randn(Pcpu.N_hidden, 1)));
end

if includeDominantModePatterns
    [dominantVector, dominantEigenvalue, eigsFlag] = ...
        dominant_recurrent_mode(Pcpu);
    fprintf('Dominant effective-current eigenvalue: %.6g %+.6gi (flag %d)\n', ...
        real(dominantEigenvalue), imag(dominantEigenvalue), eigsFlag);
    realPattern = standardize_pattern(single(real(dominantVector)));
    patternNames(end + 1, 1) = "dominant real";
    patterns(:, end + 1) = realPattern;
    if std(imag(dominantVector)) > 1e-8
        imaginaryPattern = standardize_pattern(single(imag(dominantVector)));
        patternNames(end + 1, 1) = "dominant imaginary";
        patterns(:, end + 1) = imaginaryPattern;
    end
end

%% Assemble candidates without duplicate zero-scale rows
candidatePattern = strings(0, 1);
candidateOffset = zeros(0, 1);
candidateScale = zeros(0, 1);
candidatePatternIndex = zeros(0, 1);

for offset = biasMeanOffsetsMv
    candidatePattern(end + 1, 1) = "uniform";
    candidateOffset(end + 1, 1) = offset;
    candidateScale(end + 1, 1) = 0;
    candidatePatternIndex(end + 1, 1) = 0;
end
for patternIndex = 1:numel(patternNames)
    for scale = biasPatternScalesMv
        for offset = biasMeanOffsetsMv
            candidatePattern(end + 1, 1) = patternNames(patternIndex);
            candidateOffset(end + 1, 1) = offset;
            candidateScale(end + 1, 1) = scale;
            candidatePatternIndex(end + 1, 1) = patternIndex;
        end
    end
end

candidateCount = numel(candidateOffset);
fprintf('Screening %d bias configurations using %d matched random-input trials.\n', ...
    candidateCount, randomInputTrials);

%% Generate one matched random input prefix for every candidate
driveSteps = round(randomDriveTimeSeconds / double(cfg.dt));
closedLoopSteps = round(closedLoopTimeSeconds / double(cfg.dt));
warmupSteps = round(closedLoopWarmupSeconds / double(cfg.dt));
if warmupSteps >= closedLoopSteps
    error('banff:biasRegimeWarmup', ...
        'Closed-loop warmup must be shorter than the closed-loop interval.');
end
teacherInputs = correlated_random_inputs(Pcpu.N_input, driveSteps, ...
    randomInputTrials, double(cfg.dt), randomInputCorrelationTime, ...
    randomInputRms, randomSeed + 1);

%% Screen recurrence-on conditions
screen = repmat(empty_measurement(), candidateCount, 1);
storedTraces = cell(candidateCount, 1);
for candidateIndex = 1:candidateCount
    bias = baselineBias + single(candidateOffset(candidateIndex));
    if candidatePatternIndex(candidateIndex) > 0
        bias = bias + single(candidateScale(candidateIndex)) .* ...
            patterns(:, candidatePatternIndex(candidateIndex));
    end
    [screen(candidateIndex), storedTraces{candidateIndex}] = simulate_regime( ...
        Pcpu, bias, teacherInputs, driveSteps, closedLoopSteps, warmupSteps, true);
    fprintf(['%3d/%3d %-18s offset %+5.2f mV, scale %4.2f mV: ' ...
        'rate %6.2f Hz, rec/enc %7.3f\n'], ...
        candidateIndex, candidateCount, char(candidatePattern(candidateIndex)), ...
        candidateOffset(candidateIndex), candidateScale(candidateIndex), ...
        screen(candidateIndex).rate_hz, ...
        screen(candidateIndex).recurrent_to_encoder_rms);
end

encoderRms = reshape([screen.encoder_rms_mv], [], 1);
recurrentRms = reshape([screen.recurrent_rms_mv], [], 1);
recurrentToEncoder = reshape([screen.recurrent_to_encoder_rms], [], 1);
grossEncoderRms = reshape([screen.gross_encoder_rms_mv], [], 1);
grossRecurrentRms = reshape([screen.gross_recurrent_rms_mv], [], 1);
netToGrossEncoder = reshape([screen.net_to_gross_encoder_rms], [], 1);
netToGrossRecurrent = reshape([screen.net_to_gross_recurrent_rms], [], 1);
rateHz = reshape([screen.rate_hz], [], 1);
adaptationRms = reshape([screen.adaptation_rms_mv], [], 1);
outputRms = reshape([screen.output_rms], [], 1);
activeNeuronPercent = reshape([screen.active_neuron_percent], [], 1);
finiteState = reshape([screen.finite], [], 1);
actualBiasMean = reshape([screen.bias_mean_mv], [], 1);
actualBiasStd = reshape([screen.bias_std_mv], [], 1);

plausible = finiteState & rateHz >= minimumAcceptedRateHz & ...
    rateHz <= maximumAcceptedRateHz & encoderRms >= minimumEncoderRmsMv;
recurrentDominant = plausible & ...
    recurrentToEncoder >= recurrentDominanceThreshold;

screenResults = table((1:candidateCount).', candidatePattern, ...
    candidateOffset, candidateScale, actualBiasMean, actualBiasStd, ...
    encoderRms, recurrentRms, recurrentToEncoder, grossEncoderRms, ...
    grossRecurrentRms, netToGrossEncoder, netToGrossRecurrent, rateHz, ...
    activeNeuronPercent, adaptationRms, outputRms, plausible, ...
    recurrentDominant, finiteState, 'VariableNames', ...
    {'Candidate','Pattern','MeanOffsetMv','PatternScaleMv','ActualBiasMeanMv', ...
    'ActualBiasStdMv','EncoderRmsMv','RecurrentRmsMv', ...
    'RecurrentToEncoderRms','GrossEncoderRmsMv','GrossRecurrentRmsMv', ...
    'NetToGrossEncoderRms','NetToGrossRecurrentRms', ...
    'MeanRateHz','ActiveNeuronPercent', ...
    'AdaptationRmsMv','DecoderOutputRms','PlausibleRateAndDrive', ...
    'RecurrentDominant','NumericallyFinite'});

[~, screenOrder] = sort(recurrentToEncoder, 'descend');
fprintf('\nScreening results, ordered by recurrent/encoder RMS\n');
disp(screenResults(screenOrder, :));

%% Run recurrence-off controls for the strongest viable candidates
eligible = find(plausible);
if isempty(eligible)
    eligible = find(finiteState);
end
[~, eligibleOrder] = sort(recurrentToEncoder(eligible), 'descend');
ablationIndices = eligible(eligibleOrder(1:min(maximumAblationCandidates, ...
    numel(eligibleOrder))));

% Always include the closest available representation of the baseline bias.
[~, baselineIndex] = min(abs(candidateOffset) + abs(candidateScale));
ablationIndices = unique([baselineIndex; ablationIndices(:)], 'stable');

ablationRateHz = nan(candidateCount, 1);
ablationOutputDifference = nan(candidateCount, 1);
ablationRateDifference = nan(candidateCount, 1);
ablationTraces = cell(candidateCount, 1);

fprintf('\nRunning recurrence-off controls for %d candidates.\n', ...
    numel(ablationIndices));
for localIndex = 1:numel(ablationIndices)
    candidateIndex = ablationIndices(localIndex);
    bias = baselineBias + single(candidateOffset(candidateIndex));
    if candidatePatternIndex(candidateIndex) > 0
        bias = bias + single(candidateScale(candidateIndex)) .* ...
            patterns(:, candidatePatternIndex(candidateIndex));
    end
    [ablated, ablationTraces{candidateIndex}] = simulate_regime( ...
        Pcpu, bias, teacherInputs, driveSteps, closedLoopSteps, warmupSteps, false);
    ablationRateHz(candidateIndex) = ablated.rate_hz;
    ablationRateDifference(candidateIndex) = abs( ...
        screen(candidateIndex).rate_hz - ablated.rate_hz) ./ ...
        max(screen(candidateIndex).rate_hz, eps);
    ablationOutputDifference(candidateIndex) = normalized_trace_difference( ...
        storedTraces{candidateIndex}.output, ...
        ablationTraces{candidateIndex}.output);
end

causallyRecurrent = recurrentDominant & ...
    ablationOutputDifference >= minimumAblationOutputDifference;

ablationResults = addvars(screenResults(ablationIndices, :), ...
    ablationRateHz(ablationIndices), ...
    ablationRateDifference(ablationIndices), ...
    ablationOutputDifference(ablationIndices), ...
    causallyRecurrent(ablationIndices), 'NewVariableNames', ...
    {'AblatedRateHz','RelativeRateChange','NormalizedOutputDifference', ...
    'RecurrentDominantAndCausal'});
[~, ablationOrder] = sort(ablationResults.NormalizedOutputDifference, 'descend');
ablationResults = ablationResults(ablationOrder, :);

fprintf('\nRecurrence-off controls\n');
disp(ablationResults);

%% Select and state the strongest conclusion supported by this search
qualifying = find(causallyRecurrent);
if ~isempty(qualifying)
    score = recurrentToEncoder(qualifying) .* ...
        ablationOutputDifference(qualifying);
    [~, bestLocal] = max(score);
    bestIndex = qualifying(bestLocal);
    fprintf(['\nRESULT: the fixed network admits a bias configuration with ' ...
        'recurrent RMS greater than encoder RMS and a recurrence-dependent ' ...
        'closed-loop trajectory.\n']);
else
    tested = find(isfinite(ablationOutputDifference));
    if isempty(tested)
        [~, bestIndex] = max(recurrentToEncoder);
    else
        score = recurrentToEncoder(tested) .* ...
            max(ablationOutputDifference(tested), eps);
        [~, bestLocal] = max(score);
        bestIndex = tested(bestLocal);
    end
    fprintf(['\nRESULT: no candidate met all configured recurrence-dominance, ' ...
        'rate, encoder-drive and ablation criteria. This is evidence only ' ...
        'within the searched bias family and range, not proof of impossibility.\n']);
end

fprintf(['Selected candidate %d: %s, offset %+.3g mV, scale %.3g mV; ' ...
    'encoder RMS %.4g mV, recurrent RMS %.4g mV, ratio %.4g, rate %.4g Hz.\n'], ...
    bestIndex, char(candidatePattern(bestIndex)), candidateOffset(bestIndex), ...
    candidateScale(bestIndex), encoderRms(bestIndex), recurrentRms(bestIndex), ...
    recurrentToEncoder(bestIndex), rateHz(bestIndex));
if isfinite(ablationOutputDifference(bestIndex))
    fprintf('Recurrence-off normalized output difference: %.4g.\n', ...
        ablationOutputDifference(bestIndex));
end
fprintf(['This diagnostic establishes accessibility, not learnability or target-' ...
    'task performance. Confirm any selected regime by training and held-out ' ...
    'recurrent ablation before changing publication settings.\n']);

%% Summary figures
figure('Color', 'w');
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
currentPlot = isfinite(encoderRms) & isfinite(recurrentRms) & isfinite(rateHz);
scatter(encoderRms(currentPlot), recurrentRms(currentPlot), 48, ...
    rateHz(currentPlot), 'filled');
hold on;
limit = max([encoderRms(currentPlot); recurrentRms(currentPlot)]);
if isempty(limit) || ~isfinite(limit) || limit <= 0
    limit = 1;
end
plot([0 limit], [0 limit], 'k--', 'LineWidth', 1);
hold off;
axis square;
grid on;
xlabel('Closed-loop encoder RMS (mV)');
ylabel('Closed-loop recurrent RMS (mV)');
title('Signed-net current magnitudes');
colorbar;

nexttile;
grossPlot = isfinite(grossEncoderRms) & isfinite(grossRecurrentRms) & ...
    isfinite(rateHz);
scatter(grossEncoderRms(grossPlot), grossRecurrentRms(grossPlot), 48, ...
    rateHz(grossPlot), 'filled');
hold on;
grossLimit = max([grossEncoderRms(grossPlot); grossRecurrentRms(grossPlot)]);
if isempty(grossLimit) || ~isfinite(grossLimit) || grossLimit <= 0
    grossLimit = 1;
end
plot([0 grossLimit], [0 grossLimit], 'k--', 'LineWidth', 1);
hold off;
axis square;
grid on;
xlabel('Gross encoder afferent RMS (mV)');
ylabel('Gross recurrent afferent RMS (mV)');
title('Absolute afferents before cancellation');
colorbar;

nexttile;
ratioPlot = isfinite(recurrentToEncoder) & isfinite(rateHz);
scatter(recurrentToEncoder(ratioPlot), rateHz(ratioPlot), 48, ...
    candidateScale(ratioPlot), 'filled');
xline(recurrentDominanceThreshold, 'k--');
yline(minimumAcceptedRateHz, ':');
yline(maximumAcceptedRateHz, ':');
grid on;
xlabel('Recurrent / encoder RMS');
ylabel('Population firing rate (Hz)');
title('Dominance versus operating rate');
colorbar;

nexttile;
tested = isfinite(ablationOutputDifference) & isfinite(recurrentToEncoder) & ...
    isfinite(rateHz);
scatter(recurrentToEncoder(tested), ablationOutputDifference(tested), ...
    55, rateHz(tested), 'filled');
xline(recurrentDominanceThreshold, 'k--');
yline(minimumAblationOutputDifference, 'k--');
grid on;
xlabel('Recurrent / encoder RMS');
ylabel('Normalized output change after ablation');
title('Magnitude does not imply causal use');
colorbar;

nexttile;
hold on;
uniquePatterns = unique(candidatePattern, 'stable');
for patternIndex = 1:numel(uniquePatterns)
    selected = candidatePattern == uniquePatterns(patternIndex) & ...
        isfinite(recurrentToEncoder);
    scatter(candidateOffset(selected), recurrentToEncoder(selected), 48, ...
        candidateScale(selected), 'filled', ...
        'DisplayName', char(uniquePatterns(patternIndex)));
end
hold off;
yline(recurrentDominanceThreshold, 'k--', 'HandleVisibility', 'off');
grid on;
xlabel('Mean bias offset (mV)');
ylabel('Recurrent / encoder RMS');
title('Bias pattern and offset search');
legend('Location', 'best');

nexttile;
scatter(netToGrossEncoder(ratioPlot), netToGrossRecurrent(ratioPlot), ...
    48, rateHz(ratioPlot), 'filled');
hold on; plot([0 1], [0 1], 'k--'); hold off;
grid on; axis square;
xlabel('Encoder net / gross RMS');
ylabel('Recurrent net / gross RMS');
title('Afferent cancellation');
colorbar;

%% Detailed trace for the selected candidate
selectedTrace = storedTraces{bestIndex};
figure('Color', 'w');
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
plot(selectedTrace.time_s, selectedTrace.encoder_rms_by_step, ...
    'LineWidth', 1.1);
hold on;
plot(selectedTrace.time_s, selectedTrace.recurrent_rms_by_step, ...
    'LineWidth', 1.1);
hold off;
grid on;
xlabel('Closed-loop time (s)');
ylabel('Per-step RMS current (mV)');
title('Current magnitudes');
legend({'Encoder','Recurrent'}, 'Location', 'best');

nexttile;
plot(selectedTrace.time_s, selectedTrace.gross_encoder_rms_by_step, ...
    'LineWidth', 1.1);
hold on;
plot(selectedTrace.time_s, selectedTrace.gross_recurrent_rms_by_step, ...
    'LineWidth', 1.1);
hold off;
grid on;
xlabel('Closed-loop time (s)');
ylabel('Per-step gross afferent RMS (mV)');
title('Absolute afferents before cancellation');
legend({'Encoder gross','Recurrent gross'}, 'Location', 'best');

nexttile;
plot(selectedTrace.time_s, selectedTrace.rate_hz_by_step, 'LineWidth', 1.1);
grid on;
xlabel('Closed-loop time (s)');
ylabel('Population firing rate (Hz)');
title('Closed-loop activity');

nexttile;
plot(selectedTrace.time_s, selectedTrace.adaptation_rms_by_step, ...
    'LineWidth', 1.1);
grid on;
xlabel('Closed-loop time (s)');
ylabel('Adaptation RMS (mV)');
title('Adaptation state');

nexttile;
plot(selectedTrace.time_s, squeeze(selectedTrace.output(:, :, 1)).', ...
    'LineWidth', 1.0);
hold on;
if ~isempty(ablationTraces{bestIndex})
    plot(ablationTraces{bestIndex}.time_s, ...
        squeeze(ablationTraces{bestIndex}.output(:, :, 1)).', '--', ...
        'LineWidth', 0.9);
end
hold off;
grid on;
xlabel('Closed-loop time (s)');
ylabel('Normalized decoder output');
title('Recurrence-on and recurrence-off trajectories');

nexttile;
plot(selectedTrace.time_s, selectedTrace.encoder_net_to_gross_by_step, ...
    'LineWidth', 1.1);
hold on;
plot(selectedTrace.time_s, selectedTrace.recurrent_net_to_gross_by_step, ...
    'LineWidth', 1.1);
hold off;
grid on;
xlabel('Closed-loop time (s)');
ylabel('Net / gross RMS');
title('Time-resolved afferent cancellation');
legend({'Encoder','Recurrent'}, 'Location', 'best');

%% Local functions
function inputs = correlated_random_inputs(dimension, steps, trials, dt, ...
        correlationTime, targetRms, seed)
%CORRELATED_RANDOM_INPUTS Generate unit-variance Ornstein-Uhlenbeck-like drive.
rng(seed, 'twister');
decay = exp(-dt / correlationTime);
innovationScale = sqrt(max(0, 1 - decay^2));
inputs = zeros(dimension, steps, trials, 'single');
state = single(randn(dimension, trials));
for step = 1:steps
    state = single(decay) .* state + single(innovationScale) .* ...
        single(randn(dimension, trials));
    inputs(:, step, :) = reshape(state, dimension, 1, trials);
end
observedRms = sqrt(mean(double(inputs(:)).^2));
inputs = single(targetRms) .* inputs ./ single(max(observedRms, eps));
end

function output = standardize_pattern(input)
%STANDARDIZE_PATTERN Produce a bounded zero-mean, unit-SD bias pattern.
output = single(input(:));
output = output - mean(output);
scale = std(output);
if ~isfinite(scale) || scale <= 0
    error('banff:biasRegimePattern', ...
        'A proposed spatial bias pattern has zero or invalid variance.');
end
output = output ./ scale;
output = min(max(output, single(-3)), single(3));
output = output - mean(output);
output = output ./ std(output);
end

function [vector, eigenvalue, flag] = dominant_recurrent_mode(P)
%DOMINANT_RECURRENT_MODE Find the largest-magnitude mode without dense W.
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

function [measurement, trace] = simulate_regime(Pcpu, bias, teacherInputs, ...
        driveSteps, closedLoopSteps, warmupSteps, recurrenceOn)
%SIMULATE_REGIME Run a matched random-prefix then autonomous feedback trial.
PdrivenCpu = Pcpu;
PdrivenCpu.B = single(bias(:));
Pdriven = banff_model('gpu', PdrivenCpu);

trialCount = size(teacherInputs, 3);
state = initial_diagnostic_state(Pdriven, trialCount);
previousOutput = gpuArray.zeros(Pdriven.N_output, trialCount, 'single');
teacherInputsGpu = gpuArray(single(teacherInputs));

for step = 1:driveSteps
    inputSignal = reshape(teacherInputsGpu(:, step, :), ...
        Pdriven.N_input, trialCount);
    inputCurrent = Pdriven.W_in * (Pdriven.inputScale .* inputSignal);
    [state, ~] = banff_model('gpu_step', Pdriven, state, inputCurrent, false);
    previousOutput = Pdriven.W_out * state.r;
end

% Both members of an ablation pair have exactly the same state at handoff.
% Recurrence is removed only after the common random-input prefix.
if recurrenceOn
    P = Pdriven;
else
    PclosedCpu = PdrivenCpu;
    if PclosedCpu.recurrent_mode == "low_rank"
        PclosedCpu.recurrentGain = single(0);
        PclosedCpu.self_coupling(:) = single(0);
    else
        PclosedCpu.W_recurrent = sparse([], [], [], ...
            PclosedCpu.N_hidden, PclosedCpu.N_hidden);
    end
    P = banff_model('gpu', PclosedCpu);
end

analysisSteps = closedLoopSteps - warmupSteps;
encoderRmsByStep = gpuArray.zeros(analysisSteps, 1, 'single');
recurrentRmsByStep = gpuArray.zeros(analysisSteps, 1, 'single');
grossEncoderRmsByStep = gpuArray.zeros(analysisSteps, 1, 'single');
grossRecurrentRmsByStep = gpuArray.zeros(analysisSteps, 1, 'single');
adaptationRmsByStep = gpuArray.zeros(analysisSteps, 1, 'single');
rateByStep = gpuArray.zeros(analysisSteps, 1, 'single');
outputTrace = gpuArray.zeros(P.N_output, analysisSteps, trialCount, 'single');
active = gpuArray.false(P.N_hidden, trialCount);
analysisIndex = 0;

for step = 1:closedLoopSteps
    inputCurrent = P.W_in * (P.inputScale .* previousOutput);
    recurrentCurrent = recurrent_current_for_diagnostic(P, state.r);
    grossInputCurrent = abs(P.W_in) * (P.inputScale .* abs(previousOutput));
    grossRecurrentCurrent = gross_recurrent_for_diagnostic(P, state.r);
    [state, spike] = banff_model('gpu_step', P, state, inputCurrent, false);
    previousOutput = P.W_out * state.r;

    if step > warmupSteps
        analysisIndex = analysisIndex + 1;
        encoderRmsByStep(analysisIndex) = sqrt(mean(inputCurrent.^2, 'all'));
        recurrentRmsByStep(analysisIndex) = ...
            sqrt(mean(recurrentCurrent.^2, 'all'));
        grossEncoderRmsByStep(analysisIndex) = ...
            sqrt(mean(grossInputCurrent.^2, 'all'));
        grossRecurrentRmsByStep(analysisIndex) = ...
            sqrt(mean(grossRecurrentCurrent.^2, 'all'));
        adaptationRmsByStep(analysisIndex) = sqrt(mean(state.w.^2, 'all'));
        rateByStep(analysisIndex) = sum(single(spike), 'all') ./ ...
            single(P.N_hidden * trialCount * double(P.dt));
        outputTrace(:, analysisIndex, :) = reshape(previousOutput, ...
            P.N_output, 1, trialCount);
        active = active | spike;
    end
end

trace = struct();
trace.time_s = (1:analysisSteps).' .* double(P.dt);
trace.encoder_rms_by_step = double(gather(encoderRmsByStep));
trace.recurrent_rms_by_step = double(gather(recurrentRmsByStep));
trace.gross_encoder_rms_by_step = double(gather(grossEncoderRmsByStep));
trace.gross_recurrent_rms_by_step = double(gather(grossRecurrentRmsByStep));
trace.encoder_net_to_gross_by_step = trace.encoder_rms_by_step ./ ...
    max(trace.gross_encoder_rms_by_step, realmin);
trace.recurrent_net_to_gross_by_step = trace.recurrent_rms_by_step ./ ...
    max(trace.gross_recurrent_rms_by_step, realmin);
trace.adaptation_rms_by_step = double(gather(adaptationRmsByStep));
trace.rate_hz_by_step = double(gather(rateByStep));
trace.output = double(gather(outputTrace));

measurement = empty_measurement();
measurement.encoder_rms_mv = sqrt(mean(trace.encoder_rms_by_step.^2));
measurement.recurrent_rms_mv = sqrt(mean(trace.recurrent_rms_by_step.^2));
measurement.recurrent_to_encoder_rms = measurement.recurrent_rms_mv ./ ...
    max(measurement.encoder_rms_mv, eps);
measurement.gross_encoder_rms_mv = ...
    sqrt(mean(trace.gross_encoder_rms_by_step.^2));
measurement.gross_recurrent_rms_mv = ...
    sqrt(mean(trace.gross_recurrent_rms_by_step.^2));
measurement.net_to_gross_encoder_rms = measurement.encoder_rms_mv ./ ...
    max(measurement.gross_encoder_rms_mv, eps);
measurement.net_to_gross_recurrent_rms = measurement.recurrent_rms_mv ./ ...
    max(measurement.gross_recurrent_rms_mv, eps);
measurement.adaptation_rms_mv = sqrt(mean(trace.adaptation_rms_by_step.^2));
measurement.rate_hz = mean(trace.rate_hz_by_step);
measurement.output_rms = sqrt(mean(trace.output(:).^2));
measurement.active_neuron_percent = 100 .* double(mean(gather(active), 'all'));
measurement.bias_mean_mv = mean(double(bias));
measurement.bias_std_mv = std(double(bias));
numericValues = [trace.encoder_rms_by_step; trace.recurrent_rms_by_step; ...
    trace.gross_encoder_rms_by_step; trace.gross_recurrent_rms_by_step; ...
    trace.adaptation_rms_by_step; trace.rate_hz_by_step; trace.output(:)];
measurement.finite = all(isfinite(numericValues));
end

function state = initial_diagnostic_state(P, batchSize)
%INITIAL_DIAGNOSTIC_STATE Match BANFF's production zero-state convention.
state = struct();
state.u = repmat(P.restingVoltage, P.N_hidden, batchSize);
state.w = gpuArray.zeros(P.N_hidden, batchSize, 'single');
state.x = gpuArray.zeros(P.N_hidden, batchSize, 'single');
state.r = gpuArray.zeros(P.N_hidden, batchSize, 'single');
state.epsilonVoltage = gpuArray.zeros(P.N_hidden, batchSize, 'single');
state.epsilonAdaptation = gpuArray.zeros(P.N_hidden, batchSize, 'single');
state.eligibilityRise = gpuArray.zeros(P.N_hidden, batchSize, 'single');
state.eligibilityDecay = gpuArray.zeros(P.N_hidden, batchSize, 'single');
end

function current = recurrent_current_for_diagnostic(P, filteredSpikes)
%RECURRENT_CURRENT_FOR_DIAGNOSTIC Reproduce BANFF's production current exactly.
if P.recurrent_mode == "full_rank"
    current = single(P.W_recurrent * double(filteredSpikes));
else
    latent = P.W_feedback * filteredSpikes;
    current = P.recurrentGain .* (P.recurrent_expansion * latent) ...
        - P.self_coupling .* filteredSpikes;
end
end

function gross = gross_recurrent_for_diagnostic(P, filteredSpikes)
%GROSS_RECURRENT_FOR_DIAGNOSTIC Sum absolute recurrent afferents exactly.
if P.recurrent_mode == "full_rank"
    gross = single(abs(P.W_recurrent) * double(filteredSpikes));
else
    gross = P.recurrentGain .* (P.recurrent_expansion * ...
        (abs(P.W_feedback) * filteredSpikes)) ...
        - abs(P.self_coupling) .* filteredSpikes;
    gross = max(gross, single(0));
end
end

function value = normalized_trace_difference(reference, comparison)
%NORMALIZED_TRACE_DIFFERENCE RMS trajectory change relative to signal RMS.
difference = reference - comparison;
referenceCentered = reference - mean(reference, 2);
value = sqrt(mean(difference(:).^2)) ./ ...
    max(sqrt(mean(referenceCentered(:).^2)), eps);
end

function output = empty_measurement()
%EMPTY_MEASUREMENT Define a consistent scalar result schema.
output = struct('encoder_rms_mv', NaN, 'recurrent_rms_mv', NaN, ...
    'recurrent_to_encoder_rms', NaN, 'gross_encoder_rms_mv', NaN, ...
    'gross_recurrent_rms_mv', NaN, 'net_to_gross_encoder_rms', NaN, ...
    'net_to_gross_recurrent_rms', NaN, 'adaptation_rms_mv', NaN, ...
    'rate_hz', NaN, 'output_rms', NaN, 'active_neuron_percent', NaN, ...
    'bias_mean_mv', NaN, 'bias_std_mv', NaN, 'finite', false);
end
