function search_report = banff_search_breast_cancer_recurrent_regime(userSettings)
%BANFF_SEARCH_BREAST_CANCER_RECURRENT_REGIME Find a consequential recurrent regime.
% This function searches encoder gain and a shared recurrent/decoder gain while
% preserving the current BANFF training and evaluation implementations.  Gain
% selection uses training and validation data only.  The held-out test set is
% evaluated once, after a winning regime has been frozen from validation data.
%
% The search is multi-fidelity.  Stage 1 cheaply screens a grid with a smaller
% network.  Stage 2 retrains the most promising gain pairs at the publication
% network size and, by default, across three independently initialised fixed
% networks.  A target-size regime is acceptable only if it simultaneously:
%   1. reaches the requested validation accuracy;
%   2. has similar signed-net recurrent and encoder RMS current magnitudes; and
%   3. loses a specified number of validation-accuracy percentage points when
%      the recurrent operator is set to zero without retraining.
%
% The full task evaluator is then run for the validation-selected winner.  It
% supplies the ordinary test metrics, plots, untrained/trained current
% comparison, firing-rate diagnostics and recurrent-ablation analysis.

if nargin < 1 || isempty(userSettings)
    userSettings = struct();
end
if ~isstruct(userSettings) || ~isscalar(userSettings)
    error('banff:regimeSearchSettingsType', ...
        'The optional settings input must be a scalar structure.');
end

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
addpath(fullfile(root, 'evaluation'));

%% User-editable search controls
settings = struct();

% Scientific acceptance criteria.  A ratio of one means equal population RMS
% signed-net currents.  The interval below treats agreement within 50% as a
% useful first-pass definition of "similar" rather than claiming exact equality.
settings.validation_accuracy_threshold_percent = 80;
settings.recurrent_to_encoder_ratio_interval = [2/3, 3/2];
settings.minimum_ablation_drop_percentage_points = 5;
settings.minimum_seed_pass_fraction = 2/3;

% Stage 1: inexpensive discovery.  Width-normalised fixed matrices make this a
% useful screen, but it is not treated as evidence about the final 32k model.
settings.screen_hidden_neurons = 4000;
settings.screen_epochs = 50;
settings.screen_seeds = 1;
settings.encoder_gains = [0.5, 1, 2, 4];
settings.shared_recurrent_decoder_gains = [0.05, 0.10, 0.15, 0.20, 0.30, 0.4, 0.5];

% Stage 2: target-size confirmation.  Increase confirmation_epochs if no regime
% reaches 80%; the default is intentionally shorter than the definitive 5000
% epoch breast-cancer run so that unsuitable regimes are rejected promptly.
settings.confirmation_hidden_neurons = 32000;
settings.confirmation_epochs = 1000;
settings.confirmation_seeds = 1:3;
settings.number_of_candidates_to_confirm = 3;

% Runtime and reporting controls.
settings.batch_size = 256;
settings.validate_every = 5;
settings.reuse_completed_models = true;
settings.run_full_test_evaluation = true;
settings.save_full_evaluation_figures = true;
% On ARC, set this slightly below the allocation wall time. Rerunning the same
% script/configuration resumes BANFF's matching checkpoint before continuing.
settings.checkpoint_hours = inf;
settings.output_directory = default_search_output_directory(root);

% A Live Script can override any of the defaults above by defining the scalar
% structure banff_regime_search_settings before invoking this file.
userSettingNames = fieldnames(userSettings);
for userSettingIndex = 1:numel(userSettingNames)
    settings.(userSettingNames{userSettingIndex}) = ...
        userSettings.(userSettingNames{userSettingIndex});
end
clear userSettings userSettingNames userSettingIndex;

validate_search_settings(settings);
if ~canUseGPU
    error('banff:regimeSearchNoGPU', ...
        'This search uses the BANFF GPU training path and requires a supported NVIDIA GPU.');
end
if exist(settings.output_directory, 'dir') ~= 7
    mkdir(settings.output_directory);
end
verify_mat_output_directory(settings.output_directory);

fprintf('\nBANFF breast-cancer recurrent-regime search\n');
fprintf(['Selection is validation-only. Test data are withheld until a target-size ', ...
    'winner has been frozen.\n']);
fprintf('Constraint: decoder gain = recurrent gain for every candidate.\n\n');
fprintf('Search outputs: %s\n\n', settings.output_directory);

%% Stage 1: screen the full gain grid
[encoderGrid, sharedGrid] = ndgrid(settings.encoder_gains, ...
    settings.shared_recurrent_decoder_gains);
screenCandidates = table((1:numel(encoderGrid)).', encoderGrid(:), sharedGrid(:), ...
    'VariableNames', {'Candidate','EncoderGain','SharedRecurrentDecoderGain'});

screenRuns = run_stage("screen", screenCandidates, ...
    settings.screen_hidden_neurons, settings.screen_epochs, ...
    settings.screen_seeds, settings);
if ~any(screenRuns.RunSucceeded)
    error('banff:regimeScreenFailed', ...
        'Every screening run failed. Inspect screenRuns.FailureMessage.');
end
screenSummary = summarise_candidates(screenRuns, settings);
screenSummary = rank_candidates(screenSummary);

fprintf('\nStage 1 per-run validation results\n');
disp(screenRuns);
fprintf('\nStage 1 validation summary (ranked)\n');
disp(screenSummary);
plot_search_summary(screenSummary, "Stage 1 screen");

%% Stage 2: confirm the strongest distinct gain pairs at the target width
confirmationCount = min(settings.number_of_candidates_to_confirm, ...
    height(screenSummary));
confirmationCandidates = screenSummary(1:confirmationCount, ...
    {'Candidate','EncoderGain','SharedRecurrentDecoderGain'});
confirmationCandidates.Candidate = (1:confirmationCount).';

confirmationRuns = run_stage("confirmation", confirmationCandidates, ...
    settings.confirmation_hidden_neurons, settings.confirmation_epochs, ...
    settings.confirmation_seeds, settings);
if ~any(confirmationRuns.RunSucceeded)
    error('banff:regimeConfirmationFailed', ...
        'Every target-size confirmation run failed. Inspect confirmationRuns.FailureMessage.');
end
confirmationSummary = summarise_candidates(confirmationRuns, settings);
confirmationSummary = rank_candidates(confirmationSummary);

fprintf('\nStage 2 per-run target-size validation results\n');
disp(confirmationRuns);
fprintf('\nStage 2 target-size validation summary (ranked)\n');
disp(confirmationSummary);
plot_search_summary(confirmationSummary, "Stage 2 target-size confirmation");

%% Freeze the winner from validation data, then evaluate the held-out test set
qualified = find(confirmationSummary.Acceptable, 1, 'first');
fullEvaluation = [];
if isempty(qualified)
    warning('banff:noAcceptableRecurrentRegime', [ ...
        'No target-size candidate met all predeclared validation criteria. ', ...
        'The held-out test set will not be opened. Inspect the validation table, ', ...
        'then extend the gain grid or training budget and rerun.']);
    winner = table();
else
    winner = confirmationSummary(qualified, :);
    fprintf('\nValidation-selected regime\n');
    disp(winner);

    if settings.run_full_test_evaluation
        winnerOverrides = common_overrides(settings.confirmation_hidden_neurons, ...
            settings.confirmation_epochs, settings, "confirmation");
        winnerOverrides.encoder_gain = single(winner.EncoderGain);
        winnerOverrides.recurrent_gain = ...
            single(winner.SharedRecurrentDecoderGain);
        winnerOverrides.decoder_gain = ...
            single(winner.SharedRecurrentDecoderGain);

        displayOptions = struct();
        displayOptions.save_figures = settings.save_full_evaluation_figures;
        displayOptions.output_directory = fullfile(settings.output_directory, ...
            'winner_test_evaluation');
        displayOptions.run_recurrent_ablation = true;
        displayOptions.assessment_split = "test";
        fullEvaluation = banff_evaluate_task("breast_cancer", ...
            settings.confirmation_seeds, "main", winnerOverrides, displayOptions);
    end
end

search_report = struct();
search_report.settings = settings;
search_report.screen_runs = screenRuns;
search_report.screen_summary = screenSummary;
search_report.confirmation_runs = confirmationRuns;
search_report.confirmation_summary = confirmationSummary;
search_report.validation_selected_winner = winner;
search_report.full_test_evaluation = fullEvaluation;
search_report.recommendations = make_recommendations( ...
    confirmationSummary, winner, settings, screenRuns, confirmationRuns);
search_report.selection_statement = [ ...
    "All gain tuning and acceptance decisions used validation data only. ", ...
    "The held-out test evaluation was run only after the winner was frozen."];

reportFile = fullfile(settings.output_directory, 'regime_search_report.mat');
saved_report = search_report;
% Graphics handles are session-specific and can make a MAT file unnecessarily
% large. The numerical evaluation remains in the saved report; figures are
% already visible and, when requested, exported by the full evaluator.
if isstruct(saved_report.full_test_evaluation) && ...
        isfield(saved_report.full_test_evaluation, 'figures')
    saved_report.full_test_evaluation = ...
        rmfield(saved_report.full_test_evaluation, 'figures');
end
save(reportFile, 'saved_report', '-v7.3');
writetable(screenRuns, fullfile(settings.output_directory, 'screen_runs.csv'));
writetable(screenSummary, fullfile(settings.output_directory, 'screen_summary.csv'));
writetable(confirmationRuns, ...
    fullfile(settings.output_directory, 'confirmation_runs.csv'));
writetable(confirmationSummary, ...
    fullfile(settings.output_directory, 'confirmation_summary.csv'));
fprintf('\nSaved search report: %s\n', reportFile);
fprintf('\nRecommendations and interpretation\n');
for index = 1:numel(search_report.recommendations)
    fprintf('%s\n', char(search_report.recommendations(index)));
end

end

%% Local implementation helpers
function runs = run_stage(stage, candidates, hiddenNeurons, epochs, seeds, settings)
%RUN_STAGE Train and diagnose each candidate/seed without touching test data.
rowCount = height(candidates) * numel(seeds);
runs = initialise_run_table(rowCount);
row = 0;
for candidateIndex = 1:height(candidates)
    for seedIndex = 1:numel(seeds)
        row = row + 1;
        seed = seeds(seedIndex);
        encoderGain = candidates.EncoderGain(candidateIndex);
        sharedGain = candidates.SharedRecurrentDecoderGain(candidateIndex);
        overrides = common_overrides(hiddenNeurons, epochs, settings, stage);
        overrides.seed = seed;
        overrides.encoder_gain = single(encoderGain);
        overrides.recurrent_gain = single(sharedGain);
        overrides.decoder_gain = single(sharedGain);

        cfg = banff("config", "breast_cancer", overrides);
        fprintf(['\n%s candidate %d/%d, seed %d: N=%d, encoder=%.4g, ', ...
            'recurrent=decoder=%.4g\n'], upper(char(stage)), candidateIndex, ...
            height(candidates), seed, hiddenNeurons, encoderGain, sharedGain);

        runs.Stage(row) = string(stage);
        runs.Candidate(row) = candidateIndex;
        runs.Seed(row) = seed;
        runs.HiddenNeurons(row) = hiddenNeurons;
        runs.EpochBudget(row) = epochs;
        runs.EncoderGain(row) = encoderGain;
        runs.SharedRecurrentDecoderGain(row) = sharedGain;
        try
            [trained, elapsedSeconds, reused] = ...
                train_or_reuse(cfg, overrides, settings);
            diagnostics = validation_diagnostics(trained);
            firstEpoch = first_threshold_epoch(trained.history.validation_metric, ...
                settings.validation_accuracy_threshold_percent);
            if isfinite(firstEpoch) && ~reused
                % The trainer does not timestamp each validation pass. This is an
                % explicitly labelled linear estimate; total training time is exact.
                estimatedThresholdSeconds = elapsedSeconds * firstEpoch / epochs;
            else
                estimatedThresholdSeconds = NaN;
            end

            runs.TrainingSeconds(row) = elapsedSeconds;
            runs.ReusedCompletedModel(row) = reused;
            runs.FirstThresholdEpoch(row) = firstEpoch;
            runs.EstimatedSecondsToThreshold(row) = estimatedThresholdSeconds;
            runs.BestValidationAccuracyPercent(row) = double(trained.best.metric);
            runs.ValidationAccuracyPercent(row) = diagnostics.full_accuracy_percent;
            runs.AblatedValidationAccuracyPercent(row) = diagnostics.ablated_accuracy_percent;
            runs.AblationDropPercentagePoints(row) = diagnostics.ablation_drop_percentage_points;
            runs.EncoderRmsMv(row) = diagnostics.encoder_rms_mV;
            runs.RecurrentRmsMv(row) = diagnostics.recurrent_rms_mV;
            runs.RecurrentToEncoderRms(row) = diagnostics.recurrent_to_encoder_rms;
            runs.NetToGrossRecurrentRms(row) = diagnostics.net_to_gross_recurrent_rms;
            runs.GrossEncoderRmsMv(row) = diagnostics.gross_encoder_rms_mV;
            runs.GrossRecurrentRmsMv(row) = diagnostics.gross_recurrent_rms_mV;
            runs.DecoderContributionRms(row) = diagnostics.decoder_contribution_rms;
            runs.AdaptationRmsMv(row) = diagnostics.adaptation_rms_mV;
            runs.BiasDeviationRmsMv(row) = diagnostics.bias_deviation_rms_mV;
            runs.ActiveNeuronPercent(row) = diagnostics.active_neuron_percent;
            runs.RunAcceptable(row) = run_is_acceptable(diagnostics, settings);
            runs.RunSucceeded(row) = true;
            runs.FailureIdentifier(row) = "";
            runs.FailureMessage(row) = "";
        catch exception
            % A time-limit checkpoint is an expected interruption: stop so the
            % identical job can be resubmitted and resume it. Other isolated
            % candidate failures are retained in the comparison instead of
            % discarding all completed regimes.
            if strcmp(exception.identifier, 'banff:regimeSearchCheckpoint') || ...
                    strcmp(exception.identifier, 'banff:regimeSearchProvenance')
                rethrow(exception);
            end
            runs.RunSucceeded(row) = false;
            runs.FailureIdentifier(row) = string(exception.identifier);
            runs.FailureMessage(row) = string(exception.message);
            warning('banff:regimeCandidateFailed', ...
                'Candidate %d seed %d failed: %s', candidateIndex, seed, ...
                exception.message);
        end
    end
end
end

function overrides = common_overrides(hiddenNeurons, epochs, settings, stage)
overrides = struct();
overrides.N_hidden = hiddenNeurons;
overrides.N_recurrent = 10;
overrides.epochs = epochs;
overrides.batch_size = settings.batch_size;
overrides.validate_every = settings.validate_every;
overrides.checkpoint_hours = settings.checkpoint_hours;
overrides.output_directory = fullfile(settings.output_directory, char(stage));
overrides.method = "eprop";
overrides.recurrent_mode = "low_rank";
overrides.training_profile = "main";
end

function [trained, elapsedSeconds, reused] = train_or_reuse(cfg, overrides, settings)
reused = false;
elapsedSeconds = NaN;
if settings.reuse_completed_models && exist(cfg.model_file, 'file') == 2
    loaded = load(cfg.model_file, 'result');
    if isfield(loaded, 'result') && loaded.result.complete && ...
            strcmp(loaded.result.config.scientific_config_sha256, ...
            cfg.scientific_config_sha256) && ...
            loaded.result.config.seed == cfg.seed
        if ~isfield(loaded.result, 'provenance')
            error('banff:regimeSearchProvenance', ...
                'A reusable model lacks training-source provenance: %s', cfg.model_file);
        end
        banff_provenance("assert_training_compatible", loaded.result.provenance);
        trained = loaded.result;
        reused = true;
        fprintf('Reusing completed matching model: %s\n', cfg.model_file);
        return;
    end
end
timer = tic;
trained = banff("train", "breast_cancer", overrides);
elapsedSeconds = toc(timer);
if ~trained.complete
    error('banff:regimeSearchCheckpoint', [ ...
        'A time-limit checkpoint interrupted a search run. Disable the time ', ...
        'limit or resume the identical configuration before continuing.']);
end
end

function diagnostic = validation_diagnostics(trained)
%VALIDATION_DIAGNOSTICS Re-evaluate the best validation-selected bias vector.
cfg = trained.config;
[data, ~] = banff_data('static', cfg, trained.data_information);
P = banff_model('create', size(data.X_train, 1), size(data.Y_train, 1), cfg);
P.B = single(trained.best.B);

full = banff_eval('static', banff_model('gpu', P), ...
    data.X_validation, data.Y_validation, cfg, true);
statistics = banff_metrics('classification', full.output, data.Y_validation);
currents = banff_plot('static_current_magnitudes', P, ...
    data.X_validation, cfg);

ablatedP = P;
ablatedP.recurrentGain = single(0);
ablatedP.self_coupling(:) = single(0);
ablated = banff_eval('static', banff_model('gpu', ablatedP), ...
    data.X_validation, data.Y_validation, cfg, false);

diagnostic = struct();
diagnostic.full_accuracy_percent = double(statistics.accuracy_percent);
diagnostic.ablated_accuracy_percent = double(ablated.metric);
diagnostic.ablation_drop_percentage_points = ...
    diagnostic.full_accuracy_percent - diagnostic.ablated_accuracy_percent;
diagnostic.encoder_rms_mV = double(currents.aggregate.encoder_rms_mV);
diagnostic.recurrent_rms_mV = double(currents.aggregate.net_recurrent_rms_mV);
diagnostic.recurrent_to_encoder_rms = ...
    diagnostic.recurrent_rms_mV / max(diagnostic.encoder_rms_mV, realmin);
diagnostic.net_to_gross_recurrent_rms = ...
    double(currents.aggregate.net_to_gross_recurrent_rms);
diagnostic.gross_encoder_rms_mV = double(currents.aggregate.gross_encoder_rms_mV);
diagnostic.gross_recurrent_rms_mV = ...
    double(currents.aggregate.gross_recurrent_rms_mV);
diagnostic.decoder_contribution_rms = ...
    double(currents.aggregate.decoder_contribution_rms);
diagnostic.adaptation_rms_mV = double(currents.aggregate.adaptation_rms_mV);
diagnostic.bias_deviation_rms_mV = ...
    double(currents.aggregate.bias_deviation_rms_mV);
diagnostic.active_neuron_percent = double(full.neural_activity.active_fraction_percent);

if abs(double(cfg.recurrent_gain) - double(cfg.decoder_gain)) > ...
        10 * eps(max(abs(double(cfg.recurrent_gain)), 1))
    error('banff:regimeSearchGainConstraint', ...
        'The recurrent and decoder gains differ in a supposedly constrained run.');
end
end

function tf = run_is_acceptable(diagnostic, settings)
interval = settings.recurrent_to_encoder_ratio_interval;
tf = diagnostic.full_accuracy_percent >= ...
        settings.validation_accuracy_threshold_percent && ...
    diagnostic.recurrent_to_encoder_rms >= interval(1) && ...
    diagnostic.recurrent_to_encoder_rms <= interval(2) && ...
    diagnostic.ablation_drop_percentage_points >= ...
        settings.minimum_ablation_drop_percentage_points;
end

function epoch = first_threshold_epoch(metricHistory, threshold)
index = find(isfinite(metricHistory) & metricHistory >= threshold, 1, 'first');
if isempty(index), epoch = NaN; else, epoch = index; end
end

function summary = summarise_candidates(runs, settings)
candidates = unique(runs.Candidate, 'stable');
n = numel(candidates);
summary = table('Size', [n, 22], ...
    'VariableTypes', [repmat({'double'}, 1, 20), {'logical','double'}], ...
    'VariableNames', {'Candidate','EncoderGain','SharedRecurrentDecoderGain', ...
    'MeanValidationAccuracyPercent','SdValidationAccuracyPercent', ...
    'MeanAblationDropPercentagePoints','SdAblationDropPercentagePoints', ...
    'GeometricMeanRecurrentToEncoderRms','MeanEncoderRmsMv', ...
    'MeanRecurrentRmsMv','MeanGrossEncoderRmsMv','MeanGrossRecurrentRmsMv', ...
    'MeanNetToGrossRecurrentRms','MeanDecoderContributionRms', ...
    'MeanAdaptationRmsMv','MeanBiasDeviationRmsMv','MeanActiveNeuronPercent', ...
    'MeanTrainingSeconds','MeanFirstThresholdEpoch','SeedPassFraction', ...
    'Acceptable','ValidationScore'});
for index = 1:n
    rows = runs.Candidate == candidates(index);
    summary.Candidate(index) = candidates(index);
    summary.EncoderGain(index) = runs.EncoderGain(find(rows, 1));
    summary.SharedRecurrentDecoderGain(index) = ...
        runs.SharedRecurrentDecoderGain(find(rows, 1));
    summary.MeanValidationAccuracyPercent(index) = ...
        mean_finite(runs.ValidationAccuracyPercent(rows));
    summary.SdValidationAccuracyPercent(index) = ...
        std_finite(runs.ValidationAccuracyPercent(rows));
    summary.MeanAblationDropPercentagePoints(index) = ...
        mean_finite(runs.AblationDropPercentagePoints(rows));
    summary.SdAblationDropPercentagePoints(index) = ...
        std_finite(runs.AblationDropPercentagePoints(rows));
    ratios = runs.RecurrentToEncoderRms(rows);
    ratios = ratios(isfinite(ratios) & ratios > 0);
    if isempty(ratios)
        summary.GeometricMeanRecurrentToEncoderRms(index) = NaN;
    else
        summary.GeometricMeanRecurrentToEncoderRms(index) = exp(mean(log(ratios)));
    end
    summary.MeanEncoderRmsMv(index) = mean_finite(runs.EncoderRmsMv(rows));
    summary.MeanRecurrentRmsMv(index) = mean_finite(runs.RecurrentRmsMv(rows));
    summary.MeanGrossEncoderRmsMv(index) = ...
        mean_finite(runs.GrossEncoderRmsMv(rows));
    summary.MeanGrossRecurrentRmsMv(index) = ...
        mean_finite(runs.GrossRecurrentRmsMv(rows));
    summary.MeanNetToGrossRecurrentRms(index) = ...
        mean_finite(runs.NetToGrossRecurrentRms(rows));
    summary.MeanDecoderContributionRms(index) = ...
        mean_finite(runs.DecoderContributionRms(rows));
    summary.MeanAdaptationRmsMv(index) = ...
        mean_finite(runs.AdaptationRmsMv(rows));
    summary.MeanBiasDeviationRmsMv(index) = ...
        mean_finite(runs.BiasDeviationRmsMv(rows));
    summary.MeanActiveNeuronPercent(index) = ...
        mean_finite(runs.ActiveNeuronPercent(rows));
    summary.MeanTrainingSeconds(index) = mean_finite(runs.TrainingSeconds(rows));
    summary.MeanFirstThresholdEpoch(index) = ...
        mean_finite(runs.FirstThresholdEpoch(rows));
    summary.SeedPassFraction(index) = mean(runs.RunAcceptable(rows));

    ratio = summary.GeometricMeanRecurrentToEncoderRms(index);
    interval = settings.recurrent_to_encoder_ratio_interval;
    meanCriteria = summary.MeanValidationAccuracyPercent(index) >= ...
            settings.validation_accuracy_threshold_percent && ...
        ratio >= interval(1) && ratio <= interval(2) && ...
        summary.MeanAblationDropPercentagePoints(index) >= ...
            settings.minimum_ablation_drop_percentage_points;
    summary.Acceptable(index) = meanCriteria && ...
        summary.SeedPassFraction(index) >= settings.minimum_seed_pass_fraction;

    balancePenalty = 10 * abs(log2(max(ratio, realmin)));
    accuracyShortfall = max(0, settings.validation_accuracy_threshold_percent - ...
        summary.MeanValidationAccuracyPercent(index));
    ablationShortfall = max(0, settings.minimum_ablation_drop_percentage_points - ...
        summary.MeanAblationDropPercentagePoints(index));
    summary.ValidationScore(index) = ...
        summary.MeanValidationAccuracyPercent(index) + ...
        2 * summary.MeanAblationDropPercentagePoints(index) - ...
        balancePenalty - 2 * accuracyShortfall - 2 * ablationShortfall;
    if ~isfinite(summary.ValidationScore(index))
        summary.ValidationScore(index) = -Inf;
    end
end
end

function ranked = rank_candidates(summary)
% Acceptable regimes precede near misses; test data never enter this ordering.
rankTable = table(~summary.Acceptable, -summary.ValidationScore, ...
    'VariableNames', {'NotAcceptable','NegativeScore'});
[~, order] = sortrows(rankTable, {'NotAcceptable','NegativeScore'});
ranked = summary(order, :);
end

function T = initialise_run_table(n)
names = {'Stage','Candidate','Seed','HiddenNeurons','EpochBudget', ...
    'EncoderGain','SharedRecurrentDecoderGain','TrainingSeconds', ...
    'ReusedCompletedModel','FirstThresholdEpoch','EstimatedSecondsToThreshold', ...
    'BestValidationAccuracyPercent','ValidationAccuracyPercent', ...
    'AblatedValidationAccuracyPercent','AblationDropPercentagePoints', ...
    'EncoderRmsMv','RecurrentRmsMv','RecurrentToEncoderRms', ...
    'NetToGrossRecurrentRms','GrossEncoderRmsMv','GrossRecurrentRmsMv', ...
    'DecoderContributionRms','AdaptationRmsMv','BiasDeviationRmsMv', ...
    'ActiveNeuronPercent','RunAcceptable', ...
    'RunSucceeded','FailureIdentifier','FailureMessage'};
types = [{'string'}, repmat({'double'}, 1, 7), {'logical'}, ...
    repmat({'double'}, 1, 16), {'logical','logical','string','string'}];
T = table('Size', [n, numel(names)], 'VariableTypes', types, ...
    'VariableNames', names);
% Failed runs must not acquire plausible-looking zero metrics from table
% preallocation. All numerical measurements start missing until assigned.
numericNames = names(strcmp(types, 'double'));
for index = 1:numel(numericNames)
    T.(numericNames{index})(:) = NaN;
end
end

function plot_search_summary(summary, heading)
figure('Color', 'w');
tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
scatter(summary.GeometricMeanRecurrentToEncoderRms, ...
    summary.MeanValidationAccuracyPercent, 75, ...
    summary.SharedRecurrentDecoderGain, 'filled');
xline(1, 'k--'); grid on; colorbar;
xlabel('Recurrent / encoder signed-net RMS');
ylabel('Validation accuracy (%)');
title('Current balance and performance');

nexttile;
scatter(summary.GeometricMeanRecurrentToEncoderRms, ...
    summary.MeanAblationDropPercentagePoints, 75, ...
    summary.EncoderGain, 'filled');
xline(1, 'k--'); yline(0, 'k-'); grid on; colorbar;
xlabel('Recurrent / encoder signed-net RMS');
ylabel('Accuracy loss after ablation (points)');
title('Functional recurrent dependence');

nexttile;
bar(summary.ValidationScore); grid on;
xlabel('Ranked candidate'); ylabel('Validation-only score');
title(char(heading));
end

function recommendations = make_recommendations(summary, winner, settings, ...
        screenRuns, confirmationRuns)
%MAKE_RECOMMENDATIONS Translate the predeclared criteria into an auditable
% interpretation. These statements never inspect held-out test performance.
if isempty(winner)
    best = summary(1, :);
    recommendations = [ ...
        "No gain pair should yet replace the main configuration: none met all validation criteria."; ...
        string(sprintf(['Best validation-only near miss: encoder gain %.4g, ', ...
        'recurrent=decoder gain %.4g, accuracy %.2f%%, current ratio %.3f, ', ...
        'ablation drop %.2f percentage points.'], best.EncoderGain, ...
        best.SharedRecurrentDecoderGain, best.MeanValidationAccuracyPercent, ...
        best.GeometricMeanRecurrentToEncoderRms, ...
        best.MeanAblationDropPercentagePoints))];
    interval = settings.recurrent_to_encoder_ratio_interval;
    if best.GeometricMeanRecurrentToEncoderRms < interval(1)
        recommendations(end+1,1) = ...
            "Recurrence remains too weak relative to the encoder. Extend the shared " + ...
            "recurrent/decoder-gain grid upward or reduce encoder gain, then repeat validation.";
    elseif best.GeometricMeanRecurrentToEncoderRms > interval(2)
        recommendations(end+1,1) = ...
            "Recurrence is too strong relative to the encoder. Extend the shared " + ...
            "gain grid downward or increase encoder gain, then repeat validation.";
    end
    if best.MeanValidationAccuracyPercent < ...
            settings.validation_accuracy_threshold_percent
        recommendations(end+1,1) = ...
            "Accuracy did not reach threshold. Increase the confirmation epoch budget " + ...
            "before changing the acceptance threshold.";
    end
    if best.MeanAblationDropPercentagePoints < ...
            settings.minimum_ablation_drop_percentage_points
        recommendations(end+1,1) = ...
            "The validation ablation effect is too small. Do not describe the model as " + ...
            "functionally recurrence-dependent on the basis of current magnitude alone.";
    end
else
    recommendations = [ ...
        string(sprintf(['Validation supports encoder gain %.4g and shared recurrent/', ...
        'decoder gain %.4g: mean accuracy %.2f%%, recurrent/encoder RMS %.3f, ', ...
        'and mean ablation drop %.2f percentage points.'], winner.EncoderGain, ...
        winner.SharedRecurrentDecoderGain, winner.MeanValidationAccuracyPercent, ...
        winner.GeometricMeanRecurrentToEncoderRms, ...
        winner.MeanAblationDropPercentagePoints)); ...
        string(sprintf(['%.1f%% of confirmation seeds independently met all three ', ...
        'per-seed criteria.'], 100 * winner.SeedPassFraction)); ...
        "Use this as the candidate publication regime, subject to the frozen held-out test results printed above."; ...
        "Report both net and gross recurrent RMS: a low net/gross ratio indicates substantial excitatory-inhibitory cancellation."; ...
        "Treat the ablation threshold as a predeclared practical effect size, not a formal null-hypothesis significance test with only three seeds."];
end
recommendations(end+1,1) = string(sprintf( ...
    'Exact newly executed training time recorded in this run: %.2f hours.', ...
    (sum_finite(screenRuns.TrainingSeconds) + ...
    sum_finite(confirmationRuns.TrainingSeconds)) / 3600));
end

function value = mean_finite(values)
values = double(values(:));
values = values(isfinite(values));
if isempty(values), value = NaN; else, value = mean(values); end
end

function value = std_finite(values)
values = double(values(:));
values = values(isfinite(values));
if numel(values) < 2, value = NaN; else, value = std(values, 0); end
end

function value = sum_finite(values)
values = double(values(:));
values = values(isfinite(values));
if isempty(values), value = 0; else, value = sum(values); end
end

function directory = default_search_output_directory(repositoryRoot)
% Large v7.3 files can be exposed as zero-byte cloud placeholders when saved
% directly into a synchronised OneDrive tree. Use nonsynchronised local storage
% for this exploratory search on Windows; retain repository-relative storage on
% ARC/Linux, where the project directory is not a desktop sync folder.
repositoryRoot = char(repositoryRoot);
if ispc && contains(lower(repositoryRoot), 'onedrive')
    localBase = getenv('LOCALAPPDATA');
    if isempty(localBase)
        localBase = tempdir;
    end
    directory = fullfile(localBase, 'BANFF_SNN_pub', 'regime_search', ...
        'breast_cancer');
else
    directory = fullfile(repositoryRoot, 'outputs', 'regime_search', ...
        'breast_cancer');
end
end

function verify_mat_output_directory(directory)
% Fail before expensive training if the selected filesystem cannot round-trip
% a MATLAB v7.3 file. This exercises the same HDF5-backed format as model saves.
if exist(directory, 'dir') ~= 7
    mkdir(directory);
end
probeFile = [tempname(directory), '.mat'];
cleanup = onCleanup(@() delete_if_present(probeFile));
probe = struct('value', pi, 'created', char(datetime('now')));
try
    save(probeFile, 'probe', '-v7.3');
    information = dir(probeFile);
    loaded = load(probeFile, 'probe');
catch exception
    error('banff:regimeSearchOutputDirectory', ...
        ['Cannot safely write MATLAB v7.3 results to "%s": %s. ', ...
        'Choose a nonsynchronised local output_directory.'], ...
        directory, exception.message);
end
if isempty(information) || information.bytes == 0 || ...
        ~isfield(loaded, 'probe') || loaded.probe.value ~= probe.value
    error('banff:regimeSearchOutputDirectory', ...
        ['The output-directory v7.3 round-trip check failed for "%s". ', ...
        'Choose a nonsynchronised local output_directory.'], directory);
end
end

function delete_if_present(file)
if exist(file, 'file') == 2
    delete(file);
end
end

function validate_search_settings(settings)
if settings.validation_accuracy_threshold_percent <= 0 || ...
        settings.validation_accuracy_threshold_percent > 100
    error('banff:regimeSearchThreshold', ...
        'Validation accuracy threshold must lie in (0,100].');
end
interval = settings.recurrent_to_encoder_ratio_interval;
if numel(interval) ~= 2 || any(~isfinite(interval)) || ...
        any(interval <= 0) || interval(1) >= interval(2)
    error('banff:regimeSearchRatio', ...
        'The current-ratio interval must contain two increasing positive values.');
end
positiveIntegerFields = {'screen_hidden_neurons','screen_epochs', ...
    'confirmation_hidden_neurons','confirmation_epochs','batch_size', ...
    'validate_every','number_of_candidates_to_confirm'};
for index = 1:numel(positiveIntegerFields)
    value = settings.(positiveIntegerFields{index});
    if ~isscalar(value) || ~isfinite(value) || value < 1 || value ~= round(value)
        error('banff:regimeSearchSetting', '%s must be a positive integer.', ...
            positiveIntegerFields{index});
    end
end
if any(settings.encoder_gains <= 0) || ...
        any(settings.shared_recurrent_decoder_gains <= 0)
    error('banff:regimeSearchGain', 'All searched gains must be positive.');
end
if isempty(settings.screen_seeds) || isempty(settings.confirmation_seeds) || ...
        any(settings.screen_seeds < 1) || any(settings.confirmation_seeds < 1)
    error('banff:regimeSearchSeeds', 'Seed lists must contain positive integers.');
end
if any(settings.screen_seeds ~= round(settings.screen_seeds)) || ...
        any(settings.confirmation_seeds ~= round(settings.confirmation_seeds))
    error('banff:regimeSearchSeeds', 'Seed lists must contain positive integers.');
end
if ~isscalar(settings.checkpoint_hours) || isnan(settings.checkpoint_hours) || ...
        settings.checkpoint_hours <= 0
    error('banff:regimeSearchCheckpointHours', ...
        'checkpoint_hours must be positive or Inf.');
end
end
