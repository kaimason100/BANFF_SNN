function report = banff_evaluate_task(task, seeds, profile, overrides, displayOptions)
%BANFF_EVALUATE_TASK Evaluate one publication task and reproduce its diagnostics.
%   REPORT = BANFF_EVALUATE_TASK(TASK,SEEDS,PROFILE,OVERRIDES,DISPLAYOPTIONS)
%   is the shared implementation used by the task-specific Live Scripts in
%   this folder.  Evaluation itself goes through RUN_EXPERIMENT("test",...)
%   and therefore BANFF -> BANFF_TEST -> BANFF_EVAL -> BANFF_MODEL.  Plot-only
%   replay uses BANFF_PLOT and the same BANFF_MODEL reference equations.

if nargin < 2 || isempty(seeds), seeds = 1:3; end
if nargin < 3 || isempty(profile), profile = "main"; end
if nargin < 4 || isempty(overrides), overrides = struct(); end
if nargin < 5 || isempty(displayOptions), displayOptions = struct(); end

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
task = canonical_task_local(task);
options = default_display_options(root, task, displayOptions);

if ~any(options.assessment_split==["validation","test"])
    error('banff:evaluationSplit','assessment_split must be validation or test.');
end
fprintf('\nBANFF %s evaluation: %s (%s profile)\n', ...
    lower(assessment_label(options)),task,profile);
fprintf('Seeds: %s\n\n', mat2str(double(seeds(:).')));
if isempty(options.preloaded_results)
    if options.assessment_split~="test"
        error('banff:evaluationCompletedValidation', ...
            ['Completed-model evaluation loads the saved held-out result. ', ...
             'Validation injection is reserved for the checkpoint evaluator.']);
    end
    results=run_experiment("test",task,profile,seeds,overrides);
else
    results=options.preloaded_results;
end
results = results(:).';
audit = audit_seed_models(results, seeds);

kind = string(results(1).config.kind);
switch kind
    case "classification"
        seedTable = classification_table(results);
    case "regression"
        seedTable = regression_table(results);
    case "dynamics"
        seedTable = dynamics_table(results,options);
    otherwise
        error('banff:evaluationKind', 'Unsupported task kind "%s".', kind);
end
summaryTable = summarise_numeric_columns(seedTable);

fprintf('Per-seed %s results\n',lower(assessment_label(options)));
disp(seedTable);
fprintf('Across-seed mean and sample SD\n');
disp(summaryTable);
fprintf('Seed/model audit: PASS (%d distinct requested seeds and model identities).\n\n', ...
    numel(seeds));

if options.run_recurrent_ablation
    [recurrentAblation,ablationDetails]= ...
        evaluate_recurrent_ablation(results,kind,options);
    fprintf('Inference-time recurrent ablation (trained biases retained)\n');
    disp(recurrentAblation);
else
    recurrentAblation = table();
    ablationDetails=struct([]);
end

figures = gobjects(0);
figures(end+1) = plot_summary(seedTable, summaryTable, task, kind, options); %#ok<AGROW>
figures(end+1) = plot_training_history(results, task, kind, options); %#ok<AGROW>
figures(end+1) = plot_bias_swarm(results, task, options); %#ok<AGROW>
if kind == "classification"
    figures = [figures, plot_classification(results, task, options)]; %#ok<AGROW>
elseif kind == "regression"
    figures = [figures, plot_regression(results, task, options)]; %#ok<AGROW>
else
    figures = [figures, plot_dynamics(results, task, options)]; %#ok<AGROW>
end
figures = [figures, plot_activity(results, task, kind, options)]; %#ok<AGROW>
if kind=="dynamics"
    figures=[figures plot_dynamics_current_comparison(results,task,options)]; %#ok<AGROW>
end
if options.run_recurrent_ablation
    figures=[figures plot_recurrent_ablation( ...
        recurrentAblation,ablationDetails,task,kind,options)]; %#ok<AGROW>
end

reportedOptions=options;
reportedOptions.preloaded_results=[];
report = struct('task', task, 'kind', kind, 'profile', string(profile), ...
    'seeds', double(seeds(:).'), 'results', results, 'seed_table', seedTable, ...
    'summary_table', summaryTable, 'network_seed_audit', audit, ...
    'recurrent_ablation', recurrentAblation, ...
    'recurrent_ablation_details',ablationDetails, ...
    'figures', figures(isgraphics(figures)), 'display_options', reportedOptions);
end

function options = default_display_options(root, task, changes)
options = struct();
options.save_figures = false;
options.output_directory = fullfile(root, 'outputs', 'evaluation', char(task));
options.representative_initial_condition = 1;
options.max_bias_points = 2500;
options.max_current_points = 2500;
options.max_raster_neurons = 300;
options.max_image_examples = 16;
options.max_spike_samples = 8;
options.replay_static_spikes = true;
options.run_recurrent_ablation = true;
options.assessment_split = "test";
options.preloaded_results = [];
options.figure_visibility = "on";
names = fieldnames(changes);
for index = 1:numel(names)
    options.(names{index}) = changes.(names{index});
end
options.representative_initial_condition = max(1, round(options.representative_initial_condition));
options.assessment_split=lower(string(options.assessment_split));
if options.save_figures && exist(options.output_directory, 'dir') ~= 7
    mkdir(options.output_directory);
end
end

function task = canonical_task_local(task)
task = lower(replace(replace(string(task), "-", "_"), " ", "_"));
switch task
    case {"bc", "breastcancer"}, task = "breast_cancer";
    case {"afromnist", "afro_mnist", "vai"}, task = "afro_mnist_vai";
    case {"car", "car_price"}, task = "toyota";
    case {"sprott", "sprotts"}, task = "sprott_s";
end
end

function audit = audit_seed_models(results, requestedSeeds)
actualSeeds = arrayfun(@(R) double(R.config.seed), results);
if numel(unique(double(requestedSeeds))) ~= numel(requestedSeeds)
    error('banff:evaluationDuplicateSeeds', 'The requested seed list contains duplicates.');
end
if ~isequal(actualSeeds(:), double(requestedSeeds(:)))
    error('banff:evaluationSeedMismatch', 'Loaded model seeds do not match requested seeds.');
end
modelFiles = strings(numel(results), 1);
scienceHashes = strings(numel(results), 1);
checkpointHashes = strings(numel(results), 1);
for index = 1:numel(results)
    modelFiles(index) = string(results(index).config.model_file);
    scienceHashes(index) = string(results(index).config.scientific_config_sha256);
    checkpointHashes(index) = string(results(index).config.checkpoint_config_sha256);
end
if numel(unique(modelFiles)) ~= numel(modelFiles)
    error('banff:evaluationDuplicateModels', 'Two seeds resolved to the same model file.');
end
if numel(unique(scienceHashes)) ~= 1
    error('banff:evaluationConfigMismatch', ...
        'Seeds do not share one seed-independent scientific configuration.');
end
if numel(unique(checkpointHashes)) ~= numel(checkpointHashes)
    error('banff:evaluationCheckpointIdentity', 'Seed-specific experiment identities are not unique.');
end
sourceHashes = strings(numel(results), 1);
probeHashes = strings(numel(results), 1);
for index = 1:numel(results)
    if isfield(results(index).provenance,'training_source_sha256')
        sourceHashes(index)=hash_struct( ...
            results(index).provenance.training_source_sha256);
    else
        sourceHashes(index)=hash_struct(results(index).provenance.core_source_sha256);
    end
    probeHashes(index) = fixed_network_probe(results(index).config);
end
if numel(unique(sourceHashes)) ~= 1
    error('banff:evaluationSourceMismatch', 'Seeds were trained by different scientific core sources.');
end
if numel(results) > 1 && numel(unique(probeHashes)) ~= numel(probeHashes)
    error('banff:evaluationNetworkSeedAudit', ...
        'Distinct seeds did not produce distinct fixed-network generator probes.');
end
audit = table(actualSeeds(:), modelFiles(:), checkpointHashes(:), probeHashes(:), ...
    'VariableNames', {'Seed','ModelFile','CheckpointHash','FixedNetworkProbeHash'});
end

function hash = fixed_network_probe(cfg)
% Use the current model constructor on a small network.  This directly audits
% the deterministic fixed-weight generator without allocating a second 32k net.
probeCfg = cfg;
probeCfg.N_hidden = min(64, double(cfg.N_hidden));
probeCfg.N_recurrent = min(4, double(cfg.N_recurrent));
% The probe audits only fixed-weight generation. A full neuron-specific bias
% vector cannot be reused after deliberately reducing N_hidden, and the bias
% does not enter the probe hash, so use one valid scalar representative.
if ~isscalar(probeCfg.initial_bias)
    probeCfg.initial_bias = probeCfg.initial_bias(1);
end
P = banff_model('create', 4, 3, probeCfg);
values = [single(P.W_in(:)); single(P.W_out(:)); single(P.dale_sign(:))];
if P.recurrent_mode == "low_rank"
    values = [values; P.W_feedback(:); P.recurrent_expansion(:)]; %#ok<AGROW>
else
    [row, column, weight] = find(P.W_recurrent);
    values = [values; single(row); single(column); single(weight)]; %#ok<AGROW>
end
hash = sha256_bytes(typecast(single(values(:)), 'uint8'));
end

function hash = hash_struct(S)
names = sort(fieldnames(S));
text = "";
for index = 1:numel(names)
    text = text + string(names{index}) + "=" + string(S.(names{index})) + ";";
end
hash = sha256_bytes(unicode2native(char(text), 'UTF-8'));
end

function hash = sha256_bytes(bytes)
engine = javaMethod('getInstance', 'java.security.MessageDigest', 'SHA-256');
engine.update(uint8(bytes));
digest = typecast(engine.digest(), 'uint8');
hash = string(lower(reshape(dec2hex(digest).', 1, [])));
end

function T = classification_table(results)
n = numel(results);
seed = zeros(n,1); loss = nan(n,1); accuracy = nan(n,1); crossEntropy = nan(n,1);
active = nan(n,1);
for index = 1:n
    S = results(index).test.statistics;
    seed(index) = results(index).config.seed;
    loss(index) = results(index).test.loss;
    accuracy(index) = S.accuracy_percent;
    crossEntropy(index) = S.cross_entropy;
    active(index) = results(index).test.neural_activity.active_fraction_percent;
end
T = table(seed, loss, accuracy, crossEntropy, active, 'VariableNames', ...
    {'Seed','Loss','AccuracyPercent','CrossEntropy','ActiveNeuronPercent'});
end

function T = regression_table(results)
n = numel(results);
seed = zeros(n,1); loss = nan(n,1); rmse = nan(n,1); pearsonR = nan(n,1);
pearsonP = nan(n,1); signedMean = nan(n,1); signedStd = nan(n,1); active = nan(n,1);
for index = 1:n
    S = results(index).test.statistics;
    seed(index) = results(index).config.seed;
    loss(index) = results(index).test.loss;
    rmse(index) = S.rmse; pearsonR(index) = S.pearson_r; pearsonP(index) = S.pearson_p;
    signedMean(index) = S.signed_error_mean; signedStd(index) = S.signed_error_std;
    active(index) = results(index).test.neural_activity.active_fraction_percent;
end
T = table(seed, loss, rmse, pearsonR, pearsonP, signedMean, signedStd, active, ...
    'VariableNames', {'Seed','Loss','RMSE','PearsonR','PearsonP', ...
    'SignedErrorMean','SignedErrorStd','ActiveNeuronPercent'});
end

function T = dynamics_table(results,options)
n = numel(results);
seed = zeros(n,1); phaseDistance = nan(n,1); bestValidationDistance = nan(n,1);
initialConditionCount = zeros(n,1);
for index = 1:n
    seed(index) = results(index).config.seed;
    phaseDistance(index) = results(index).test.phase_distance;
    bestValidationDistance(index) = results(index).best.metric;
    initialConditionCount(index) = numel(results(index).test.phase_distance_by_initial_condition);
end
T = table(seed, phaseDistance, bestValidationDistance, initialConditionCount, ...
    'VariableNames', {'Seed','PhaseDistance','BestValidationPhaseDistance','InitialConditions'});
if options.assessment_split=="validation"
    T.Properties.VariableNames{4}='ValidationInitialConditions';
else
    T.Properties.VariableNames{4}='TestInitialConditions';
end
end

function [T,details] = evaluate_recurrent_ablation(results,kind,options)
% Measure the dependence of a trained readout on recurrent synaptic drive.
% Only the evaluation copy of the fixed recurrent operator is zeroed; the
% learned biases, encoder, decoder, data split and evaluation protocol are
% unchanged. Consequently this diagnostic neither retrains nor edits a model.
n=numel(results);
seed=zeros(n,1); fullMetric=nan(n,1); ablatedMetric=nan(n,1);
degradation=nan(n,1); fullLoss=nan(n,1); ablatedLoss=nan(n,1);
details=repmat(struct('seed',NaN,'full',[],'zero_recurrence',[]),1,n);
for index=1:n
    result=results(index);
    cfg=result.config;
    seed(index)=double(cfg.seed);
    details(index).seed=seed(index);
    if kind=="dynamics"
        dimension=numel(result.data_information.mean);
        P=banff_model('create',dimension,dimension,cfg);
        P.B=single(result.best.B);
        P=remove_recurrent_operator(P);
        ablated=banff_eval('closed_loop',banff_model('gpu',P),cfg, ...
            result.data_information,options.assessment_split,false);
        fullMetric(index)=double(result.test.phase_distance);
        ablatedMetric(index)=double(ablated.phase_distance);
        details(index).full=result.test;
        details(index).zero_recurrence=ablated;
    else
        [data,~]=banff_data('static',cfg,result.data_information);
        P=banff_model('create',size(data.X_train,1),size(data.Y_train,1),cfg);
        P.B=single(result.best.B);
        P=remove_recurrent_operator(P);
        [assessmentX,assessmentY]=static_assessment_data(data,options.assessment_split);
        ablated=banff_eval('static',banff_model('gpu',P),assessmentX, ...
            assessmentY,cfg,true);
        fullLoss(index)=double(result.test.loss);
        ablatedLoss(index)=double(ablated.loss);
        if kind=="classification"
            fullMetric(index)=double(result.test.statistics.accuracy_percent);
            ablatedMetric(index)=double(ablated.metric);
        else
            statistics=banff_metrics('regression',ablated.output,assessmentY, ...
                data.target_mean,data.target_std);
            fullMetric(index)=double(result.test.statistics.rmse);
            ablatedMetric(index)=double(statistics.rmse);
        end
    end
    if kind=="classification"
        degradation(index)=fullMetric(index)-ablatedMetric(index);
    else
        degradation(index)=ablatedMetric(index)-fullMetric(index);
    end
end
metricName=repmat(primary_metric(kind),n,1);
T=table(seed,metricName,fullMetric,ablatedMetric,degradation,fullLoss,ablatedLoss, ...
    'VariableNames',{'Seed','Metric','FullRecurrence','ZeroRecurrence', ...
    'Degradation','FullLoss','ZeroRecurrenceLoss'});
end

function P = remove_recurrent_operator(P)
% Preserve the architecture while making its recurrent current identically zero.
if P.recurrent_mode=="low_rank"
    P.recurrentGain=single(0);
    P.self_coupling(:)=single(0);
else
    P.W_recurrent=sparse(P.N_hidden,P.N_hidden);
end
end

function figures = plot_recurrent_ablation(T,details,task,kind,options)
fig=new_figure(options,task+" recurrent ablation");
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
nexttile; hold on;
for index=1:height(T)
    plot([1 2],[T.FullRecurrence(index) T.ZeroRecurrence(index)],'o-', ...
        'LineWidth',1.2,'DisplayName',sprintf('Seed %g',T.Seed(index)));
end
hold off; grid on; xlim([.7 2.3]);
set(gca,'XTick',[1 2],'XTickLabel',{'Full recurrence','Zero recurrence'});
ylabel(display_name(primary_metric(kind))); title("Paired "+lower(assessment_label(options))+" metric");
legend('Location','best');
nexttile; bar(T.Seed,T.Degradation); yline(0,'k-'); grid on;
xlabel('Initialisation seed'); ylabel('Performance degradation');
if kind=="classification"
    title('Full minus ablated accuracy (percentage points)');
else
    title('Ablated minus full error/distance');
end
save_figure(fig,options,'recurrent_ablation.png');
figures=fig;
if kind=="dynamics"
    figures=[figures plot_ablation_phase_portraits(details,task,options)]; %#ok<AGROW>
end
end

function figures = plot_ablation_phase_portraits(details,task,options)
figures=gobjects(0);
for seedIndex=1:numel(details)
condition=details(seedIndex);
for ic=1:numel(condition.full.prediction)
    fullPrediction=double(condition.full.prediction{ic});
    fullTruth=double(condition.full.truth{ic});
    zeroPrediction=double(condition.zero_recurrence.prediction{ic});
    zeroTruth=double(condition.zero_recurrence.truth{ic});
    if ~isequal(size(fullPrediction),size(fullTruth)) || ...
            ~isequal(size(zeroPrediction),size(zeroTruth)) || ...
            size(fullPrediction,2)~=size(zeroPrediction,2)
        error('banff:evaluationAblationTrajectoryShape', ...
            'Full and zero-recurrence phase trajectories have inconsistent shapes.');
    end
    dimension=size(fullPrediction,2);
    if dimension<2
        error('banff:evaluationAblationPhaseDimension', ...
            'Phase portraits require at least two state dimensions.');
    end
    pairs=nchoosek(1:dimension,2);
    fig=new_figure(options,task+" recurrent-ablation phase portraits");
    tiledlayout(size(pairs,1),2,'TileSpacing','compact','Padding','compact');
    for pairIndex=1:size(pairs,1)
        a=pairs(pairIndex,1); b=pairs(pairIndex,2);
        limits=phase_limits([fullTruth(:,[a b]);fullPrediction(:,[a b]); ...
            zeroTruth(:,[a b]);zeroPrediction(:,[a b])]);
        nexttile; plot(fullTruth(:,a),fullTruth(:,b),'k-','LineWidth',1.1); hold on;
        plot(fullPrediction(:,a),fullPrediction(:,b),'-','Color',[0 .447 .741], ...
            'LineWidth',1.1); hold off; axis equal; grid on;
        xlim(limits(1:2)); ylim(limits(3:4));
        xlabel(sprintf('x_%d',a)); ylabel(sprintf('x_%d',b));
        title('Full recurrence');
        if pairIndex==1, legend({'Reference','Network'},'Location','best'); end
        nexttile; plot(zeroTruth(:,a),zeroTruth(:,b),'k-','LineWidth',1.1); hold on;
        plot(zeroPrediction(:,a),zeroPrediction(:,b),'-','Color',[.85 .325 .098], ...
            'LineWidth',1.1); hold off; axis equal; grid on;
        xlim(limits(1:2)); ylim(limits(3:4));
        xlabel(sprintf('x_%d',a)); ylabel(sprintf('x_%d',b));
        title('Zero recurrence');
        if pairIndex==1, legend({'Reference','Network'},'Location','best'); end
    end
    sgtitle(sprintf('%s %s recurrent ablation, seed %g, IC %d', ...
        task_title(task),lower(assessment_label(options)),condition.seed,ic));
    save_figure(fig,options, ...
        seed_filename(options,condition.seed, ...
        sprintf('recurrent_ablation_phase_portraits_ic%02d.png',ic)));
    figures(end+1)=fig; %#ok<AGROW>
end
end
end

function S = summarise_numeric_columns(T)
names = string(T.Properties.VariableNames);
names(names=="Seed" | endsWith(names,"InitialConditions"))=[];
metric = strings(numel(names),1); meanValue = nan(numel(names),1);
sdValue = nan(numel(names),1); finiteN = zeros(numel(names),1);
for index = 1:numel(names)
    values = double(T.(names(index)));
    values = values(isfinite(values));
    metric(index) = names(index); finiteN(index) = numel(values);
    if ~isempty(values), meanValue(index) = mean(values); end
    if numel(values) > 1, sdValue(index) = std(values, 0); elseif numel(values) == 1, sdValue(index) = 0; end
end
S = table(metric, meanValue, sdValue, finiteN, ...
    'VariableNames', {'Metric','Mean','SampleSD','FiniteN'});
end

function fig = plot_summary(T, S, task, kind, options)
fig = new_figure(options, task + " held-out summary");
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
nexttile;
metricName = primary_metric(kind);
plot(T.Seed, T.(metricName), 'o-', 'LineWidth', 1.4, 'MarkerSize', 7);
grid on; xlabel('Initialisation seed'); ylabel(display_name(metricName));
title(display_name(metricName) + " by seed");
nexttile; axis off;
lines = [task_title(task) + " " + lower(assessment_label(options)), "", ...
    "Across-seed mean +/- sample SD"];
for index = 1:height(S)
    lines(end+1) = sprintf('%s: %.6g +/- %.6g (n=%d)', ... %#ok<AGROW>
        display_name(S.Metric(index)), S.Mean(index), S.SampleSD(index), S.FiniteN(index));
end
text(.02,.98,strjoin(lines,newline),'VerticalAlignment','top','Units','normalized');
save_figure(fig, options, 'summary_by_seed.png');
end

function name = primary_metric(kind)
if kind == "classification", name = "AccuracyPercent";
elseif kind == "regression", name = "RMSE";
else, name = "PhaseDistance";
end
end

function text = display_name(name)
switch string(name)
    case "AccuracyPercent", text = "Accuracy (%)";
    case "CrossEntropy", text = "Cross-entropy";
    case "ActiveNeuronPercent", text = "Active neurons (%)";
    case "PearsonR", text = "Pearson r";
    case "PearsonP", text = "Pearson p";
    case "SignedErrorMean", text = "Signed error mean";
    case "SignedErrorStd", text = "Signed error SD";
    case "PhaseDistance", text = "Phase-space distance";
    case "BestValidationPhaseDistance", text = "Best validation phase distance";
    otherwise, text = string(name);
end
end

function fig = plot_training_history(results, task, kind, options)
fig = new_figure(options, task + " training history");
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
colors = lines(numel(results));
nexttile; hold on;
for index = 1:numel(results)
    y = double(results(index).history.train_loss(:)); ok = isfinite(y);
    plot(find(ok), y(ok), '-', 'Color', colors(index,:), 'LineWidth', 1.1, ...
        'DisplayName', sprintf('Seed %g', results(index).config.seed));
end
hold off; grid on; xlabel('Epoch'); ylabel('Training loss'); title('Training loss');
set_log_if_positive(gca); legend('Location','best');
nexttile; hold on;
for index = 1:numel(results)
    if kind == "dynamics"
        y = double(results(index).history.validation_distance(:)); label = 'Validation phase distance';
    else
        y = double(results(index).history.validation_loss(:)); label = 'Validation loss';
    end
    ok = isfinite(y); plot(find(ok), y(ok), 'o-', 'Color', colors(index,:), ...
        'LineWidth', 1.1, 'MarkerSize', 3, 'DisplayName', sprintf('Seed %g', results(index).config.seed));
end
hold off; grid on; xlabel('Epoch'); ylabel(label); title(label); set_log_if_positive(gca);
legend('Location','best'); save_figure(fig, options, 'training_validation_history.png');
end

function fig = plot_bias_swarm(results, task, options)
fig = new_figure(options, task + " learned bias"); hold on;
colors = lines(numel(results));
xValues = [];
biasValues = [];
pointColors = zeros(0,3);
medianValues = nan(1,numel(results));
for index = 1:numel(results)
    values = double(results(index).best.B(:)); values = values(isfinite(values));
    if numel(values) > options.max_bias_points
        take = unique(round(linspace(1,numel(values),options.max_bias_points)));
        values = values(take);
    end
    xValues = [xValues; repmat(index,numel(values),1)]; %#ok<AGROW>
    biasValues = [biasValues; values]; %#ok<AGROW>
    pointColors = [pointColors; repmat(colors(index,:),numel(values),1)]; %#ok<AGROW>
    if ~isempty(values), medianValues(index) = median(values); end
end
if ~isempty(biasValues)
    swarmchart(xValues,biasValues,7,pointColors,'filled', ...
        'XJitter','density','XJitterWidth',.65, ...
        'MarkerFaceAlpha',.25,'MarkerEdgeAlpha',.20);
end
for index = 1:numel(results)
    if isfinite(medianValues(index))
        plot(index+[-.25 .25],medianValues(index).*[1 1],'k-','LineWidth',1.5);
    end
end
hold off; grid on; xlim([.5 numel(results)+.5]);
set(gca,'XTick',1:numel(results),'XTickLabel',arrayfun(@(R) string(R.config.seed),results));
xlabel('Initialisation seed'); ylabel('Learned hidden bias (mV)');
title('Learned bias distributions by seed'); save_figure(fig, options, 'learned_bias_swarm.png');
end

function figures = plot_classification(results, task, options)
figures = gobjects(0);
for seedIndex=1:numel(results)
result=results(seedIndex);
truth = double(result.test.statistics.true_class(:));
predicted = double(result.test.statistics.predicted_class(:));
classCount = max([truth; predicted]); confusion = accumarray([truth predicted],1,[classCount classCount]);
fig = new_figure(options, task + " confusion matrix"); imagesc(confusion); axis image;
colorbar; xlabel('Predicted class'); ylabel('True class'); title(sprintf( ...
    '%s confusion matrix, seed %g', task_title(task), result.config.seed));
set(gca,'XTick',1:classCount,'YTick',1:classCount); annotate_matrix(confusion);
save_figure(fig, options, seed_filename(options,result.config.seed,'confusion_matrix.png')); figures(end+1)=fig;

logits = double(result.test.output); shifted = logits-max(logits,[],1);
probability = exp(shifted)./sum(exp(shifted),1); confidence = max(probability,[],1);
fig = new_figure(options, task + " classification confidence");
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
nexttile; histogram(confidence(predicted==truth),20,'DisplayName','Correct'); hold on;
histogram(confidence(predicted~=truth),20,'DisplayName','Incorrect'); hold off;
xlabel('Maximum softmax probability'); ylabel('Samples'); grid on; legend; title('Prediction confidence');
nexttile; bar([sum(confusion,2), sum(confusion,1).']); grid on; xlabel('Class');
ylabel('Samples'); legend({'True','Predicted'},'Location','best');
title(sprintf('%s class counts, seed %g',assessment_label(options),result.config.seed));
save_figure(fig, options, seed_filename(options,result.config.seed,'classification_details.png')); figures(end+1)=fig;

if any(task == ["mnist","afro_mnist_vai"])
    fig = plot_image_examples(result, task, options);
    if isgraphics(fig), figures(end+1)=fig; end
end
end
end

function fig = plot_image_examples(result, task, options)
fig = gobjects(0);
try
    [data,~] = banff_data('static', result.config, result.data_information);
    [assessmentX,~]=static_assessment_data(data,options.assessment_split);
    X=double(assessmentX); side = round(sqrt(size(X,1)));
    if side*side ~= size(X,1), return; end
    truth = double(result.test.statistics.true_class(:));
    predicted = double(result.test.statistics.predicted_class(:));
    wrong = find(truth~=predicted); correct = find(truth==predicted);
    count = min(options.max_image_examples,numel(truth));
    selection = [wrong(1:min(numel(wrong),ceil(count/2))); ...
        correct(1:min(numel(correct),count-min(numel(wrong),ceil(count/2))))];
    if numel(selection)<count
        remaining = setdiff((1:numel(truth)).',selection,'stable');
        selection = [selection; remaining(1:min(numel(remaining),count-numel(selection)))];
    end
    columns = ceil(sqrt(numel(selection))); rows = ceil(numel(selection)/columns);
    fig = new_figure(options, task + " " + lower(assessment_label(options)) + " images");
    tiledlayout(rows,columns,'TileSpacing','compact','Padding','compact');
    for index = 1:numel(selection)
        sample = selection(index); nexttile;
        % FLATTEN_IMAGES uses MATLAB column-major vectorisation; RESHAPE alone
        % restores the stored row/column orientation. A transpose here would
        % reflect every displayed digit across its diagonal.
        imagesc(reshape(X(:,sample),side,side)); axis image off; colormap(gray);
        title(sprintf('T:%d P:%d',truth(sample)-1,predicted(sample)-1), ...
            'Color', ternary(truth(sample)==predicted(sample),[0 .5 0],[.8 0 0]));
    end
    sgtitle(sprintf('%s normalized %s examples, seed %g',task_title(task), ...
        lower(assessment_label(options)),result.config.seed));
    save_figure(fig, options,seed_filename(options,result.config.seed,'image_examples.png'));
catch exception
    warning('banff:evaluationImagePlot','Image examples were skipped: %s',exception.message);
end
end

function figures = plot_regression(results, task, options)
figures = gobjects(0);
for seedIndex=1:numel(results)
result=results(seedIndex);
S = result.test.statistics; truth = double(S.truth(:)); prediction = double(S.prediction(:));
errorValue = prediction-truth;
fig = new_figure(options, task + " " + lower(assessment_label(options)) + " regression");
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
nexttile; scatter(truth,prediction,18,'filled','MarkerFaceAlpha',.45); hold on;
limits = finite_limits([truth;prediction]); plot(limits,limits,'k--','LineWidth',1.2); hold off;
axis square; xlim(limits); ylim(limits); grid on; xlabel('Truth'); ylabel('Prediction');
title(sprintf('%s predictions, seed %g',assessment_label(options),result.config.seed));
nexttile; scatter(truth,errorValue,18,'filled','MarkerFaceAlpha',.45); yline(0,'k--');
grid on; xlabel('Truth'); ylabel('Prediction - truth'); title('Residuals versus truth');
save_figure(fig, options,seed_filename(options,result.config.seed,'prediction_and_residuals.png')); figures(end+1)=fig;
end

fig = new_figure(options, task + " residual distribution");
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
nexttile; hold on;
for seedIndex=1:numel(results)
    S=results(seedIndex).test.statistics;
    errorValue=double(S.prediction(:))-double(S.truth(:));
    histogram(errorValue,30,'DisplayStyle','stairs','LineWidth',1.2, ...
        'DisplayName',sprintf('Seed %g',results(seedIndex).config.seed));
end
hold off; grid on; xlabel('Prediction - truth'); ylabel('Samples');
title('Signed-error distributions by seed'); legend('Location','best');
nexttile; errorbar(1:numel(results), arrayfun(@(R) double(R.test.statistics.signed_error_mean),results), ...
    arrayfun(@(R) double(R.test.statistics.signed_error_std),results), 'o','LineWidth',1.3);
grid on; xlim([.5 numel(results)+.5]); set(gca,'XTick',1:numel(results), ...
    'XTickLabel',arrayfun(@(R) string(R.config.seed),results));
xlabel('Initialisation seed'); ylabel('Signed error mean +/- SD'); title('Residual summary by seed');
save_figure(fig, options, assessment_filename(options,'residual_distribution.png')); figures(end+1)=fig;
end

function figures = plot_dynamics(results, task, options)
figures = gobjects(0);
maxIC = max(arrayfun(@(R) numel(R.test.phase_distance_by_initial_condition),results));
distance = nan(numel(results),maxIC);
for index=1:numel(results)
    values=double(results(index).test.phase_distance_by_initial_condition(:));
    distance(index,1:numel(values))=values;
end
fig = new_figure(options, task + " phase distance");
plot(1:maxIC,distance.','o-','LineWidth',1.2); grid on;
xlabel(assessment_label(options)+" initial condition");
ylabel('Phase-space distance'); title('Per-initial-condition closed-loop distance');
legend(arrayfun(@(R) sprintf('Seed %g',R.config.seed),results,'UniformOutput',false),'Location','best');
save_figure(fig, options, 'phase_distance_by_initial_condition.png'); figures(end+1)=fig;

for seedIndex=1:numel(results)
    result=results(seedIndex);
    for ic=1:numel(result.test.prediction)
        prediction=double(result.test.prediction{ic}); truth=double(result.test.truth{ic});
        if ~isequal(size(prediction),size(truth))
            error('banff:evaluationTrajectoryShape', ...
                'Saved prediction and truth trajectories have different shapes.');
        end
        n=size(prediction,1);
        time=(0:n-1).'*double(result.config.dt);
        fig = new_figure(options, sprintf('%s time series IC %d',task,ic));
        tiledlayout(size(truth,2),1,'TileSpacing','compact','Padding','compact');
        for dimension=1:size(truth,2)
            nexttile; plot(time,truth(:,dimension),'k-','LineWidth',1.2); hold on;
            plot(time,prediction(:,dimension),'r--','LineWidth',1.1); hold off; grid on;
            ylabel(sprintf('x_%d',dimension));
            if dimension==1, legend({'True','Network'},'Location','best'); end
            if dimension==size(truth,2), xlabel('Time (s)'); end
        end
        sgtitle(sprintf('%s closed loop, seed %g, %s IC %d',task_title(task), ...
            result.config.seed,lower(assessment_label(options)),ic));
        save_figure(fig,options,seed_filename(options,result.config.seed, ...
            sprintf('trajectory_time_series_ic%02d.png',ic)));
        figures(end+1)=fig; %#ok<AGROW>
        if size(truth,2)>=2
            pairs=nchoosek(1:size(truth,2),2);
            fig=new_figure(options,sprintf('%s phase IC %d',task,ic));
            tiledlayout(size(pairs,1),2,'TileSpacing','compact','Padding','compact');
            for pair=1:size(pairs,1)
                a=pairs(pair,1); b=pairs(pair,2);
                limits=phase_limits([truth(:,[a b]);prediction(:,[a b])]);
                nexttile; plot(truth(:,a),truth(:,b),'k-','LineWidth',1.1); axis equal; grid on;
                xlim(limits(1:2)); ylim(limits(3:4)); xlabel(sprintf('x_%d',a)); ylabel(sprintf('x_%d',b)); title('True');
                nexttile; plot(prediction(:,a),prediction(:,b),'r--','LineWidth',1.1); axis equal; grid on;
                xlim(limits(1:2)); ylim(limits(3:4)); xlabel(sprintf('x_%d',a)); ylabel(sprintf('x_%d',b)); title('Network');
            end
            sgtitle(sprintf('%s phase portraits, seed %g, %s IC %d',task_title(task), ...
                result.config.seed,lower(assessment_label(options)),ic));
            save_figure(fig,options,seed_filename(options,result.config.seed, ...
                sprintf('trajectory_phase_portraits_ic%02d.png',ic)));
            figures(end+1)=fig; %#ok<AGROW>
        end
    end
end
end

function figures = plot_dynamics_current_comparison(results,task,options)
% Use the same fixed realization and assessment initial conditions for the
% initial-bias and selected trained-bias networks. Each plotted datum is one
% hidden neuron; warmup establishes state but is excluded from the RMS.
figures=gobjects(0);
for seedIndex=1:numel(results)
result=results(seedIndex);
cfg=result.config;
dimension=numel(result.data_information.mean);
untrainedP=banff_model('create',dimension,dimension,cfg);
trainedP=untrainedP;
trainedP.B=single(result.best.B);
untrainedCurrent=banff_plot('dynamics_current_magnitudes',untrainedP,cfg, ...
    result.data_information,options.assessment_split);
trainedCurrent=banff_plot('dynamics_current_magnitudes',trainedP,cfg, ...
    result.data_information,options.assessment_split);
fprintf('Seed %g exact full-%s DS current magnitudes (scored interval; warmup excluded)\n', ...
    result.config.seed,lower(assessment_label(options)));
disp(current_magnitude_table(untrainedCurrent,trainedCurrent));
fprintf(['DS current comparison: %d initial conditions and %d scored timesteps ', ...
    '(%g observations per neuron and condition).\n'], ...
    trainedCurrent.test_samples,trainedCurrent.timesteps_per_sample, ...
    trainedCurrent.observations_per_neuron);
fig=new_figure(options,task+" dynamics current magnitudes");
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
nexttile; plot_direct_current_comparison(untrainedCurrent,trainedCurrent,options);
nexttile; plot_afferent_comparison(untrainedCurrent,trainedCurrent,options);
nexttile; plot_decoder_comparison(untrainedCurrent,trainedCurrent,options);
sgtitle(sprintf('%s closed-loop current contributions, seed %g', ...
    task_title(task),result.config.seed));
save_figure(fig,options,seed_filename(options,result.config.seed, ...
    'dynamics_current_magnitude_comparison.png'));
figures(end+1)=fig; %#ok<AGROW>
end
end

function figures = plot_activity(results, task, kind, options)
figures=gobjects(0);
for seedIndex=1:numel(results)
result=results(seedIndex);
if kind ~= "dynamics"
    rates=double(result.test.neural_activity.mean_firing_rate_by_neuron_hz(:));
    fig=new_figure(options,task+" spiking activity");
    tiledlayout(2,4,'TileSpacing','compact','Padding','compact');
    nexttile; histogram(rates,50); grid on; xlabel('Mean firing rate (Hz)'); ylabel('Neurons');
    title(sprintf('%s rates; %.2f%% active',assessment_label(options), ...
        result.test.neural_activity.active_fraction_percent));
    if options.replay_static_spikes
        plot_static_diagnostics(result,options);
    else
        nexttile;
        plot(sort(rates,'descend'),'LineWidth',1.1); xlabel('Ranked neuron'); ylabel('Mean rate (Hz)'); grid on;
        title('Ranked firing rates');
        for index=1:6, nexttile; axis off; end
    end
    sgtitle(sprintf('%s spiking diagnostics, seed %g',task_title(task),result.config.seed));
    save_figure(fig,options,seed_filename(options,result.config.seed,'spiking_diagnostics.png'));
else
    events=result.test.events{min(options.representative_initial_condition,numel(result.test.events))};
    fig=new_figure(options,task+" spiking activity");
    tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
    nexttile; plot_event_raster(events,result.config,options);
    [eventRates, ~] = event_statistics(events,result.config,options);
    nexttile; histogram(eventRates,50); grid on; xlabel('Firing rate (Hz)'); ylabel('Active neurons');
    title(sprintf('Event rates; %.2f%% active',100*numel(eventRates)/result.config.N_hidden));
    nexttile;
    isiByInitialCondition=cell(numel(result.test.events),1);
    for initialCondition=1:numel(result.test.events)
        [~,localIsi]=event_statistics( ...
            result.test.events{initialCondition}, ...
            result.config,options);
        isiByInitialCondition{initialCondition}=localIsi(:);
    end
    allEventIsi=vertcat(isiByInitialCondition{:});
    inverseIsiRate = inverse_isi_rate_hz(allEventIsi);
    if isempty(inverseIsiRate), axis off; text(.5,.5,'No repeated-neuron spikes','HorizontalAlignment','center');
    else, histogram(inverseIsiRate,50); grid on; xlabel('Inverse ISI (Hz)'); ylabel('Intervals'); title("Full-"+lower(assessment_label(options))+" instantaneous rate distribution"); end
    nexttile; rho=double(events.rho(:)); rho=rho(isfinite(rho));
    if isempty(rho), axis off; text(.5,.5,'No recorded spikes','HorizontalAlignment','center');
    else, histogram(rho,40); grid on; xlabel('\rho at spike'); ylabel('Spikes'); title('Within-step event fraction'); end
    fprintf('Dynamics seed %g: %.2f%% active, %.2f%% silent neurons; %d spikes.\n', ...
        result.config.seed,100*numel(eventRates)/result.config.N_hidden, ...
        100*(1-numel(eventRates)/result.config.N_hidden),numel(events.step));
    fprintf(['Inverse-ISI distribution pools %d within-trajectory intervals ', ...
        'from all %d neurons and all %d %s initial conditions.\n'], ...
        numel(inverseIsiRate),result.config.N_hidden, ...
        numel(result.test.events),lower(assessment_label(options)));
    sgtitle(sprintf('%s spiking diagnostics, seed %g',task_title(task),result.config.seed));
    save_figure(fig,options,seed_filename(options,result.config.seed,'spiking_diagnostics.png'));
end
figures(end+1)=fig; %#ok<AGROW>
end
end

function plot_static_diagnostics(result,options)
try
    [data,~]=banff_data('static',result.config,result.data_information);
    untrainedP=banff_model('create',size(data.X_train,1),size(data.Y_train,1),result.config);
    trainedP=untrainedP;
    trainedP.B=single(result.best.B);
    [assessmentX,~]=static_assessment_data(data,options.assessment_split);
    sampleCount=min(options.max_spike_samples,size(assessmentX,2));
    [spikes,voltage]=banff_plot('static_traces',trainedP, ...
        assessmentX(:,1:sampleCount),result.config);
    untrainedCurrent=banff_plot('static_current_magnitudes', ...
        untrainedP,assessmentX,result.config);
    trainedCurrent=banff_plot('static_current_magnitudes', ...
        trainedP,assessmentX,result.config);
    fprintf('Exact full-%s current magnitudes (RMS over neurons, samples and time)\n', ...
        lower(assessment_label(options)));
    disp(current_magnitude_table(untrainedCurrent,trainedCurrent));
    duration=double(result.config.presentation_steps)*double(result.config.dt);
    replayRates=sum(spikes,[2 3])./(sampleCount*duration);
    activeBySample=squeeze(mean(any(spikes,2),1))*100;
    fprintf(['Static seed %g: %.2f%% active, %.2f%% silent neurons over %d samples; ', ...
        'mean per-sample active fraction %.2f%%.\n'],result.config.seed, ...
        100*mean(replayRates>0),100*mean(replayRates==0),sampleCount,mean(activeBySample));

    nexttile; plot(sort(replayRates,'descend'),'LineWidth',1.1); grid on;
    xlabel('Ranked neuron'); ylabel('Replay firing rate (Hz)'); title('Ranked replay firing rates');
    counts=sum(spikes(:,:,1),2); [~,order]=sort(counts,'descend');
    keep=order(1:min(options.max_raster_neurons,numel(order))); [neuron,step]=find(spikes(keep,:,1));
    nexttile;
    if isempty(step)
        axis off; text(.5,.5,'No spikes in representative sample','HorizontalAlignment','center');
    else
        scatter(step,neuron,7,'k','filled'); set(gca,'YDir','reverse'); grid on;
        xlabel('Presentation step'); ylabel('Ranked active neuron');
        title("Representative "+lower(assessment_label(options))+" spike raster");
    end
    nexttile;
    traceKeep=keep(1:min(20,numel(keep))); plot(double(voltage(traceKeep,:)).','LineWidth',.7); grid on;
    xlabel('Presentation step'); ylabel('Voltage (mV)'); title('Representative neuron voltages');
    nexttile;
    inverseIsiRate=banff_plot('static_inverse_isi_rates',trainedP, ...
        assessmentX,result.config);
    if isempty(inverseIsiRate), axis off; text(.5,.5,'No repeated-neuron spikes','HorizontalAlignment','center');
    else, histogram(inverseIsiRate,50); grid on; xlabel('Inverse ISI (Hz)'); ylabel('Intervals'); title('Full-set instantaneous rate distribution'); end
    fprintf(['Inverse-ISI distribution pools %d within-sample intervals from ', ...
        'all %d neurons and all %d %s samples.\n'],numel(inverseIsiRate), ...
        trainedP.N_hidden,size(assessmentX,2),lower(assessment_label(options)));
    nexttile; plot_direct_current_comparison(untrainedCurrent,trainedCurrent,options);
    nexttile; plot_afferent_comparison(untrainedCurrent,trainedCurrent,options);
    nexttile; plot_decoder_comparison(untrainedCurrent,trainedCurrent,options);
    fprintf(['Current comparison: all %d %s samples and %d presentation steps ', ...
        '(%g observations per neuron and condition).\n'],trainedCurrent.test_samples, ...
        lower(assessment_label(options)),trainedCurrent.timesteps_per_sample, ...
        trainedCurrent.observations_per_neuron);
catch exception
    warning('banff:evaluationStaticRaster','Static raster replay was skipped: %s',exception.message);
    for index=1:7, nexttile; axis off; text(.5,.5,'Static replay unavailable','HorizontalAlignment','center'); end
end
end

function T = current_magnitude_table(untrained,trained)
condition=["Untrained";"Trained"];
summaries={untrained.aggregate;trained.aggregate};
encoderRmsMv=cellfun(@(S)S.encoder_rms_mV,summaries);
netRecurrentRmsMv=cellfun(@(S)S.net_recurrent_rms_mV,summaries);
recurrentToEncoderRms=cellfun(@(S)S.recurrent_to_encoder_rms,summaries);
grossEncoderRmsMv=cellfun(@(S)S.gross_encoder_rms_mV,summaries);
grossRecurrentRmsMv=cellfun(@(S)S.gross_recurrent_rms_mV,summaries);
netToGrossEncoderRms=cellfun(@(S)S.net_to_gross_encoder_rms,summaries);
netToGrossRecurrentRms=cellfun(@(S)S.net_to_gross_recurrent_rms,summaries);
adaptationRmsMv=cellfun(@(S)S.adaptation_rms_mV,summaries);
biasDeviationRmsMv=cellfun(@(S)S.bias_deviation_rms_mV,summaries);
decoderContributionRms=cellfun(@(S)S.decoder_contribution_rms,summaries);
T=table(condition,encoderRmsMv,netRecurrentRmsMv,recurrentToEncoderRms, ...
    grossEncoderRmsMv,grossRecurrentRmsMv,netToGrossRecurrentRms, ...
    netToGrossEncoderRms, ...
    adaptationRmsMv,biasDeviationRmsMv,decoderContributionRms, ...
    'VariableNames',{'Condition','EncoderRmsMv','NetRecurrentRmsMv', ...
    'RecurrentToEncoderRms','GrossEncoderRmsMv','GrossRecurrentRmsMv', ...
    'NetToGrossRecurrentRms','NetToGrossEncoderRms', ...
    'AdaptationRmsMv','BiasDeviationRmsMv', ...
    'DecoderContributionRms'});
end

function plot_direct_current_comparison(untrained,trained,options)
% Net encoder and recurrent currents are the signed sums that enter the
% membrane equation. Adaptation is subtractive; bias is shown relative to the
% configured initial value. Their common per-neuron RMS definition makes this
% the primary comparison of dynamical contribution scales.
biasReference=double(trained.bias_reference_mV);
fields={'encoder_net_rms','recurrent_net_rms','adaptation_rms','bias_deviation'};
if isscalar(biasReference)
    biasLabel=sprintf('Bias - %.3g mV',biasReference);
else
    biasLabel='Bias - neuron-specific initial value';
end
labels={'Encoder','Net recurrent','Adaptation',biasLabel};
plot_paired_distribution(untrained,trained,options,fields,labels, ...
    'Per-neuron RMS contribution (mV)', ...
    {'Direct membrane-related contribution scale', ...
    char("RMS over the full "+lower(assessment_label(options))+ ...
    " set and all presentation timesteps")});
end

function plot_afferent_comparison(untrained,trained,options)
% Gross afferent magnitudes retain excitation and inhibition separately until
% after absolute values are taken, unlike the net membrane current.
fields={'encoder_gross_afferent_rms','recurrent_gross_afferent_rms'};
labels={'Encoder gross','Recurrent gross'};
plot_paired_distribution(untrained,trained,options,fields,labels, ...
    'Per-neuron RMS gross afferent magnitude (mV)', ...
    {'Absolute afferent drive before cancellation', ...
    char("Full "+lower(assessment_label(options))+ ...
    " set; one datum per postsynaptic neuron")});
end

function plot_decoder_comparison(untrained,trained,options)
% Decoder contributions are not membrane currents. Keep their normalized
% output units on a separate axis rather than making an invalid mV comparison.
plot_paired_distribution(untrained,trained,options,{'decoder_presynaptic_rms'}, ...
    {'Hidden to output'},'Per-neuron RMS output contribution', ...
    {'Presynaptic decoder contributions', ...
    'Normalized-output units; scored decoder window'});
end

function plot_paired_distribution(untrained,trained,options,fields,labels,yLabel,titleText)
% Box charts use every neuron; the visible swarm is deterministically thinned.
untrainedReference=double(untrained.bias_reference_mV(:));
trainedReference=double(trained.bias_reference_mV(:));
referenceMismatch=numel(untrainedReference)~=numel(trainedReference) || ...
    any(abs(untrainedReference-trainedReference)>1e-6);
if referenceMismatch || ...
        untrained.test_samples~=trained.test_samples || ...
        untrained.timesteps_per_sample~=trained.timesteps_per_sample
    error('banff:evaluationCurrentComparison', ...
        'Trained and untrained current summaries use inconsistent references or data.');
end
conditions={untrained,trained};
conditionColors={[.50 .50 .50],[0 .4470 .7410]};
neuronCount=numel(trained.(fields{1}));
if neuronCount<1 || ~isscalar(options.max_current_points) || ...
        ~isfinite(options.max_current_points) || options.max_current_points<1
    error('banff:evaluationCurrentDisplay', ...
        'Current summaries and max_current_points must be nonempty and valid.');
end
swarmCount=min(neuronCount,max(1,round(options.max_current_points)));
swarmIndex=unique(round(linspace(1,neuronCount,swarmCount)));
hold on;
for condition=1:2
    offset=(condition-1.5)*.34;
    for index=1:numel(fields)
        values=double(conditions{condition}.(fields{index})(:));
        if numel(values)~=neuronCount || any(~isfinite(values)) || any(values<0)
            error('banff:evaluationCurrentValues', ...
                'Current summaries must be equally sized, finite and nonnegative.');
        end
        position=index+offset;
        boxchart(repmat(position,neuronCount,1),values, ...
            'BoxFaceColor',conditionColors{condition},'BoxWidth',.28, ...
            'MarkerStyle','none');
        swarmchart(repmat(position,numel(swarmIndex),1),values(swarmIndex),5, ...
            conditionColors{condition},'filled','XJitter','density', ...
            'XJitterWidth',.14,'MarkerFaceAlpha',.18,'MarkerEdgeAlpha',.12);
    end
end
h=gobjects(1,2);
h(1)=plot(nan,nan,'s','MarkerFaceColor',conditionColors{1}, ...
    'MarkerEdgeColor',conditionColors{1},'DisplayName','Untrained');
h(2)=plot(nan,nan,'s','MarkerFaceColor',conditionColors{2}, ...
    'MarkerEdgeColor',conditionColors{2},'DisplayName','Trained');
hold off; grid on; xlim([.5 numel(fields)+.5]);
set(gca,'XTick',1:numel(fields),'XTickLabel',labels);
ylabel(yLabel);
title(titleText);
legend(h,'Location','best');
end

function rate = inverse_isi_rate_hz(isi)
% Convert positive inter-spike intervals in seconds to instantaneous rates.
% The result is an interval-weighted distribution of 1/ISI and is distinct
% from the per-neuron mean firing rates shown in the adjacent panels.
isi=double(isi(:));
isi=isi(isfinite(isi) & isi>0);
rate=1./isi;
end

function [rates,isi] = event_statistics(events,cfg,options)
neuron=double(events.neuron(:)); step=double(events.step(:));
rho=double(events.rho(:));
if numel(rho)~=numel(step) || any(~isfinite(rho)) || any(rho<0 | rho>1)
    error('banff:evaluationEventRho', ...
        'Recorded spike event fractions must be finite values in [0,1].');
end
% Report activity only over the scored assessment interval. Warmup spikes are useful
% for trajectory initialization and raster context, but must not contribute to
% firing rates, active-neuron fractions, or inter-spike intervals.
if options.assessment_split=="validation"
    warmup=cfg.validation_warmup_time;
    duration=cfg.validation_time;
else
    warmup=cfg.test_warmup_time;
    duration=cfg.test_time;
end
warmupSteps=round(double(warmup)/double(cfg.dt));
recordingSteps=round(double(duration)/double(cfg.dt));
lastRecordingStep=warmupSteps+recordingSteps;
if any(step < 1 | step > lastRecordingStep)
    error('banff:evaluationEventStep', ...
        'A recorded spike lies outside the configured evaluation interval.');
end
scored=step>warmupSteps;
neuron=neuron(scored);
eventTime=(step(scored)-warmupSteps-1+rho(scored)).*double(cfg.dt);
if isempty(neuron), rates=[]; isi=[]; return; end
[uniqueNeuron,~,group]=unique(neuron); counts=accumarray(group,1);
duration=max(1,recordingSteps)*double(cfg.dt); rates=counts./duration;
isiByNeuron=cell(numel(uniqueNeuron),1);
for index=1:numel(uniqueNeuron)
    times=sort(eventTime(group==index));
    if numel(times)>1
        isiByNeuron{index}=diff(times);
    else
        isiByNeuron{index}=zeros(0,1);
    end
end
isi=vertcat(isiByNeuron{:});
end

function plot_event_raster(events,cfg,options)
neuron=double(events.neuron(:)); step=double(events.step(:));
if isempty(step), axis off; text(.5,.5,'No recorded spikes','HorizontalAlignment','center'); return; end
[uniqueNeuron,~,group]=unique(neuron); counts=accumarray(group,1); [~,order]=sort(counts,'descend');
keep=uniqueNeuron(order(1:min(options.max_raster_neurons,numel(order))));
selected=ismember(neuron,keep); [~,rank]=ismember(neuron(selected),keep);
scatter(step(selected).*double(cfg.dt),rank,7,'k','filled'); set(gca,'YDir','reverse'); grid on;
xlabel('Time (s)'); ylabel('Ranked active neuron'); title('Representative closed-loop spike raster');
end

function fig = new_figure(options,~)
% Do not set a window Name: Live Scripts identify figures by their visible
% titles and can capture unnamed visible figures in the inline output panel.
properties={'Color','w'};
if ~strcmpi(string(options.figure_visibility),"on")
    % Explicitly hidden figures remain available for headless export. For
    % visible figures, omitting Visible lets the Live Editor capture the figure
    % in the .mlx output panel instead of forcing an external figure window.
    properties=[properties,{'Visible',char(options.figure_visibility)}]; %#ok<AGROW>
end
fig=figure(properties{:});
end

function save_figure(fig,options,filename)
drawnow;
if ~options.save_figures, return; end
path=fullfile(options.output_directory,filename);
if exist('exportgraphics','file')==2, exportgraphics(fig,path,'Resolution',180); else, saveas(fig,path); end
end

function set_log_if_positive(ax)
objects=findobj(ax,'Type','line'); values=[];
for index=1:numel(objects), values=[values,objects(index).YData]; end %#ok<AGROW>
values=values(isfinite(values)); if ~isempty(values)&&all(values>0), ax.YScale='log'; end
end

function annotate_matrix(matrix)
limit=max(matrix(:)); if limit==0, limit=1; end
for row=1:size(matrix,1)
    for column=1:size(matrix,2)
        color='k'; if matrix(row,column)>.55*limit, color='w'; end
        text(column,row,sprintf('%d',matrix(row,column)),'HorizontalAlignment','center','Color',color);
    end
end
end

function limits = finite_limits(values)
values=values(isfinite(values));
if isempty(values), limits=[-1 1]; return; end
lo=min(values); hi=max(values); if lo==hi, pad=max(1,abs(lo))*.05; else, pad=(hi-lo)*.05; end
limits=[lo-pad hi+pad];
end

function limits = phase_limits(values)
x=finite_limits(values(:,1)); y=finite_limits(values(:,2)); limits=[x y];
end

function value = ternary(condition,yes,no)
if condition, value=yes; else, value=no; end
end

function label = assessment_label(options)
if options.assessment_split=="validation"
    label="Validation";
else
    label="Held-out test";
end
end

function filename = assessment_filename(options,suffix)
if options.assessment_split=="validation"
    prefix='validation_';
else
    prefix='held_out_';
end
filename=[prefix char(suffix)];
end

function filename = seed_filename(options,seed,suffix)
% Give every seed-resolved export a unique, sortable file name.
filename=assessment_filename(options,sprintf('seed%03d_%s',round(double(seed)),char(suffix)));
end

function [X,Y] = static_assessment_data(data,split)
if split=="validation"
    X=data.X_validation;
    Y=data.Y_validation;
elseif split=="test"
    X=data.X_test;
    Y=data.Y_test;
else
    error('banff:evaluationSplit','assessment_split must be validation or test.');
end
end

function text = task_title(task)
switch string(task)
    case "breast_cancer", text="Breast cancer";
    case "mnist", text="MNIST";
    case "afro_mnist_vai", text="Afro-MNIST (Vai)";
    case "abalone", text="Abalone";
    case "toyota", text="Toyota";
    case "yacht", text="Yacht hydrodynamics";
    case "lorenz", text="Lorenz";
    case "sprott_s", text="Sprott-S";
    case "vanderpol", text="Van der Pol";
    otherwise, text=string(task);
end
end
