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

fprintf('\nBANFF held-out evaluation: %s (%s profile)\n', task, profile);
fprintf('Seeds: %s\n\n', mat2str(double(seeds(:).')));
results = run_experiment("test", task, profile, seeds, overrides);
results = results(:).';
audit = audit_seed_models(results, seeds);

kind = string(results(1).config.kind);
switch kind
    case "classification"
        seedTable = classification_table(results);
    case "regression"
        seedTable = regression_table(results);
    case "dynamics"
        seedTable = dynamics_table(results);
    otherwise
        error('banff:evaluationKind', 'Unsupported task kind "%s".', kind);
end
summaryTable = summarise_numeric_columns(seedTable);

fprintf('Per-seed held-out results\n');
disp(seedTable);
fprintf('Across-seed mean and sample SD\n');
disp(summaryTable);
fprintf('Seed/model audit: PASS (%d distinct requested seeds and model identities).\n\n', ...
    numel(seeds));

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
figures(end+1) = plot_activity(results, task, kind, options); %#ok<AGROW>

report = struct('task', task, 'kind', kind, 'profile', string(profile), ...
    'seeds', double(seeds(:).'), 'results', results, 'seed_table', seedTable, ...
    'summary_table', summaryTable, 'network_seed_audit', audit, ...
    'figures', figures(isgraphics(figures)), 'display_options', options);
end

function options = default_display_options(root, task, changes)
options = struct();
options.save_figures = false;
options.output_directory = fullfile(root, 'outputs', 'evaluation', char(task));
options.representative_seed_index = 1;
options.representative_initial_condition = 1;
options.max_bias_points = 2500;
options.max_raster_neurons = 300;
options.max_image_examples = 16;
options.max_spike_samples = 8;
options.replay_static_spikes = true;
options.figure_visibility = "on";
names = fieldnames(changes);
for index = 1:numel(names)
    options.(names{index}) = changes.(names{index});
end
options.representative_seed_index = max(1, round(options.representative_seed_index));
options.representative_initial_condition = max(1, round(options.representative_initial_condition));
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
    sourceHashes(index) = hash_struct(results(index).provenance.core_source_sha256);
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

function T = dynamics_table(results)
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
    'VariableNames', {'Seed','PhaseDistance','BestValidationPhaseDistance','TestInitialConditions'});
end

function S = summarise_numeric_columns(T)
names = string(T.Properties.VariableNames);
names(names == "Seed" | names == "TestInitialConditions") = [];
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
lines = [task_title(task) + " held-out test", "", "Across-seed mean +/- sample SD"];
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
figures = gobjects(0); representative = pick_result(results, options);
truth = double(representative.test.statistics.true_class(:));
predicted = double(representative.test.statistics.predicted_class(:));
classCount = max([truth; predicted]); confusion = accumarray([truth predicted],1,[classCount classCount]);
fig = new_figure(options, task + " confusion matrix"); imagesc(confusion); axis image;
colorbar; xlabel('Predicted class'); ylabel('True class'); title(sprintf( ...
    '%s confusion matrix, seed %g', task_title(task), representative.config.seed));
set(gca,'XTick',1:classCount,'YTick',1:classCount); annotate_matrix(confusion);
save_figure(fig, options, 'held_out_confusion_matrix.png'); figures(end+1)=fig;

logits = double(representative.test.output); shifted = logits-max(logits,[],1);
probability = exp(shifted)./sum(exp(shifted),1); confidence = max(probability,[],1);
fig = new_figure(options, task + " classification confidence");
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
nexttile; histogram(confidence(predicted==truth),20,'DisplayName','Correct'); hold on;
histogram(confidence(predicted~=truth),20,'DisplayName','Incorrect'); hold off;
xlabel('Maximum softmax probability'); ylabel('Samples'); grid on; legend; title('Prediction confidence');
nexttile; bar([sum(confusion,2), sum(confusion,1).']); grid on; xlabel('Class');
ylabel('Samples'); legend({'True','Predicted'},'Location','best'); title('Held-out class counts');
save_figure(fig, options, 'held_out_classification_details.png'); figures(end+1)=fig;

if any(task == ["mnist","afro_mnist_vai"])
    fig = plot_image_examples(representative, task, options);
    if isgraphics(fig), figures(end+1)=fig; end
end
end

function fig = plot_image_examples(result, task, options)
fig = gobjects(0);
try
    [data,~] = banff_data('static', result.config, result.data_information);
    X = double(data.X_test); side = round(sqrt(size(X,1)));
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
    fig = new_figure(options, task + " held-out images");
    tiledlayout(rows,columns,'TileSpacing','compact','Padding','compact');
    for index = 1:numel(selection)
        sample = selection(index); nexttile;
        imagesc(reshape(X(:,sample),side,side).'); axis image off; colormap(gray);
        title(sprintf('T:%d P:%d',truth(sample)-1,predicted(sample)-1), ...
            'Color', ternary(truth(sample)==predicted(sample),[0 .5 0],[.8 0 0]));
    end
    sgtitle(sprintf('%s normalized held-out examples',task_title(task)));
    save_figure(fig, options, 'held_out_image_examples.png');
catch exception
    warning('banff:evaluationImagePlot','Image examples were skipped: %s',exception.message);
end
end

function figures = plot_regression(results, task, options)
figures = gobjects(0); representative = pick_result(results, options);
S = representative.test.statistics; truth = double(S.truth(:)); prediction = double(S.prediction(:));
errorValue = prediction-truth;
fig = new_figure(options, task + " held-out regression");
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
nexttile; scatter(truth,prediction,18,'filled','MarkerFaceAlpha',.45); hold on;
limits = finite_limits([truth;prediction]); plot(limits,limits,'k--','LineWidth',1.2); hold off;
axis square; xlim(limits); ylim(limits); grid on; xlabel('Truth'); ylabel('Prediction');
title(sprintf('Held-out predictions, seed %g',representative.config.seed));
nexttile; scatter(truth,errorValue,18,'filled','MarkerFaceAlpha',.45); yline(0,'k--');
grid on; xlabel('Truth'); ylabel('Prediction - truth'); title('Residuals versus truth');
save_figure(fig, options, 'held_out_prediction_and_residuals.png'); figures(end+1)=fig;

fig = new_figure(options, task + " residual distribution");
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
nexttile; histogram(errorValue,30); xline(mean(errorValue),'r-','LineWidth',1.4);
grid on; xlabel('Prediction - truth'); ylabel('Samples'); title('Signed-error distribution');
nexttile; errorbar(1:numel(results), arrayfun(@(R) double(R.test.statistics.signed_error_mean),results), ...
    arrayfun(@(R) double(R.test.statistics.signed_error_std),results), 'o','LineWidth',1.3);
grid on; xlim([.5 numel(results)+.5]); set(gca,'XTick',1:numel(results), ...
    'XTickLabel',arrayfun(@(R) string(R.config.seed),results));
xlabel('Initialisation seed'); ylabel('Signed error mean +/- SD'); title('Residual summary by seed');
save_figure(fig, options, 'held_out_residual_distribution.png'); figures(end+1)=fig;
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
plot(1:maxIC,distance.','o-','LineWidth',1.2); grid on; xlabel('Held-out initial condition');
ylabel('Phase-space distance'); title('Per-initial-condition closed-loop distance');
legend(arrayfun(@(R) sprintf('Seed %g',R.config.seed),results,'UniformOutput',false),'Location','best');
save_figure(fig, options, 'phase_distance_by_initial_condition.png'); figures(end+1)=fig;

representative = pick_result(results,options);
for ic=1:numel(representative.test.prediction)
    prediction=double(representative.test.prediction{ic}); truth=double(representative.test.truth{ic});
    if ~isequal(size(prediction),size(truth))
        error('banff:evaluationTrajectoryShape', ...
            'Saved prediction and truth trajectories have different shapes.');
    end
    n=size(prediction,1);
    time=(0:n-1).'*double(representative.config.dt);
    fig = new_figure(options, sprintf('%s time series IC %d',task,ic));
    tiledlayout(size(truth,2),1,'TileSpacing','compact','Padding','compact');
    for dimension=1:size(truth,2)
        nexttile; plot(time,truth(:,dimension),'k-','LineWidth',1.2); hold on;
        plot(time,prediction(:,dimension),'r--','LineWidth',1.1); hold off; grid on;
        ylabel(sprintf('x_%d',dimension)); if dimension==1, legend({'True','Network'},'Location','best'); end
        if dimension==size(truth,2), xlabel('Time (s)'); end
    end
    sgtitle(sprintf('%s closed loop, seed %g, test IC %d',task_title(task),representative.config.seed,ic));
    save_figure(fig,options,sprintf('trajectory_time_series_ic%02d.png',ic)); figures(end+1)=fig;
    if size(truth,2)>=2
        pairs=nchoosek(1:size(truth,2),2); fig=new_figure(options,sprintf('%s phase IC %d',task,ic));
        tiledlayout(size(pairs,1),2,'TileSpacing','compact','Padding','compact');
        for pair=1:size(pairs,1)
            a=pairs(pair,1); b=pairs(pair,2); limits=phase_limits([truth(:,[a b]);prediction(:,[a b])]);
            nexttile; plot(truth(:,a),truth(:,b),'k-','LineWidth',1.1); axis equal; grid on;
            xlim(limits(1:2)); ylim(limits(3:4)); xlabel(sprintf('x_%d',a)); ylabel(sprintf('x_%d',b)); title('True');
            nexttile; plot(prediction(:,a),prediction(:,b),'r--','LineWidth',1.1); axis equal; grid on;
            xlim(limits(1:2)); ylim(limits(3:4)); xlabel(sprintf('x_%d',a)); ylabel(sprintf('x_%d',b)); title('Network');
        end
        sgtitle(sprintf('%s phase portraits, seed %g, test IC %d',task_title(task),representative.config.seed,ic));
        save_figure(fig,options,sprintf('trajectory_phase_portraits_ic%02d.png',ic)); figures(end+1)=fig;
    end
end
end

function fig = plot_activity(results, task, kind, options)
representative=pick_result(results,options);
if kind ~= "dynamics"
    rates=double(representative.test.neural_activity.mean_firing_rate_by_neuron_hz(:));
    fig=new_figure(options,task+" spiking activity");
    tiledlayout(2,3,'TileSpacing','compact','Padding','compact');
    nexttile; histogram(rates,50); grid on; xlabel('Mean firing rate (Hz)'); ylabel('Neurons');
    title(sprintf('Held-out rates; %.2f%% active',representative.test.neural_activity.active_fraction_percent));
    if options.replay_static_spikes
        plot_static_diagnostics(representative,options);
    else
        nexttile;
        plot(sort(rates,'descend'),'LineWidth',1.1); xlabel('Ranked neuron'); ylabel('Mean rate (Hz)'); grid on;
        title('Ranked firing rates');
        for index=1:4, nexttile; axis off; end
    end
    save_figure(fig,options,'representative_spiking_diagnostics.png');
else
    events=representative.test.events{min(options.representative_initial_condition,numel(representative.test.events))};
    fig=new_figure(options,task+" spiking activity");
    tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
    nexttile; plot_event_raster(events,representative.config,options);
    [eventRates, eventIsi] = event_statistics(events,representative.config);
    nexttile; histogram(eventRates,50); grid on; xlabel('Firing rate (Hz)'); ylabel('Active neurons');
    title(sprintf('Event rates; %.2f%% active',100*numel(eventRates)/representative.config.N_hidden));
    nexttile;
    inverseIsiRate = inverse_isi_rate_hz(eventIsi);
    if isempty(inverseIsiRate), axis off; text(.5,.5,'No repeated-neuron spikes','HorizontalAlignment','center');
    else, histogram(inverseIsiRate,50); grid on; xlabel('Inverse ISI (Hz)'); ylabel('Intervals'); title('Instantaneous rate distribution'); end
    nexttile; rho=double(events.rho(:)); rho=rho(isfinite(rho));
    if isempty(rho), axis off; text(.5,.5,'No recorded spikes','HorizontalAlignment','center');
    else, histogram(rho,40); grid on; xlabel('\rho at spike'); ylabel('Spikes'); title('Within-step event fraction'); end
    fprintf('Representative dynamics seed %g: %.2f%% active, %.2f%% silent neurons; %d spikes.\n', ...
        representative.config.seed,100*numel(eventRates)/representative.config.N_hidden, ...
        100*(1-numel(eventRates)/representative.config.N_hidden),numel(events.step));
    save_figure(fig,options,'representative_spiking_diagnostics.png');
end
end

function plot_static_diagnostics(result,options)
try
    [data,~]=banff_data('static',result.config,result.data_information);
    P=banff_model('create',size(data.X_train,1),size(data.Y_train,1),result.config);
    P.B=single(result.best.B);
    sampleCount=min(options.max_spike_samples,size(data.X_test,2));
    [spikes,voltage,current]=banff_plot('static_traces',P,data.X_test(:,1:sampleCount),result.config);
    duration=double(result.config.presentation_steps)*double(result.config.dt);
    replayRates=sum(spikes,[2 3])./(sampleCount*duration);
    activeBySample=squeeze(mean(any(spikes,2),1))*100;
    fprintf(['Representative static seed %g: %.2f%% active, %.2f%% silent neurons over %d samples; ', ...
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
        xlabel('Presentation step'); ylabel('Ranked active neuron'); title('Representative held-out spike raster');
    end
    nexttile;
    traceKeep=keep(1:min(20,numel(keep))); plot(double(voltage(traceKeep,:)).','LineWidth',.7); grid on;
    xlabel('Presentation step'); ylabel('Voltage (mV)'); title('Representative neuron voltages');
    nexttile; isi=spike_isi(spikes(:,:,1),double(result.config.dt));
    inverseIsiRate=inverse_isi_rate_hz(isi);
    if isempty(inverseIsiRate), axis off; text(.5,.5,'No repeated-neuron spikes','HorizontalAlignment','center');
    else, histogram(inverseIsiRate,50); grid on; xlabel('Inverse ISI (Hz)'); ylabel('Intervals'); title('Instantaneous rate distribution'); end
    nexttile;
    t=double(current.time_seconds); plot(t,double(current.mean_encoder_current),'LineWidth',1.1); hold on;
    plot(t,double(current.mean_recurrent_current),'LineWidth',1.1);
    plot(t,double(current.mean_bias_current),'LineWidth',1.1);
    plot(t,double(current.mean_adaptation_current),'LineWidth',1.1); hold off; grid on;
    xlabel('Time (s)'); ylabel('Population-mean current (mV)'); title('Current components');
    legend({'Encoder','Recurrent','Bias','Adaptation'},'Location','best');
catch exception
    warning('banff:evaluationStaticRaster','Static raster replay was skipped: %s',exception.message);
    for index=1:5, nexttile; axis off; text(.5,.5,'Static replay unavailable','HorizontalAlignment','center'); end
end
end

function isi = spike_isi(spikeMatrix,dt)
isi=[];
for neuron=1:size(spikeMatrix,1)
    times=find(spikeMatrix(neuron,:));
    if numel(times)>1, isi=[isi,diff(times).*dt]; end %#ok<AGROW>
end
end

function rate = inverse_isi_rate_hz(isi)
% Convert positive inter-spike intervals in seconds to instantaneous rates.
% The result is an interval-weighted distribution of 1/ISI and is distinct
% from the per-neuron mean firing rates shown in the adjacent panels.
isi=double(isi(:));
isi=isi(isfinite(isi) & isi>0);
rate=1./isi;
end

function [rates,isi] = event_statistics(events,cfg)
neuron=double(events.neuron(:)); step=double(events.step(:));
% Report activity only over the scored test interval. Warmup spikes are useful
% for trajectory initialization and raster context, but must not contribute to
% firing rates, active-neuron fractions, or inter-spike intervals.
warmupSteps=round(double(cfg.test_warmup_time)/double(cfg.dt));
recordingSteps=round(double(cfg.test_time)/double(cfg.dt));
lastRecordingStep=warmupSteps+recordingSteps;
if any(step < 1 | step > lastRecordingStep)
    error('banff:evaluationEventStep', ...
        'A recorded spike lies outside the configured evaluation interval.');
end
scored=step>warmupSteps;
neuron=neuron(scored);
step=step(scored)-warmupSteps;
if isempty(neuron), rates=[]; isi=[]; return; end
[uniqueNeuron,~,group]=unique(neuron); counts=accumarray(group,1);
duration=max(1,recordingSteps)*double(cfg.dt); rates=counts./duration; isi=[];
for index=1:numel(uniqueNeuron)
    times=sort(step(group==index));
    if numel(times)>1, isi=[isi;diff(times).*double(cfg.dt)]; end %#ok<AGROW>
end
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

function R = pick_result(results,options)
R=results(min(options.representative_seed_index,numel(results)));
end

function fig = new_figure(options,name)
properties={'Color','w','Name',char(name)};
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
