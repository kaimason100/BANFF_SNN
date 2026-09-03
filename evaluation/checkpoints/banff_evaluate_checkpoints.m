function report = banff_evaluate_checkpoints(task,seeds,profile,overrides,displayOptions)
%BANFF_EVALUATE_CHECKPOINTS Read-only evaluation of interrupted training runs.
%   REPORT = BANFF_EVALUATE_CHECKPOINTS(TASK,SEEDS,PROFILE,OVERRIDES,OPTIONS)
%   locates the time-limit checkpoint belonging to each requested experiment,
%   reconstructs the current and validation-selected network states, and
%   evaluates them without resuming training or modifying the checkpoint.
%
%   Validation is used by default so repeated monitoring cannot leak held-out
%   test information into training decisions. Set OPTIONS.assessment_split to
%   "test" only for a deliberately declared interim test analysis. Explicit
%   files may be supplied through OPTIONS.checkpoint_files.

if nargin<2 || isempty(seeds), seeds=1:3; end
if nargin<3 || isempty(profile), profile="main"; end
if nargin<4 || isempty(overrides), overrides=struct(); end
if nargin<5 || isempty(displayOptions), displayOptions=struct(); end

checkpointDirectory=fileparts(mfilename('fullpath'));
evaluationDirectory=fileparts(checkpointDirectory);
root=fileparts(evaluationDirectory);
addpath(root,evaluationDirectory,checkpointDirectory);
task=canonical_task_local(task);
options=default_options(root,task,displayOptions);

if ~canUseGPU
    error('banff:checkpointEvaluationGPU', ...
        'Checkpoint evaluation uses the production simulator and requires a supported GPU.');
end
if options.assessment_split=="test"
    warning('banff:checkpointInterimTest', ...
        ['Held-out test evaluation is enabled. Do not use these results to ', ...
        'choose epochs, gains, architectures, or stopping decisions.']);
end

[files,expectedConfigs]=resolve_checkpoint_files(task,seeds,profile,overrides,options);
entries=struct([]);
figures=gobjects(0);
missing=strings(0,1);

fprintf('\nBANFF saved-checkpoint evaluation: %s (%s profile)\n',task,profile);
fprintf('Assessment split: %s | checkpoint state for full diagnostics: %s\n\n', ...
    options.assessment_split,options.checkpoint_state);

for fileIndex=1:numel(files)
    file=files(fileIndex);
    if exist(file,'file')~=2
        if options.skip_missing
            missing(end+1,1)=file; %#ok<AGROW>
            warning('banff:checkpointMissing','Skipping missing checkpoint %s.',file);
            continue;
        end
        error('banff:checkpointMissing','Could not find checkpoint %s.',file);
    end

    expected=[];
    if fileIndex<=numel(expectedConfigs), expected=expectedConfigs{fileIndex}; end
    entry=evaluate_one_checkpoint(file,expected,task,options);
    if isempty(entries), entries=entry; else, entries(end+1)=entry; end %#ok<AGROW>

    figures(end+1)=plot_validation_comparison(entry,options); %#ok<AGROW>
end

if isempty(entries)
    error('banff:noCheckpointsFound', ...
        'No requested checkpoint files were found. See the warnings above.');
end

summary=checkpoint_summary(entries);
fprintf('\nCheckpoint progress and independently recomputed validation performance\n');
disp(summary);
if options.assessment_split=="test"
    fprintf('Explicit interim held-out test results\n');
    disp(test_summary(entries));
end

fullResults=checkpoint_results_for_full_evaluation(entries,options);
fullOptions=options;
fullOptions.preloaded_results=fullResults;
fullOptions.output_directory=fullfile(options.output_directory, ...
    char(options.assessment_split),char(options.checkpoint_state));
actualSeeds=arrayfun(@(entry)double(entry.config.seed),entries);
fullReport=banff_evaluate_task(task,actualSeeds,profile,overrides,fullOptions);
figures=[figures fullReport.figures];

report=struct('task',task,'profile',string(profile),'entries',entries, ...
    'summary_table',summary,'missing_files',missing, ...
    'seed_table',fullReport.seed_table,'metric_summary_table', ...
    fullReport.summary_table,'recurrent_ablation',fullReport.recurrent_ablation, ...
    'recurrent_ablation_details',fullReport.recurrent_ablation_details, ...
    'full_evaluation',fullReport,'figures',figures(isgraphics(figures)), ...
    'display_options',options);
end

function options = default_options(root,task,changes)
options=struct();
options.checkpoint_files=strings(0,1);
options.skip_missing=true;
options.assessment_split="validation";
options.checkpoint_state="current";
options.max_bias_points=2500;
options.save_figures=false;
options.output_directory=fullfile(root,'outputs','checkpoint_evaluation',char(task));
options.figure_visibility="on";
names=fieldnames(changes);
for index=1:numel(names), options.(names{index})=changes.(names{index}); end
% Retain the first release's Boolean switch as a compatibility alias.
if isfield(changes,'evaluate_test')
    if logical(changes.evaluate_test), options.assessment_split="test";
    else, options.assessment_split="validation"; end
end
options.assessment_split=lower(string(options.assessment_split));
options.checkpoint_state=lower(string(options.checkpoint_state));
if ~any(options.assessment_split==["validation","test"])
    error('banff:checkpointSplit','assessment_split must be validation or test.');
end
if ~any(options.checkpoint_state==["current","best"])
    error('banff:checkpointStateChoice','checkpoint_state must be current or best.');
end
options.max_bias_points=max(1,round(double(options.max_bias_points)));
if options.save_figures && exist(options.output_directory,'dir')~=7
    mkdir(options.output_directory);
end
end

function [files,expectedConfigs] = ...
        resolve_checkpoint_files(task,seeds,profile,overrides,options)
explicit=string(options.checkpoint_files(:));
explicit=explicit(strlength(explicit)>0);
if ~isempty(explicit)
    files=explicit;
    expectedConfigs=cell(size(files));
    return;
end
files=strings(numel(seeds),1);
expectedConfigs=cell(numel(seeds),1);
profileOptions=profile_overrides(profile,overrides);
for index=1:numel(seeds)
    oneOptions=profileOptions;
    oneOptions.seed=seeds(index);
    cfg=banff('config',task,oneOptions);
    [folder,name]=fileparts(cfg.model_file);
    files(index)=string(fullfile(folder,[name '_checkpoint.mat']));
    expectedConfigs{index}=cfg;
end
end

function options = profile_overrides(profile,overrides)
options=overrides;
switch lower(string(profile))
    case "main"
        options.method="eprop";
        options.recurrent_mode="low_rank";
        options.training_profile="main";
    case "full_rank"
        options.method="eprop";
        options.recurrent_mode="full_rank";
        if ~isfield(options,'N_hidden'), options.N_hidden=6000; end
    case "spsa"
        options.method="spsa";
        options.recurrent_mode="low_rank";
    case "neuron_sweep"
        options.method="eprop";
        options.recurrent_mode="low_rank";
        options.training_profile="neuron_sweep";
        if ~isfield(options,'N_hidden')
            error('banff:checkpointNeuronSweepSize', ...
                'neuron_sweep requires overrides.N_hidden.');
        end
    otherwise
        error('banff:checkpointProfile', ...
            'profile must be main, full_rank, spsa or neuron_sweep.');
end
end

function entry = evaluate_one_checkpoint(file,expectedCfg,requestedTask,options)
loaded=load(file,'checkpoint');
if ~isfield(loaded,'checkpoint') || ~isstruct(loaded.checkpoint)
    error('banff:checkpointVariable','%s does not contain a checkpoint structure.',file);
end
checkpoint=loaded.checkpoint;
required={'epoch','state','history','best','config'};
missing=required(~isfield(checkpoint,required));
if ~isempty(missing)
    error('banff:checkpointFields','Checkpoint %s lacks: %s.', ...
        file,strjoin(missing,', '));
end
cfg=checkpoint.config;
if canonical_task_local(cfg.task)~=requestedTask
    error('banff:checkpointTaskMismatch', ...
        'Checkpoint task %s does not match requested task %s.',cfg.task,requestedTask);
end
if ~isempty(expectedCfg) && ...
        (~isfield(cfg,'checkpoint_config_sha256') || ...
        ~strcmp(cfg.checkpoint_config_sha256,expectedCfg.checkpoint_config_sha256))
    error('banff:checkpointConfigurationMismatch', ...
        ['Checkpoint %s does not match the requested profile, seed, or ', ...
        'scientific overrides.'],file);
end
assert_checkpoint_source(checkpoint);
validate_checkpoint_shapes(checkpoint,file);

[models,dataInformation,data]=reconstruct_models(checkpoint);
[currentValidation,bestValidation]=evaluate_split( ...
    models,cfg,dataInformation,data,"validation",false);
if options.assessment_split=="test"
    [currentTest,bestTest]=evaluate_split( ...
        models,cfg,dataInformation,data,"test",false);
else
    currentTest=[]; bestTest=[];
end

entry=struct('file',char(file),'checkpoint',checkpoint,'config',cfg, ...
    'epoch',double(checkpoint.epoch),'fraction_complete', ...
    double(checkpoint.epoch)/double(cfg.epochs), ...
    'current_bias',gather(models.current.B), ...
    'best_bias',gather_best_bias(models), ...
    'data_information',dataInformation,'validation_current',currentValidation, ...
    'validation_best',bestValidation,'test_current',currentTest, ...
    'test_best',bestTest);
fprintf('Loaded seed %d: epoch %d/%d (%.2f%%), best epoch %d\n', ...
    cfg.seed,checkpoint.epoch,cfg.epochs,100*entry.fraction_complete, ...
    double(checkpoint.best.epoch));
end

function assert_checkpoint_source(checkpoint)
if isfield(checkpoint,'training_source_sha256')
    saved=struct('training_source_sha256',checkpoint.training_source_sha256);
elseif isfield(checkpoint,'core_source_sha256')
    saved=struct('core_source_sha256',checkpoint.core_source_sha256);
else
    error('banff:checkpointProvenanceMissing', ...
        'Checkpoint has no source-code provenance.');
end
banff_provenance('assert_training_compatible',saved,checkpoint.config);
end

function validate_checkpoint_shapes(checkpoint,file)
cfg=checkpoint.config;
stateRequired={'B','m','v','vMax','adamStep'};
missing=stateRequired(~isfield(checkpoint.state,stateRequired));
if ~isempty(missing)
    error('banff:checkpointStateFields','Checkpoint %s state lacks: %s.', ...
        file,strjoin(missing,', '));
end
vectors={'B','m','v','vMax'};
for index=1:numel(vectors)
    value=checkpoint.state.(vectors{index});
    if numel(value)~=cfg.N_hidden || any(~isfinite(value),'all')
        error('banff:checkpointStateShape', ...
            'Checkpoint %s has an invalid %s vector.',file,vectors{index});
    end
end
if checkpoint.epoch<1 || checkpoint.epoch>cfg.epochs || ...
        checkpoint.epoch~=round(checkpoint.epoch)
    error('banff:checkpointEpoch','Checkpoint %s has an invalid epoch.',file);
end
if ~isfield(checkpoint.best,'epoch') || ~isfield(checkpoint.best,'B') || ...
        (~isempty(checkpoint.best.B) && numel(checkpoint.best.B)~=cfg.N_hidden)
    error('banff:checkpointBest','Checkpoint %s has an invalid best state.',file);
end
end

function [models,dataInformation,data] = reconstruct_models(checkpoint)
cfg=checkpoint.config;
if cfg.kind=="dynamics"
    [~,dataInformation]=banff_data('dynamics',cfg);
    dimension=numel(dataInformation.mean);
    data=[];
    base=banff_model('create',dimension,dimension,cfg);
else
    [data,dataInformation]=banff_data('static',cfg);
    base=banff_model('create',size(data.X_train,1),size(data.Y_train,1),cfg);
end
models=struct();
models.current=base;
models.current.B=single(checkpoint.state.B);
models.has_best=~isempty(checkpoint.best.B) && checkpoint.best.epoch>0;
if models.has_best
    models.best=base;
    models.best.B=single(checkpoint.best.B);
else
    models.best=[];
end
end

function [currentEvaluation,bestEvaluation] = ...
        evaluate_split(models,cfg,dataInformation,data,role,recordEvents)
if cfg.kind=="dynamics"
    currentEvaluation=banff_eval('closed_loop', ...
        banff_model('gpu',models.current),cfg,dataInformation,role,recordEvents);
    if models.has_best
        bestEvaluation=banff_eval('closed_loop', ...
            banff_model('gpu',models.best),cfg,dataInformation,role,recordEvents);
    else
        bestEvaluation=[];
    end
else
    if role=="validation"
        X=data.X_validation; Y=data.Y_validation;
    else
        X=data.X_test; Y=data.Y_test;
    end
    currentEvaluation=complete_static_evaluation( ...
        models.current,X,Y,data,cfg,role);
    if models.has_best
        bestEvaluation=complete_static_evaluation(models.best,X,Y,data,cfg,role);
    else
        bestEvaluation=[];
    end
end
end

function evaluation = complete_static_evaluation(P,X,Y,data,cfg,role)
evaluation=banff_eval('static',banff_model('gpu',P),X,Y,cfg,true);
% BANFF_EVAL's activity field is shared with final testing and therefore has
% a generic historical label. Record the actual split used by this monitor.
evaluation.neural_activity.calculation.context=char(role);
if cfg.kind=="classification"
    evaluation.statistics=banff_metrics('classification',evaluation.output,Y);
else
    evaluation.statistics=banff_metrics('regression',evaluation.output,Y, ...
        data.target_mean,data.target_std);
end
end

function results = checkpoint_results_for_full_evaluation(entries,options)
% Adapt checkpoint states to the completed-model evaluator's immutable result
% interface. The field named `test` contains the explicitly selected assessment
% split; BANFF_EVALUATE_TASK labels it from OPTIONS.assessment_split.
results=struct([]);
for index=1:numel(entries)
    entry=entries(index);
    checkpoint=entry.checkpoint;
    useBest=options.checkpoint_state=="best";
    if useBest && isempty(entry.best_bias)
        error('banff:checkpointBestUnavailable', ...
            'Seed %d has no validation-selected best state yet.',entry.config.seed);
    end

    if options.assessment_split=="validation"
        if useBest, assessment=entry.validation_best;
        else, assessment=entry.validation_current; end
    else
        if useBest, assessment=entry.test_best;
        else, assessment=entry.test_current; end
    end

    % Event rasters and rho/ISI diagnostics require a recorded dynamics pass.
    % Static evaluations already retain outputs and activity counts.
    if entry.config.kind=="dynamics"
        [models,dataInformation,data]=reconstruct_models(checkpoint);
        [currentRecorded,bestRecorded]=evaluate_split(models,entry.config, ...
            dataInformation,data,options.assessment_split,true);
        if useBest, assessment=bestRecorded; else, assessment=currentRecorded; end
    end

    selectedBest=checkpoint.best;
    if useBest
        selectedBest.B=entry.best_bias;
    else
        selectedBest.B=entry.current_bias;
    end
    provenance=struct();
    if isfield(checkpoint,'core_source_sha256')
        provenance.core_source_sha256=checkpoint.core_source_sha256;
    else
        provenance.core_source_sha256=banff_provenance('all');
    end
    if isfield(checkpoint,'training_source_sha256')
        provenance.training_source_sha256=checkpoint.training_source_sha256;
    end
    one=struct('config',entry.config,'history',checkpoint.history, ...
        'best',selectedBest,'data_information',entry.data_information, ...
        'provenance',provenance,'test',assessment,'complete',false, ...
        'checkpoint_epoch',entry.epoch, ...
        'evaluated_checkpoint_state',options.checkpoint_state);
    if isempty(results), results=one; else, results(end+1)=one; end %#ok<AGROW>
end
end

function T = checkpoint_summary(entries)
n=numel(entries);
task=strings(n,1); seed=zeros(n,1); epoch=zeros(n,1); totalEpochs=zeros(n,1);
percentComplete=zeros(n,1); bestEpoch=zeros(n,1); metric=strings(n,1);
currentValidation=nan(n,1); bestValidation=nan(n,1); file=strings(n,1);
for index=1:n
    entry=entries(index); cfg=entry.config;
    task(index)=cfg.task; seed(index)=cfg.seed; epoch(index)=entry.epoch;
    totalEpochs(index)=cfg.epochs; percentComplete(index)=100*entry.fraction_complete;
    bestEpoch(index)=entry.checkpoint.best.epoch; metric(index)=metric_name(cfg.kind);
    currentValidation(index)=primary_metric(entry.validation_current,cfg.kind);
    if ~isempty(entry.validation_best)
        bestValidation(index)=primary_metric(entry.validation_best,cfg.kind);
    end
    file(index)=entry.file;
end
T=table(task,seed,epoch,totalEpochs,percentComplete,bestEpoch,metric, ...
    currentValidation,bestValidation,file, ...
    'VariableNames',{'Task','Seed','CheckpointEpoch','TotalEpochs', ...
    'PercentComplete','BestEpoch','ValidationMetric','CurrentValidation', ...
    'BestStateValidation','CheckpointFile'});
end

function T = test_summary(entries)
n=numel(entries); seed=zeros(n,1); metric=strings(n,1);
currentTest=nan(n,1); bestTest=nan(n,1);
for index=1:n
    seed(index)=entries(index).config.seed;
    metric(index)=metric_name(entries(index).config.kind);
    currentTest(index)=primary_metric(entries(index).test_current, ...
        entries(index).config.kind);
    if ~isempty(entries(index).test_best)
        bestTest(index)=primary_metric(entries(index).test_best, ...
            entries(index).config.kind);
    end
end
T=table(seed,metric,currentTest,bestTest,'VariableNames', ...
    {'Seed','Metric','CurrentCheckpointState','BestValidationState'});
end

function value = primary_metric(evaluation,kind)
if isempty(evaluation), value=NaN; return; end
if kind=="classification"
    value=double(evaluation.statistics.accuracy_percent);
elseif kind=="regression"
    value=double(evaluation.statistics.rmse);
else
    value=double(evaluation.phase_distance);
end
end

function name = metric_name(kind)
if kind=="classification", name="AccuracyPercent";
elseif kind=="regression", name="RMSE";
else, name="PhaseDistance";
end
end

function fig = plot_training_progress(entry,options)
cfg=entry.config; history=entry.checkpoint.history; last=entry.epoch;
fig=new_figure(options);
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
nexttile;
plot_history_field(history,'train_loss',last,'Training loss');
title(sprintf('Training loss through epoch %d',last));
nexttile;
if cfg.kind=="dynamics"
    plot_history_field(history,'validation_distance',last,'Phase-space distance');
    title('Validation phase distance');
else
    plot_history_field(history,'validation_loss',last,'Validation loss');
    title('Validation loss');
end
nexttile;
if cfg.kind=="dynamics"
    plot_history_field(history,'train_loss',last,'Training loss');
    set_log_if_positive(gca); title('Training-loss scale');
else
    plot_history_field(history,'validation_metric',last, ...
        char(metric_name(cfg.kind)));
    title('Validation metric');
end
nexttile; axis off;
lines=[string(sprintf('%s checkpoint, seed %d',char(cfg.task),cfg.seed)); ...
    string(sprintf('epoch %d / %d (%.2f%%)',last,cfg.epochs,100*entry.fraction_complete)); ...
    string(sprintf('best validation state: epoch %d',entry.checkpoint.best.epoch)); ...
    string(sprintf('current validation %s: %.6g',metric_name(cfg.kind), ...
        primary_metric(entry.validation_current,cfg.kind)))];
if ~isempty(entry.validation_best)
    lines(end+1)=string(sprintf('best-state validation %s: %.6g', ...
        metric_name(cfg.kind),primary_metric(entry.validation_best,cfg.kind)));
end
text(.02,.98,strjoin(lines,newline),'Units','normalized','VerticalAlignment','top');
sgtitle(sprintf('%s training progress, seed %d',task_title(cfg.task),cfg.seed));
save_figure(fig,options,sprintf('seed%03d_training_progress.png',cfg.seed));
end

function plot_history_field(history,field,last,yLabel)
if ~isfield(history,field)
    axis off; text(.5,.5,'History field unavailable','HorizontalAlignment','center');
    return;
end
y=double(history.(field)(1:min(last,numel(history.(field)))));
ok=isfinite(y);
if any(ok)
    plot(find(ok),y(ok),'-','LineWidth',1.1); grid on; xlabel('Epoch'); ylabel(yLabel);
else
    axis off; text(.5,.5,'No completed measurements','HorizontalAlignment','center');
end
end

function fig = plot_validation_comparison(entry,options)
cfg=entry.config; fig=new_figure(options);
current=primary_metric(entry.validation_current,cfg.kind);
best=primary_metric(entry.validation_best,cfg.kind);
bar([current best]); grid on;
set(gca,'XTick',1:2,'XTickLabel',{'Current checkpoint state','Best validation state'});
ylabel(metric_name(cfg.kind));
title(sprintf('%s validation comparison, seed %d',task_title(cfg.task),cfg.seed));
if isempty(entry.validation_best)
    text(2,0,'No validation-selected state yet','HorizontalAlignment','center');
end
save_figure(fig,options,sprintf('seed%03d_validation_comparison.png',cfg.seed));
end

function fig = plot_bias_comparison(entry,options)
fig=new_figure(options); hold on;
values={double(entry.current_bias(:))}; labels="Current";
if ~isempty(entry.best_bias)
    values{2}=double(entry.best_bias(:)); labels(2)="Best";
end
colors=lines(numel(values)); handles=gobjects(1,numel(values));
for condition=1:numel(values)
    local=values{condition};
    keep=unique(round(linspace(1,numel(local),min(numel(local),options.max_bias_points))));
    x=repmat(condition,numel(keep),1);
    swarmchart(x,local(keep),6,colors(condition,:),'filled', ...
        'XJitter','density','XJitterWidth',.55,'MarkerFaceAlpha',.20, ...
        'MarkerEdgeAlpha',.12);
    handles(condition)=plot(condition+[-.25 .25],median(local).*[1 1], ...
        'Color',colors(condition,:),'LineWidth',2,'DisplayName',labels(condition));
end
hold off; grid on; xlim([.5 numel(values)+.5]);
set(gca,'XTick',1:numel(values),'XTickLabel',labels);
ylabel('Hidden bias (mV)'); title(sprintf('Checkpoint bias states, seed %d', ...
    entry.config.seed)); legend(handles,'Location','best');
save_figure(fig,options,sprintf('seed%03d_bias_states.png',entry.config.seed));
end

function figures = plot_task_diagnostic(entry,options)
cfg=entry.config; figures=gobjects(0);
if cfg.kind=="classification"
    fig=new_figure(options); tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
    nexttile; plot_confusion(entry.validation_current.statistics,'Current');
    nexttile;
    if isempty(entry.validation_best), axis off; text(.5,.5,'No best state yet','HorizontalAlignment','center');
    else, plot_confusion(entry.validation_best.statistics,'Best validation state'); end
    sgtitle(sprintf('%s validation confusion, seed %d',task_title(cfg.task),cfg.seed));
    save_figure(fig,options,sprintf('seed%03d_validation_confusion.png',cfg.seed));
    figures=fig;
elseif cfg.kind=="regression"
    fig=new_figure(options); tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
    plot_regression_pair(entry.validation_current,'Current checkpoint state');
    if isempty(entry.validation_best), nexttile; axis off; text(.5,.5,'No best state yet','HorizontalAlignment','center');
    else, plot_regression_pair(entry.validation_best,'Best validation state'); end
    sgtitle(sprintf('%s validation predictions, seed %d',task_title(cfg.task),cfg.seed));
    save_figure(fig,options,sprintf('seed%03d_validation_regression.png',cfg.seed));
    figures=fig;
else
    fig=new_figure(options); tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
    plot_phase_pair(entry.validation_current,'Current checkpoint state');
    if isempty(entry.validation_best), nexttile; axis off; text(.5,.5,'No best state yet','HorizontalAlignment','center');
    else, plot_phase_pair(entry.validation_best,'Best validation state'); end
    sgtitle(sprintf('%s validation trajectory, seed %d',task_title(cfg.task),cfg.seed));
    save_figure(fig,options,sprintf('seed%03d_validation_trajectory.png',cfg.seed));
    figures=fig;
end
end

function plot_confusion(statistics,label)
truth=double(statistics.true_class(:));
predicted=double(statistics.predicted_class(:));
classCount=max([truth;predicted]);
matrix=accumarray([truth predicted],1,[classCount classCount]);
imagesc(double(matrix)); axis image; colorbar; grid off;
xlabel('Predicted class'); ylabel('True class'); title(label);
end

function bias = gather_best_bias(models)
if models.has_best
    bias=gather(models.best.B);
else
    bias=[];
end
end

function plot_regression_pair(evaluation,label)
nexttile;
truth=double(evaluation.statistics.truth(:)); prediction=double(evaluation.statistics.prediction(:));
scatter(truth,prediction,12,'filled','MarkerFaceAlpha',.35); hold on;
limits=finite_limits([truth;prediction]); plot(limits,limits,'k--'); hold off;
xlim(limits); ylim(limits); axis square; grid on;
xlabel('Truth'); ylabel('Prediction');
title(sprintf('%s; RMSE %.4g',label,evaluation.statistics.rmse));
end

function plot_phase_pair(evaluation,label)
nexttile;
prediction=double(evaluation.prediction{1}); truth=double(evaluation.truth{1});
if size(prediction,2)>=2
    plot(truth(:,1),truth(:,2),'k-','LineWidth',1); hold on;
    plot(prediction(:,1),prediction(:,2),'Color',[0 .447 .741],'LineWidth',1); hold off;
    xlabel('State 1'); ylabel('State 2'); axis equal; grid on;
    legend({'Truth','Network'},'Location','best');
else
    plot(truth,'k-'); hold on; plot(prediction,'LineWidth',1); hold off; grid on;
    xlabel('Timestep'); ylabel('Normalized state');
end
title(sprintf('%s; distance %.4g',label,evaluation.phase_distance));
end

function fig = plot_current_magnitudes(entry,options)
current=entry.current_magnitudes.current;
hasBest=~isempty(entry.current_magnitudes.best);
if hasBest, best=entry.current_magnitudes.best; end
fig=new_figure(options); tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
nexttile;
labels={'Encoder','Net recurrent','Adaptation','Bias deviation'};
currentValues=[current.aggregate.encoder_rms_mV,current.aggregate.net_recurrent_rms_mV, ...
    current.aggregate.adaptation_rms_mV,current.aggregate.bias_deviation_rms_mV];
if hasBest
    bestValues=[best.aggregate.encoder_rms_mV,best.aggregate.net_recurrent_rms_mV, ...
        best.aggregate.adaptation_rms_mV,best.aggregate.bias_deviation_rms_mV];
    bar([currentValues;bestValues].'); legend({'Current','Best'},'Location','best');
else
    bar(currentValues);
end
set(gca,'XTick',1:numel(labels),'XTickLabel',labels); grid on;
ylabel('Global RMS contribution (mV)'); title('Direct contribution scale');
nexttile;
afferentLabels={'Encoder gross','Recurrent gross'};
currentAfferents=[current.aggregate.gross_encoder_rms_mV, ...
    current.aggregate.gross_recurrent_rms_mV];
if hasBest
    bestAfferents=[best.aggregate.gross_encoder_rms_mV, ...
        best.aggregate.gross_recurrent_rms_mV];
    bar([currentAfferents;bestAfferents].');
    legend({'Current','Best'},'Location','best');
else
    bar(currentAfferents);
end
set(gca,'XTick',1:numel(afferentLabels),'XTickLabel',afferentLabels);
grid on; ylabel('Global RMS gross afferent magnitude (mV)');
title('Absolute afferents before cancellation');
nexttile;
currentRatio=[current.aggregate.recurrent_to_encoder_rms, ...
    current.aggregate.net_to_gross_encoder_rms, ...
    current.aggregate.net_to_gross_recurrent_rms];
if hasBest
    bestRatio=[best.aggregate.recurrent_to_encoder_rms, ...
        best.aggregate.net_to_gross_encoder_rms, ...
        best.aggregate.net_to_gross_recurrent_rms];
    bar([currentRatio;bestRatio].'); legend({'Current','Best'},'Location','best');
else
    bar(currentRatio);
end
set(gca,'XTick',1:3,'XTickLabel',{'Recurrence / encoder', ...
    'Net / gross encoder','Net / gross recurrence'});
grid on; ylabel('RMS ratio'); title('Balance and cancellation');
sgtitle(sprintf('Validation current magnitudes, seed %d',entry.config.seed));
save_figure(fig,options,sprintf('seed%03d_validation_currents.png',entry.config.seed));
end

function fig = new_figure(options)
properties={'Color','w'};
if ~strcmpi(string(options.figure_visibility),"on")
    properties=[properties,{'Visible',char(options.figure_visibility)}]; %#ok<AGROW>
end
fig=figure(properties{:});
end

function save_figure(fig,options,name)
drawnow;
if ~options.save_figures, return; end
path=fullfile(options.output_directory,name);
if exist('exportgraphics','file')==2
    exportgraphics(fig,path,'Resolution',180);
else
    saveas(fig,path);
end
end

function set_log_if_positive(ax)
lines=findobj(ax,'Type','line'); values=[];
for index=1:numel(lines), values=[values lines(index).YData]; end %#ok<AGROW>
values=values(isfinite(values));
if ~isempty(values) && all(values>0), ax.YScale='log'; end
end

function limits = finite_limits(values)
values=values(isfinite(values));
if isempty(values), limits=[-1 1]; return; end
lo=min(values); hi=max(values);
if lo==hi, padding=max(1,abs(lo))*.05; else, padding=(hi-lo)*.05; end
limits=[lo-padding hi+padding];
end

function task = canonical_task_local(task)
task=lower(replace(replace(string(task),"-","_")," ","_"));
switch task
    case {"bc","breastcancer"}, task="breast_cancer";
    case {"afromnist","afro_mnist","vai"}, task="afro_mnist_vai";
    case {"car","car_price"}, task="toyota";
    case {"sprott","sprotts"}, task="sprott_s";
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
