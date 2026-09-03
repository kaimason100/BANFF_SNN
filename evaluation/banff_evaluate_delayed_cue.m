function report = banff_evaluate_delayed_cue(seeds,profile,overrides,displayOptions)
%BANFF_EVALUATE_DELAYED_CUE Evaluate memory and recurrent dependence.
%   REPORT = BANFF_EVALUATE_DELAYED_CUE(SEEDS,PROFILE,OVERRIDES,OPTIONS)
%   tests completed delayed-cue models and compares the intact network with
%   zero recurrent current and cue-removed controls. A model demonstrates the
%   intended recurrent-memory regime only if it satisfies all predeclared
%   accuracy, ablation-drop, and cue-removal criteria.

if nargin<1 || isempty(seeds), seeds=1:3; end
if nargin<2 || isempty(profile), profile="main"; end
if nargin<3 || isempty(overrides), overrides=struct(); end
if nargin<4 || isempty(displayOptions), displayOptions=struct(); end
root=fileparts(fileparts(mfilename('fullpath')));
addpath(root);
options=default_options(root,displayOptions);
results=run_experiment('test','delayed_cue',profile,seeds,overrides);
results=results(:).';

n=numel(results);
fullAccuracy=nan(n,1); ablatedAccuracy=nan(n,1); cueRemovedAccuracy=nan(n,1);
fullLoss=nan(n,1); ablatedLoss=nan(n,1); cueRemovedLoss=nan(n,1);
meanRate=nan(n,1); activePercent=nan(n,1);
for index=1:n
    result=results(index); cfg=result.config;
    [data,~]=banff_data('temporal',cfg,result.data_information);
    P=banff_model('create',size(data.X_train,1),size(data.Y_train,1),cfg);
    P.B=single(result.best.B);
    P=banff_model('gpu',P);
    full=result.test;
    ablatedP=remove_recurrence(P);
    ablated=banff_eval('temporal',ablatedP,data.X_test,data.Y_test,cfg,true);
    cueRemovedX=data.X_test;
    cueRemovedX(1,1:cfg.sequence_cue_steps,:)=single(0);
    cueRemoved=banff_eval('temporal',P,cueRemovedX,data.Y_test,cfg,true);
    fullAccuracy(index)=double(full.metric);
    ablatedAccuracy(index)=double(ablated.metric);
    cueRemovedAccuracy(index)=double(cueRemoved.metric);
    fullLoss(index)=double(full.loss);
    ablatedLoss(index)=double(ablated.loss);
    cueRemovedLoss(index)=double(cueRemoved.loss);
    rates=double(full.neural_activity.mean_firing_rate_by_neuron_hz);
    meanRate(index)=mean(rates);
    activePercent(index)=double(full.neural_activity.active_fraction_percent);
end

drop=fullAccuracy-ablatedAccuracy;
passesAccuracy=fullAccuracy>=options.minimum_full_accuracy_percent;
passesAblation=drop>=options.minimum_ablation_drop_points;
passesCueControl=cueRemovedAccuracy<=options.maximum_cue_removed_accuracy_percent;
demonstratesRecurrence=passesAccuracy & passesAblation & passesCueControl;
seed=double(seeds(:));
summary=table(seed,fullAccuracy,ablatedAccuracy,drop,cueRemovedAccuracy, ...
    fullLoss,ablatedLoss,cueRemovedLoss,meanRate,activePercent, ...
    passesAccuracy,passesAblation,passesCueControl,demonstratesRecurrence, ...
    'VariableNames',{'Seed','FullAccuracyPercent','ZeroRecurrenceAccuracyPercent', ...
    'AblationDropPoints','CueRemovedAccuracyPercent','FullLoss', ...
    'ZeroRecurrenceLoss','CueRemovedLoss','MeanRateHz','ActiveNeuronPercent', ...
    'PassesAccuracy','PassesAblation','PassesCueControl', ...
    'DemonstratesRecurrentDependence'});
fprintf('\nDelayed cue-response recurrent-memory evaluation\n');
fprintf(['Criteria: full accuracy >= %.1f%%, recurrence-ablation drop >= %.1f ', ...
    'points, cue-removed accuracy <= %.1f%%.\n'], ...
    options.minimum_full_accuracy_percent,options.minimum_ablation_drop_points, ...
    options.maximum_cue_removed_accuracy_percent);
disp(summary);

figures=gobjects(0);
figures(end+1)=plot_accuracy(summary,options);
figures(end+1)=plot_training(results,options);
figures(end+1)=plot_biases(results,options);
for index=1:numel(results)
    figures(end+1)=plot_example_trace(results(index),options); %#ok<AGROW>
end
report=struct('task',"delayed_cue",'results',results,'summary',summary, ...
    'criteria',rmfield(options,intersect(fieldnames(options), ...
    {'save_figures','output_directory','figure_visibility'})), ...
    'figures',figures(isgraphics(figures)),'display_options',options);
end

function options=default_options(root,changes)
options=struct('minimum_full_accuracy_percent',80, ...
    'minimum_ablation_drop_points',20, ...
    'maximum_cue_removed_accuracy_percent',60, ...
    'save_figures',false,'output_directory', ...
    fullfile(root,'outputs','evaluation','delayed_cue'), ...
    'figure_visibility',"on");
names=fieldnames(changes);
for index=1:numel(names), options.(names{index})=changes.(names{index}); end
if options.save_figures && exist(options.output_directory,'dir')~=7
    mkdir(options.output_directory);
end
end

function P=remove_recurrence(P)
if P.recurrent_mode=="low_rank"
    P.recurrentGain=single(0);
    P.self_coupling(:)=single(0);
else
    P.W_recurrent=sparse(P.N_hidden,P.N_hidden);
    if isa(P.B,'gpuArray'), P.W_recurrent=gpuArray(P.W_recurrent); end
end
end

function fig=plot_accuracy(summary,options)
fig=new_figure(options);
values=[summary.FullAccuracyPercent summary.ZeroRecurrenceAccuracyPercent ...
    summary.CueRemovedAccuracyPercent];
bar(categorical("Seed "+string(summary.Seed)),values); grid on;
yline(50,'k:','Chance'); ylim([0 100]); ylabel('Test accuracy (%)');
legend({'Full network','Zero recurrent current','Cue removed'},'Location','best');
title('Delayed cue-response dependence controls');
save_figure(fig,options,'accuracy_and_recurrence_controls.png');
end

function fig=plot_training(results,options)
fig=new_figure(options);
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
nexttile; hold on;
for index=1:numel(results)
    y=double(results(index).history.train_loss(:));
    plot(find(isfinite(y)),y(isfinite(y)),'DisplayName',sprintf('Seed %g',results(index).config.seed));
end
hold off; grid on; xlabel('Epoch'); ylabel('Cross-entropy'); title('Training loss'); legend('Location','best');
nexttile; hold on;
for index=1:numel(results)
    y=double(results(index).history.validation_metric(:));
    plot(find(isfinite(y)),y(isfinite(y)),'DisplayName',sprintf('Seed %g',results(index).config.seed));
end
hold off; grid on; ylim([0 100]); yline(50,'k:'); xlabel('Epoch');
ylabel('Validation accuracy (%)'); title('Held-out validation'); legend('Location','best');
save_figure(fig,options,'training_history.png');
end

function fig=plot_biases(results,options)
fig=new_figure(options); hold on;
colors=lines(numel(results));
for index=1:numel(results)
    values=double(results(index).best.B(:));
    sample=unique(round(linspace(1,numel(values),min(2500,numel(values)))));
    swarmchart(repmat(index,numel(sample),1),values(sample),5,colors(index,:), ...
        'filled','XJitter','density','XJitterWidth',.25, ...
        'MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.1);
    boxchart(repmat(index,numel(values),1),values,'BoxFaceColor',colors(index,:), ...
        'MarkerStyle','none');
end
hold off; grid on; set(gca,'XTick',1:numel(results), ...
    'XTickLabel',"Seed "+string(arrayfun(@(R)R.config.seed,results)));
ylabel('Learned bias current (mV)'); title('Learned bias distributions');
save_figure(fig,options,'learned_biases.png');
end

function fig=plot_example_trace(result,options)
cfg=result.config; [data,~]=banff_data('temporal',cfg,result.data_information);
[~,labels]=max(data.Y_test,[],1);
indices=[find(labels==1,1) find(labels==2,1)];
P=banff_model('create',size(data.X_train,1),size(data.Y_train,1),cfg);
P.B=single(result.best.B); P=banff_model('gpu',P);
[~,~,~,trace]=banff_model('temporal',P,data.X_test(:,:,indices), ...
    cfg.sequence_response_steps,false,false);
trace=gather(trace); shifted=trace-max(trace,[],1);
probability=exp(shifted)./sum(exp(shifted),1);
time=(0:size(trace,2)-1).*double(cfg.dt);
fig=new_figure(options);
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');
for column=1:2
    nexttile; plot(time,squeeze(probability(:,:,column)).','LineWidth',1.1); grid on;
    xline(double(cfg.sequence_cue_steps)*double(cfg.dt),'k:','Cue off');
    xline(double(cfg.sequence_cue_steps+cfg.sequence_delay_steps)*double(cfg.dt), ...
        'k--','Response'); ylim([0 1]); ylabel('Decoder probability');
    title(sprintf('True class %d example',column));
end
xlabel('Time (s)'); legend({'Class 1','Class 2'},'Location','best');
sgtitle(sprintf('Delayed cue-response memory traces, seed %g',cfg.seed));
save_figure(fig,options,sprintf('seed%03d_example_memory_traces.png', ...
    round(double(cfg.seed))));
end

function fig=new_figure(options)
% Visible figures omit the explicit Visible property so the Live Editor can
% capture them in the .mlx output panel; hidden figures remain exportable.
properties={'Color','w'};
if ~strcmpi(string(options.figure_visibility),"on")
    properties=[properties,{'Visible',char(options.figure_visibility)}]; %#ok<AGROW>
end
fig=figure(properties{:});
end

function save_figure(fig,options,name)
if ~options.save_figures, return; end
path=fullfile(options.output_directory,name);
if exist('exportgraphics','file')==2
    exportgraphics(fig,path,'Resolution',180);
else
    saveas(fig,path);
end
end
