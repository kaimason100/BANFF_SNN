function [reports, comparison] = banff_evaluate_saved_lorenz(modelFiles, displayOptions)
%BANFF_EVALUATE_SAVED_LORENZ Evaluate explicitly selected trained models.
%   [REPORTS,COMPARISON] = BANFF_EVALUATE_SAVED_LORENZ(MODELFILES,OPTIONS)
%   loads completed Lorenz result MAT files without reconstructing their names
%   from the current default configuration. Each model is tested using its own
%   saved scientific configuration and data provenance, then passed through
%   the complete standard Lorenz analysis (activity, inverse ISIs, currents,
%   untrained comparison, recurrent ablation, trajectories and phase portraits).
%
%   MODELFILES may be a string array, character vector, or cell array of paths.
%   Omit it, or pass [], to select one or more files interactively. Running the
%   call from a Live Script keeps the generated figures in the Live Editor
%   output when MATLAB is configured to display figures inline.
%
%   Models are combined into one across-seed report only when their saved
%   scientific-configuration and training-source hashes are identical. Models
%   with different gains, architectures, or initialization regimes are analysed
%   separately, preventing scientifically inappropriate pooled summaries.

if nargin < 1 || isempty(modelFiles)
    modelFiles = select_model_files();
end
if nargin < 2 || isempty(displayOptions)
    displayOptions = struct();
end

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
modelFiles = normalize_model_files(modelFiles, root);
if isempty(modelFiles)
    error('banff:savedLorenzNoFiles', 'No trained Lorenz model files were selected.');
end

tested = cell(numel(modelFiles), 1);
groupKeys = strings(numel(modelFiles), 1);
fprintf('\nEvaluating %d explicitly selected Lorenz model file(s).\n',numel(modelFiles));
for index = 1:numel(modelFiles)
    file = modelFiles(index);
    validate_saved_result(file);
    tested{index} = banff_test(struct('model_file',char(file)));
    cfg = tested{index}.config;
    if canonical_task(cfg.task) ~= "lorenz"
        error('banff:savedLorenzTask', ...
            'Selected file is for task "%s", not Lorenz: %s',cfg.task,file);
    end
    % Record the path actually selected for this read-only evaluation. The
    % saved MAT file itself is not modified.
    tested{index}.config.model_file = char(file);
    groupKeys(index) = comparable_group_key(tested{index});
    fprintf('  %d. seed %g: %s\n',index,double(cfg.seed),file);
end

[uniqueKeys, groupMembership] = stable_groups(groupKeys);
reports = cell(numel(uniqueKeys), 1);
for groupIndex = 1:numel(uniqueKeys)
    members = find(groupMembership == groupIndex);
    groupResults = [tested{members}];
    seeds = arrayfun(@(result) double(result.config.seed),groupResults);
    options = displayOptions;
    options.preloaded_results = groupResults;
    options.assessment_split = "test";
    if ~isfield(options,'run_recurrent_ablation')
        options.run_recurrent_ablation = true;
    end
    if ~isfield(options,'figure_visibility')
        options.figure_visibility = "on";
    end
    if isfield(options,'output_directory') && numel(uniqueKeys) > 1
        options.output_directory = fullfile(options.output_directory, ...
            sprintf('configuration_%02d',groupIndex));
    end
    fprintf('\nConfiguration group %d/%d: %d model(s), seed(s) %s\n', ...
        groupIndex,numel(uniqueKeys),numel(members),mat2str(seeds));
    reports{groupIndex} = banff_evaluate_task( ...
        'lorenz',seeds,"saved_local",struct(),options);
end

comparison = build_comparison(reports);
fprintf('\nSelected-model comparison\n');
disp(comparison);
if numel(uniqueKeys) > 1
    fprintf(['Note: rows belong to different scientific configurations or ', ...
        'training-source identities and were therefore not pooled.\n']);
end
end

function files = select_model_files()
root = fileparts(fileparts(mfilename('fullpath')));
start = fullfile(root,'outputs','models','*.mat');
[names,folder] = uigetfile(start,'Select completed Lorenz model files', ...
    'MultiSelect','on');
if isequal(names,0)
    files = strings(0,1);
    return;
end
if ischar(names), names = {names}; end
files = string(fullfile(folder,names(:)));
end

function files = normalize_model_files(value, root)
if ischar(value)
    files = string({value});
elseif iscell(value)
    files = string(value(:));
else
    files = string(value(:));
end
files = strip(files);
files(files == "") = [];
for index = 1:numel(files)
    candidate = files(index);
    if exist(candidate,'file') ~= 2
        rooted = string(fullfile(root,char(candidate)));
        if exist(rooted,'file') == 2
            candidate = rooted;
        else
            error('banff:savedLorenzMissing', ...
                'Could not find selected model file: %s',files(index));
        end
    end
    [~,~,extension] = fileparts(candidate);
    if ~strcmpi(extension,'.mat')
        error('banff:savedLorenzExtension', ...
            'Expected a MAT file, received: %s',candidate);
    end
    fileObject = javaObject('java.io.File',char(candidate));
    files(index) = string(char(fileObject.getCanonicalPath()));
end
if numel(unique(files)) ~= numel(files)
    error('banff:savedLorenzDuplicate', ...
        'The selected model-file list contains duplicate paths.');
end
end

function validate_saved_result(file)
variables = whos('-file',char(file));
if ~any(strcmp({variables.name},'result'))
    if any(strcmp({variables.name},'checkpoint'))
        error('banff:savedLorenzCheckpoint', ...
            ['%s is a checkpoint rather than a completed model. Use the ', ...
             'Lorenz checkpoint evaluator for this file.'],file);
    end
    error('banff:savedLorenzVariable', ...
        '%s does not contain a trained result variable.',file);
end
loaded = load(char(file),'result');
required = {'config','best','provenance','complete'};
missing = required(~isfield(loaded.result,required));
if ~isempty(missing)
    error('banff:savedLorenzStructure', ...
        'Saved result %s lacks required fields: %s.',file,strjoin(missing,', '));
end
if ~isscalar(loaded.result.complete) || ~logical(loaded.result.complete)
    error('banff:savedLorenzIncomplete', ...
        'Saved result is not marked complete: %s',file);
end
end

function task = canonical_task(task)
task = lower(replace(replace(string(task),'-','_'),' ','_'));
end

function key = comparable_group_key(result)
cfg = result.config;
if ~isfield(cfg,'scientific_config_sha256')
    error('banff:savedLorenzConfigHash', ...
        'Saved model lacks its scientific-configuration hash.');
end
if isfield(result.provenance,'training_source_sha256')
    source = result.provenance.training_source_sha256;
elseif isfield(result.provenance,'core_source_sha256')
    source = result.provenance.core_source_sha256;
else
    error('banff:savedLorenzSourceHash', ...
        'Saved model lacks training-source provenance.');
end
key = string(cfg.scientific_config_sha256) + "|" + hash_struct(source);
end

function hash = hash_struct(value)
names = sort(fieldnames(value));
text = "";
for index = 1:numel(names)
    text = text + string(names{index}) + "=" + ...
        string(value.(names{index})) + ";";
end
engine = javaMethod('getInstance','java.security.MessageDigest','SHA-256');
engine.update(uint8(unicode2native(char(text),'UTF-8')));
digest = typecast(engine.digest(),'uint8');
hash = string(lower(reshape(dec2hex(digest).',1,[])));
end

function [keys,membership] = stable_groups(values)
keys = strings(0,1);
membership = zeros(numel(values),1);
for index = 1:numel(values)
    group = find(keys == values(index),1);
    if isempty(group)
        keys(end+1,1) = values(index); %#ok<AGROW>
        group = numel(keys);
    end
    membership(index) = group;
end
end

function comparison = build_comparison(reports)
rows = cell(0,1);
for groupIndex = 1:numel(reports)
    report = reports{groupIndex};
    for index = 1:numel(report.results)
        result = report.results(index);
        cfg = result.config;
        row = struct();
        row.ConfigurationGroup = groupIndex;
        row.Seed = double(cfg.seed);
        row.ModelFile = string(cfg.model_file);
        row.HiddenNeurons = double(cfg.N_hidden);
        row.RecurrentRank = double(cfg.N_recurrent);
        row.EncoderGain = double(cfg.encoder_gain);
        row.RecurrentGain = double(cfg.recurrent_gain);
        row.DecoderGain = double(cfg.decoder_gain);
        row.InitialBiasMeanMv = mean(double(cfg.initial_bias(:)));
        row.InitialBiasSdMv = std(double(cfg.initial_bias(:)),0);
        row.TrainedBiasMeanMv = mean(double(result.best.B(:)));
        row.TrainedBiasSdMv = std(double(result.best.B(:)),0);
        row.BestEpoch = double(result.best.epoch);
        row.BestValidationPhaseDistance = double(result.best.metric);
        row.TestPhaseDistance = double(result.test.phase_distance);
        if isempty(report.recurrent_ablation)
            row.ZeroRecurrencePhaseDistance = NaN;
            row.AblationDegradation = NaN;
        else
            ablation = report.recurrent_ablation(index,:);
            row.ZeroRecurrencePhaseDistance = double(ablation.ZeroRecurrence);
            row.AblationDegradation = double(ablation.Degradation);
        end
        rows{end+1,1} = row; %#ok<AGROW>
    end
end
comparison = struct2table(vertcat(rows{:}));
end
