function varargout = banff_data(action, varargin)
%BANFF_DATA Dataset splitting, normalisation and target-system simulation.
%   All normalisation is fitted on training data only. Exact duplicate
%   feature-target rows remain in one split to prevent direct leakage.

switch lower(string(action))
    case "static"
        [varargout{1:nargout}] = load_static_data(varargin{:});
    case "dynamics"
        [varargout{1:nargout}] = make_dynamics_pool(varargin{:});
    case "temporal"
        [varargout{1:nargout}] = make_delayed_cue_data(varargin{:});
    case "system"
        varargout{1} = target_system(varargin{:});
    case "trajectory"
        varargout{1} = simulate_target(varargin{:});
    case "initial_conditions"
        varargout{1} = evaluation_initial_conditions(varargin{:});
    otherwise
        error('banff:dataAction', 'Unknown data action "%s".', action);
end
end

%% Delayed cue-response temporal classification
function [data, information] = make_delayed_cue_data(cfg, savedInformation)
if nargin < 2
    savedInformation = struct();
end
if string(cfg.task) ~= "delayed_cue"
    error('banff:temporalTask','Temporal data are defined for delayed_cue only.');
end

totalSteps = double(cfg.sequence_cue_steps + cfg.sequence_delay_steps + ...
    cfg.sequence_response_steps);
data.X_train = delayed_cue_split(cfg.sequence_train_samples, ...
    cfg.split_seed + 101,totalSteps,cfg);
data.Y_train = delayed_cue_targets(cfg.sequence_train_samples,cfg.split_seed + 101);
data.X_validation = delayed_cue_split(cfg.sequence_validation_samples, ...
    cfg.split_seed + 202,totalSteps,cfg);
data.Y_validation = delayed_cue_targets(cfg.sequence_validation_samples,cfg.split_seed + 202);
data.X_test = delayed_cue_split(cfg.sequence_test_samples, ...
    cfg.split_seed + 303,totalSteps,cfg);
data.Y_test = delayed_cue_targets(cfg.sequence_test_samples,cfg.split_seed + 303);

information = struct('generator','paired_balanced_delayed_cue_v2', ...
    'input_channels',{{'binary cue','delay distractor','response signal'}}, ...
    'total_steps',totalSteps,'cue_steps',double(cfg.sequence_cue_steps), ...
    'delay_steps',double(cfg.sequence_delay_steps), ...
    'response_steps',double(cfg.sequence_response_steps), ...
    'distractor_block_steps',double(cfg.sequence_distractor_block_steps), ...
    'distractor_sd',double(cfg.sequence_distractor_sd), ...
    'train_samples',double(cfg.sequence_train_samples), ...
    'validation_samples',double(cfg.sequence_validation_samples), ...
    'test_samples',double(cfg.sequence_test_samples), ...
    'split_seed',double(cfg.split_seed));
if ~isempty(fieldnames(savedInformation))
    required = {'generator','total_steps','cue_steps','delay_steps', ...
        'response_steps','distractor_block_steps','distractor_sd', ...
        'train_samples','validation_samples','test_samples','split_seed'};
    if ~all(isfield(savedInformation,required))
        error('banff:temporalProvenance', ...
            'Saved delayed-cue data information is incomplete.');
    end
    for index = 1:numel(required)
        name = required{index};
        if ~isequal(savedInformation.(name),information.(name))
            error('banff:temporalProvenance', ...
                'Saved delayed-cue generator setting %s does not match.',name);
        end
    end
    information = savedInformation;
end
end

function X = delayed_cue_split(sampleCount, seed, totalSteps, cfg)
[labels,stream,pairIndex] = balanced_labels(sampleCount,seed);
X = zeros(3,totalSteps,sampleCount,'single');
cue = single(2 .* (labels - 1) - 1);
X(1,1:cfg.sequence_cue_steps,:) = repmat(reshape(cue,1,1,[]), ...
    1,double(cfg.sequence_cue_steps),1);

delayFirst = double(cfg.sequence_cue_steps) + 1;
delayLast = delayFirst + double(cfg.sequence_delay_steps) - 1;
block = double(cfg.sequence_distractor_block_steps);
blockCount = ceil(double(cfg.sequence_delay_steps) / block);
distractorByPair = single(cfg.sequence_distractor_sd) .* ...
    single(randn(stream,blockCount,sampleCount/2));
distractor=distractorByPair(:,pairIndex);
for index = 1:blockCount
    first = delayFirst + (index-1)*block;
    last = min(delayLast,first+block-1);
    X(2,first:last,:) = repmat(reshape(distractor(index,:),1,1,[]), ...
        1,last-first+1,1);
end
responseFirst = delayLast + 1;
X(3,responseFirst:totalSteps,:) = single(1);
end

function Y = delayed_cue_targets(sampleCount, seed)
labels = balanced_labels(sampleCount,seed);
Y = zeros(2,sampleCount,'single');
Y(sub2ind(size(Y),labels,1:sampleCount)) = single(1);
end

function [labels,stream,pairIndex] = balanced_labels(sampleCount,seed)
stream = RandStream('mt19937ar','Seed',double(seed));
labels = mod(0:double(sampleCount)-1,2) + 1;
pairIndex=ceil((1:double(sampleCount))/2);
permutation=randperm(stream,double(sampleCount));
labels = labels(permutation);
pairIndex=pairIndex(permutation);
end

%% Static datasets
function [data, information] = load_static_data(cfg, savedInformation)
if nargin < 2
    savedInformation = struct();
end
rng(cfg.split_seed, 'twister');
task = string(cfg.task);
resolvedDatasetFile = dataset_path(cfg.dataset_file);
datasetSha256 = file_sha256(resolvedDatasetFile);
if isempty(datasetSha256)
    error('banff:datasetHash', 'Could not calculate the dataset SHA-256 hash.');
end
if isfield(savedInformation, 'dataset_sha256') && ...
        ~strcmpi(char(savedInformation.dataset_sha256), datasetSha256)
    error('banff:datasetChanged', ...
        'The dataset contents differ from those used to train this model.');
end

duplicateGroupCount = 0;
if task == "mnist" || task == "afro_mnist_vai"
    [rawX, rawLabels, officialTestStart] = load_image_dataset(cfg);
    [trainIndex, validationIndex, testIndex] = image_split( ...
        size(rawX, 1), officialTestStart, savedInformation);
    expectedTest = officialTestStart:size(rawX, 1);
    if ~isequal(sort(testIndex(:)).', expectedTest)
        error('banff:imageSplit', ...
            'Image-task test indices must equal the official test partition.');
    end
    labelIndex = labels_to_indices(rawLabels);
    splitPolicy = 'official_test_random_80_20_train_validation';
elseif task == "breast_cancer"
    matrix = load_numeric_matrix(resolvedDatasetFile);
    rawX = single(matrix(:, 3:end));
    rawLabels = matrix(:, 2);
    labelIndex = labels_to_indices(rawLabels);
    groupId = exact_group_ids_sorted(rawX, labelIndex);
    duplicateGroupCount = sum(accumarray(groupId, 1) > 1);
    [trainIndex, validationIndex, testIndex] = classification_split( ...
        rawX, labelIndex, savedInformation);
    splitPolicy = 'stratified_exact_duplicate_groups_60_20_20';
else
    matrix = load_numeric_matrix(resolvedDatasetFile);
    [rawX, rawLabels] = regression_columns(task, matrix);
    labelIndex = [];
    groupId = exact_group_ids(rawX, rawLabels);
    duplicateGroupCount = sum(accumarray(groupId, 1) > 1);
    [trainIndex, validationIndex, testIndex] = regression_split( ...
        rawX, rawLabels, savedInformation);
    splitPolicy = 'exact_duplicate_groups_60_20_20';
end

validate_split(trainIndex, validationIndex, testIndex, size(rawX, 1));
[featureMean, featureStd] = preprocessing_statistics( ...
    rawX, trainIndex, savedInformation, 'feature');
normalisedX = single((rawX - featureMean) ./ featureStd);

if cfg.kind == "classification"
    classCount = max(labelIndex);
    targets = zeros(numel(labelIndex), classCount, 'single');
    targets(sub2ind(size(targets), (1:numel(labelIndex)).', labelIndex)) = 1;
    targetMean = single(0);
    targetStd = single(1);
else
    [targetMean, targetStd] = preprocessing_statistics( ...
        rawLabels, trainIndex, savedInformation, 'target');
    targets = single((rawLabels - targetMean) ./ targetStd);
end

data = struct();
data.X_train = normalisedX(trainIndex, :).';
data.Y_train = targets(trainIndex, :).';
data.X_validation = normalisedX(validationIndex, :).';
data.Y_validation = targets(validationIndex, :).';
data.X_test = normalisedX(testIndex, :).';
data.Y_test = targets(testIndex, :).';
data.target_mean = single(targetMean);
data.target_std = single(targetStd);

information = struct();
information.train_index = uint32(trainIndex(:));
information.validation_index = uint32(validationIndex(:));
information.test_index = uint32(testIndex(:));
information.feature_mean = single(featureMean);
information.feature_std = single(featureStd);
information.target_mean = single(targetMean);
information.target_std = single(targetStd);
information.split_seed = cfg.split_seed;
information.split_policy = splitPolicy;
information.duplicate_group_count = uint32(duplicateGroupCount);
information.dataset_file = cfg.dataset_file;
information.dataset_sha256 = datasetSha256;
end

function [meanValue, stdValue] = preprocessing_statistics(values, trainIndex, saved, prefix)
% Reuse the exact training-time preprocessing statistics when available.
meanField = [prefix '_mean'];
stdField = [prefix '_std'];
computedMean = mean(values(trainIndex, :), 1);
computedStd = std(values(trainIndex, :), 0, 1);
computedStd(computedStd == 0) = 1;
if isstruct(saved) && isfield(saved, meanField) && isfield(saved, stdField) ...
        && ~isempty(saved.(meanField)) && ~isempty(saved.(stdField))
    meanValue = single(saved.(meanField));
    stdValue = single(saved.(stdField));
    if ~isequal(size(meanValue), size(single(computedMean))) || ...
            ~isequal(size(stdValue), size(single(computedStd)))
        error('banff:preprocessingShape', ...
            'Saved %s preprocessing statistics have the wrong shape.', prefix);
    end
    tolerance = 100 * eps('single') .* max(single(1), max(abs(single(computedMean)), [], 'all'));
    if any(abs(meanValue - single(computedMean)) > tolerance, 'all') || ...
            any(abs(stdValue - single(computedStd)) > tolerance, 'all')
        error('banff:preprocessingMismatch', ...
            'Saved %s preprocessing statistics do not match the verified dataset/split.', prefix);
    end
else
    meanValue = single(computedMean);
    stdValue = single(computedStd);
end
stdValue(stdValue == 0) = 1;
end

function [rawX, labels, testStart] = load_image_dataset(cfg)
loaded = load(dataset_path(cfg.dataset_file), 'training', 'test');
if ~isfield(loaded, 'training') || ~isfield(loaded, 'test')
    error('banff:imageDataset', ...
        'Image data must contain training and test structures.');
end
trainX = flatten_images(loaded.training.images);
testX = flatten_images(loaded.test.images);
trainLabels = loaded.training.labels;
testLabels = loaded.test.labels;
rawX = [trainX; testX];
labels = [trainLabels(:); testLabels(:)];
testStart = size(trainX, 1) + 1;
end

function images = flatten_images(images)
images = single(images);
dimensions = size(images);
if ndims(images) == 4
    images = reshape(images, [], dimensions(4)).';
elseif ndims(images) == 3
    images = reshape(images, [], dimensions(3)).';
elseif ismatrix(images) && size(images, 1) == 784
    images = images.';
elseif ~(ismatrix(images) && size(images, 2) == 784)
    error('banff:imageShape', 'Images must flatten to 784 pixels.');
end
if max(images, [], 'all') > 1
    images = images ./ single(255);
end
if any(~isfinite(images), 'all')
    error('banff:imageValues', 'Image data contain non-finite values.');
end
end

function indices = labels_to_indices(labels)
labels = labels(:);
if isnumeric(labels)
    if any(~isfinite(labels))
        error('banff:labels', 'Labels must be finite.');
    end
    classes = unique(double(labels));
    if isequal(classes(:).', 0:max(classes))
        indices = double(labels) + 1;
    else
        [~, ~, indices] = unique(labels);
    end
else
    [~, ~, indices] = unique(string(labels));
end
indices = double(indices(:));
end

function [trainIndex, validationIndex, testIndex] = ...
        image_split(sampleCount, officialTestStart, saved)
if has_saved_split(saved)
    [trainIndex, validationIndex, testIndex] = saved_split(saved);
    return;
end
trainingPool = 1:(officialTestStart - 1);
trainingPool = trainingPool(randperm(numel(trainingPool)));
trainCount = floor(0.8 * numel(trainingPool));
trainIndex = trainingPool(1:trainCount);
validationIndex = trainingPool(trainCount + 1:end);
testIndex = officialTestStart:sampleCount;
end

function [trainIndex, validationIndex, testIndex] = ...
        classification_split(rawX, labels, saved)
groupId = exact_group_ids_sorted(rawX, labels);
if has_saved_split(saved)
    [trainIndex, validationIndex, testIndex] = saved_split(saved);
    assert_no_group_overlap(groupId, trainIndex, validationIndex, testIndex);
    return;
end
trainIndex = [];
validationIndex = [];
testIndex = [];
for class = 1:max(labels)
    classRows = find(labels == class);
    groups = unique(groupId(classRows));
    groups = groups(randperm(numel(groups)));
    trainTarget = floor(0.6 * numel(classRows));
    validationTarget = floor(0.2 * numel(classRows));
    classTrain = [];
    classValidation = [];
    classTest = [];
    for group = groups(:).'
        members = find(groupId == group).';
        if numel(classTrain) < trainTarget
            classTrain = [classTrain members]; %#ok<AGROW>
        elseif numel(classValidation) < validationTarget
            classValidation = [classValidation members]; %#ok<AGROW>
        else
            classTest = [classTest members]; %#ok<AGROW>
        end
    end
    trainIndex = [trainIndex classTrain]; %#ok<AGROW>
    validationIndex = [validationIndex classValidation]; %#ok<AGROW>
    testIndex = [testIndex classTest]; %#ok<AGROW>
end
trainIndex = trainIndex(randperm(numel(trainIndex)));
validationIndex = validationIndex(randperm(numel(validationIndex)));
testIndex = testIndex(randperm(numel(testIndex)));
assert_no_group_overlap(groupId, trainIndex, validationIndex, testIndex);
end

function [trainIndex, validationIndex, testIndex] = ...
        regression_split(rawX, targets, saved)
groupId = exact_group_ids(rawX, targets);
if has_saved_split(saved)
    [trainIndex, validationIndex, testIndex] = saved_split(saved);
    assert_no_group_overlap(groupId, trainIndex, validationIndex, testIndex);
    return;
end
sampleCount = size(rawX, 1);
targetCounts = [floor(0.6 * sampleCount), floor(0.2 * sampleCount)];
targetCounts(3) = sampleCount - sum(targetCounts);
groupCount = max(groupId);
members = accumarray(groupId, (1:sampleCount).', [groupCount 1], @(x) {x});
groupSizes = cellfun(@numel, members);
order = randperm(groupCount);
counts = zeros(1, 3);
assignment = zeros(groupCount, 1);
for index = 1:groupCount
    group = order(index);
    remaining = targetCounts - counts;
    fits = remaining >= groupSizes(group);
    if any(fits)
        candidates = find(fits);
        [~, selected] = max(remaining(candidates));
        partition = candidates(selected);
    else
        [~, partition] = max(remaining);
    end
    assignment(group) = partition;
    counts(partition) = counts(partition) + groupSizes(group);
end
trainIndex = vertcat(members{assignment == 1}).';
validationIndex = vertcat(members{assignment == 2}).';
testIndex = vertcat(members{assignment == 3}).';
assert_no_group_overlap(groupId, trainIndex, validationIndex, testIndex);
end

function groupId = exact_group_ids(features, targets)
[~, ~, groupId] = unique([double(features) double(targets)], 'rows', 'stable');
end

function groupId = exact_group_ids_sorted(features, targets)
[~, ~, groupId] = unique([double(features) double(targets)], 'rows');
end

function assert_no_group_overlap(groupId, trainIndex, validationIndex, testIndex)
if ~isempty(intersect(groupId(trainIndex), groupId(validationIndex))) || ...
        ~isempty(intersect(groupId(trainIndex), groupId(testIndex))) || ...
        ~isempty(intersect(groupId(validationIndex), groupId(testIndex)))
    error('banff:splitLeakage', ...
        'An exact feature-target duplicate group crosses data splits.');
end
end

function [features, targets] = regression_columns(task, matrix)
switch task
    case "yacht"
        featureColumns = 1:6;
        targetColumn = 7;

    case "toyota"
        targetColumn = 3;
        featureColumns = setdiff(1:size(matrix, 2), targetColumn, 'stable');

    case "abalone"
        if ~isequal(size(matrix), [4177, 9])
            error('banff:abaloneShape', ...
                'Expected Abalone dataset to be 4177x9, but loaded %dx%d.', ...
                size(matrix, 1), size(matrix, 2));
        end

        targetColumn = 9;
        featureColumns = 1:8;

    otherwise
        error('banff:regressionTask', 'Unknown regression task "%s".', task);
end
features = single(matrix(:, featureColumns));
targets = single(matrix(:, targetColumn));
if any(~isfinite(features), 'all') || any(~isfinite(targets), 'all')
    error('banff:datasetValues', 'Dataset contains non-finite values.');
end
end

function matrix = load_numeric_matrix(file)
loaded = load(file);
if isfield(loaded, 'data')
    matrix = loaded.data;
else
    names = fieldnames(loaded);

    numericNames = {};
    for index = 1:numel(names)
        value = loaded.(names{index});
        if isnumeric(value) || istable(value)
            numericNames{end+1} = names{index}; %#ok<AGROW>
        end
    end

    if numel(numericNames) ~= 1
        error('banff:datasetFormat', ...
            'Expected one numeric dataset variable, found: %s', ...
            strjoin(numericNames, ', '));
    end

    matrix = loaded.(numericNames{1});
end
if istable(matrix)
    matrix = table2array(matrix);
end
matrix = single(matrix);
end

function file = dataset_path(name)
root = fileparts(mfilename('fullpath'));
file = fullfile(root, 'data', 'raw', name);

if exist(file, 'file') ~= 2
    error('banff:dataMissing', ...
        'Could not find dataset "%s" at "%s".', name, file);
end
end

function hash = file_sha256(file)
hash = '';
try
    engine = javaMethod('getInstance', ...
        'java.security.MessageDigest', 'SHA-256');
    fileId = fopen(file, 'r');
    if fileId < 0
        return;
    end
    cleanup = onCleanup(@() fclose(fileId));
    while true
        bytes = fread(fileId, 1024 * 1024, '*uint8');
        if isempty(bytes)
            break;
        end
        engine.update(bytes);
    end
    digest = typecast(engine.digest(), 'uint8');
    hash = lower(reshape(dec2hex(digest).', 1, []));
catch
    hash = '';
end
end

function yes = has_saved_split(saved)
yes = isstruct(saved) && all(isfield(saved, ...
    {'train_index','validation_index','test_index'}));
end

function [trainIndex, validationIndex, testIndex] = saved_split(saved)
trainIndex = double(saved.train_index(:)).';
validationIndex = double(saved.validation_index(:)).';
testIndex = double(saved.test_index(:)).';
end

function validate_split(trainIndex, validationIndex, testIndex, sampleCount)
allIndices = [trainIndex(:); validationIndex(:); testIndex(:)];
if isempty(trainIndex) || isempty(validationIndex) || isempty(testIndex) || ...
        numel(allIndices) ~= sampleCount || numel(unique(allIndices)) ~= sampleCount || ...
        any(allIndices < 1) || any(allIndices > sampleCount)
    error('banff:invalidSplit', ...
        'Train, validation and test indices must partition every sample once.');
end
end

%% Dynamical-system data
function [pool, information] = make_dynamics_pool(cfg, savedInformation)
if nargin < 2
    savedInformation = struct();
end
rng(cfg.split_seed, 'twister');
system = target_system(cfg.task);
raw = simulate_target(system, system.initial_state, cfg.long_simulation_time, cfg);
% Column one is the state at t=0. Discarding a duration T therefore starts at
% column round(T/dt)+1, the sample at t=T, rather than one timestep earlier.
burnIndex = max(1, round(cfg.burn_in_time / cfg.dt) + 1);
burnIndex = min(burnIndex, size(raw, 2));
raw = raw(:, burnIndex:end);
if isfield(savedInformation, 'mean') && isfield(savedInformation, 'std')
    stateMean = single(savedInformation.mean);
    stateStd = single(savedInformation.std);
else
    stateMean = mean(raw, 2);
    stateStd = std(raw, 0, 2);
    stateStd(stateStd == 0) = 1;
end
normalised = single((raw - stateMean) ./ stateStd);
windowSamples = round(cfg.training_window / cfg.dt) + 1;
if size(normalised, 2) < windowSamples
    error('banff:dynamicsPool', 'The trajectory pool is shorter than a training window.');
end
pool = struct('states', normalised, 'window_samples', windowSamples, ...
    'max_start', size(normalised, 2) - windowSamples + 1);
information = struct('mean', single(stateMean), 'std', single(stateStd), ...
    'burn_index', burnIndex, 'sample_convention', 'endpoint_inclusive');
end

function system = target_system(task)
switch lower(string(task))
    case "vanderpol"
        system.name = 'vanderpol';
        system.initial_state = single([0.2; 0.3]);
        system.parameters = struct('mu', single(5));
        system.derivative = @(x, p) single([ ...
            p.mu * (x(1) - x(1).^3 / 3 - x(2)); x(1) / p.mu]);
    case "lorenz"
        system.name = 'lorenz';
        system.initial_state = single([0.1; 0.1; 0.1]);
        system.parameters = struct('sigma', single(10), ...
            'rho', single(28), 'beta', single(8/3));
        system.derivative = @(x, p) single([p.sigma * (x(2) - x(1)); ...
            x(1) * (p.rho - x(3)) - x(2); x(1) * x(2) - p.beta * x(3)]);
    case {"sprott", "sprott_s", "sprotts"}
        system.name = 'sprott_s';
        system.initial_state = single([0; 0; 1]);
        system.parameters = struct();
        system.derivative = @(x, ~) single([ ...
            -x(1) - 4*x(2); x(1) + x(3).^2; 1 + x(1)]);
    otherwise
        error('banff:dynamicsTask', 'Unknown dynamical system "%s".', task);
end
end

function states = simulate_target(system, initialState, duration, cfg)
sampleCount = round(duration / cfg.dt) + 1;
states = zeros(numel(initialState), sampleCount, 'single');
states(:, 1) = single(initialState);
for step = 2:sampleCount
    derivative = cfg.system_rate .* system.derivative( ...
        states(:, step - 1), system.parameters);
    states(:, step) = states(:, step - 1) + cfg.dt .* derivative;
end
if any(~isfinite(states), 'all')
    error('banff:dynamicsDiverged', 'Target-system integration became non-finite.');
end
end

function initialConditions = evaluation_initial_conditions(cfg, role)
previousRandomState = rng;
restoreRandomState = onCleanup(@() rng(previousRandomState));
system = target_system(cfg.task);
if lower(string(role)) == "validation"
    seed = cfg.validation_initial_condition_seed;
    count = cfg.validation_initial_conditions;
    includeReference = true;
else
    seed = cfg.test_initial_condition_seed;
    count = cfg.test_initial_conditions;
    includeReference = false;
end
rng(seed, 'twister');
base = system.initial_state(:);
initialConditions = repmat(base, 1, count);
if includeReference
    initialConditions(1, 1) = initialConditions(1, 1) + cfg.initial_condition_jitter;
    firstRandom = 2;
else
    firstRandom = 1;
end
if firstRandom <= count
    % initial_condition_jitter is the maximum absolute perturbation in each
    % coordinate, so draw uniformly from [-jitter,+jitter].
    initialConditions(:, firstRandom:end) = base ...
        + cfg.initial_condition_jitter .* (single(2) .* ...
        rand(numel(base), count - firstRandom + 1, 'single') - single(1));
end
end
