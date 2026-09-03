function varargout = banff_eval(action, varargin)
%BANFF_EVAL Shared loss and evaluation code used by training and testing.
%   The network dynamics themselves always run through BANFF_MODEL. This file
%   only computes losses, held-out metrics and closed-loop evaluation protocol.

switch lower(string(action))
    case "static"
        varargout{1} = evaluate_static(varargin{:});
    case "temporal"
        varargout{1} = evaluate_temporal(varargin{:});
    case "loss"
        [varargout{1:nargout}] = supervised_loss(varargin{:});
    case "closed_loop"
        varargout{1} = closed_loop_evaluation(varargin{:});
    otherwise
        error('banff:evaluationAction', 'Unknown evaluation action "%s".', action);
end
end

function evaluation = evaluate_temporal(P,X,Y,cfg,keepOutput)
sampleCount = size(X,3);
lossSum = gpuArray.zeros(1,1,'single');
correct = gpuArray.zeros(1,1,'single');
allOutput = zeros(P.N_output,sampleCount,'single');
spikeCount = gpuArray.zeros(P.N_hidden,1,'single');
for first = 1:cfg.batch_size:sampleCount
    indices = first:min(sampleCount,first+cfg.batch_size-1);
    targets = ensure_gpu(Y(:,indices));
    [output,~,batchSpikeCount] = banff_model('temporal',P, ...
        ensure_gpu(X(:,:,indices)),cfg.sequence_response_steps,false,keepOutput);
    [batchLoss,~,batchCorrect] = supervised_loss(output,targets,cfg.kind);
    lossSum = lossSum + batchLoss;
    correct = correct + batchCorrect;
    if keepOutput
        allOutput(:,indices) = gather(output);
        spikeCount = spikeCount + sum(batchSpikeCount,2);
    end
end
evaluation.loss = single(gather(lossSum)/sampleCount);
evaluation.metric = single(100*gather(correct)/sampleCount);
if keepOutput
    evaluation.output = allOutput;
    duration = single(size(X,2))*P.dt;
    rates = gather(spikeCount)./single(sampleCount)./duration;
    active = rates>0;
    evaluation.neural_activity = struct( ...
        'mean_firing_rate_by_neuron_hz',single(rates(:)), ...
        'active_neuron_mask',active(:),'active_fraction',double(mean(active)), ...
        'active_fraction_percent',100*double(mean(active)), ...
        'calculation',struct('context','full delayed-cue sequence', ...
        'rate_units','Hz','n_test_samples',sampleCount));
end
end

function evaluation = evaluate_static(P, X, Y, cfg, keepOutput)
sampleCount = size(X, 2);
lossSum = gpuArray.zeros(1, 1, 'single');
correct = gpuArray.zeros(1, 1, 'single');
regressionOutput = zeros(P.N_output, sampleCount, 'single');
spikeCount = gpuArray.zeros(P.N_hidden, 1, 'single');
for first = 1:cfg.batch_size:sampleCount
    indices = first:min(sampleCount, first + cfg.batch_size - 1);
    targets = ensure_gpu(Y(:, indices));
    [output, ~, batchSpikeCount] = banff_model( ...
        'static', P, ensure_gpu(X(:, indices)), false, keepOutput);
    [batchLoss, ~, batchCorrect] = supervised_loss(output, targets, cfg.kind);
    lossSum = lossSum + batchLoss;
    correct = correct + batchCorrect;
    if keepOutput || cfg.kind == "regression"
        regressionOutput(:, indices) = gather(output);
    end
    if keepOutput
        spikeCount = spikeCount + sum(batchSpikeCount, 2);
    end
end
evaluation.loss = single(gather(lossSum) / sampleCount);
if cfg.kind == "classification"
    evaluation.metric = single(100 * gather(correct) / sampleCount);
else
    targetValues = Y(:);
    if isa(targetValues, 'gpuArray')
        targetValues = gather(targetValues);
    end
    correlation = corrcoef(double(regressionOutput(:)), double(targetValues));
    if numel(correlation) == 4
        evaluation.metric = single(correlation(1, 2));
    else
        evaluation.metric = single(NaN);
    end
end
if keepOutput
    evaluation.output = regressionOutput;
    duration = single(P.presentationSteps) .* P.dt;
    rates = gather(spikeCount) ./ single(sampleCount) ./ duration;
    active = rates > 0;
    evaluation.neural_activity = struct( ...
        'mean_firing_rate_by_neuron_hz', single(rates(:)), ...
        'active_neuron_mask', active(:), ...
        'active_fraction', double(mean(active)), ...
        'active_fraction_percent', 100 * double(mean(active)), ...
        'calculation', struct('context', 'full held-out test', ...
        'rate_units', 'Hz', 'n_test_samples', sampleCount));
end
end

function value = ensure_gpu(value)
% Keep resident arrays resident while retaining CPU inputs at the public API.
if ~isa(value, 'gpuArray')
    value = gpuArray(value);
end
end

function [loss, gradient, correct] = supervised_loss(output, targets, kind)
if kind == "classification"
    shifted = output - max(output, [], 1);
    probability = exp(shifted) ./ sum(exp(shifted), 1);
    gradient = probability - targets;
    loss = -sum(targets .* log(max(probability, realmin('single'))), 'all');
    [~, predicted] = max(probability, [], 1);
    [~, truth] = max(targets, [], 1);
    correct = sum(predicted == truth, 'all');
else
    gradient = output - targets;
    loss = single(0.5) .* sum(gradient .* gradient, 'all');
    correct = gpuArray.zeros(1, 1, 'single');
end
end

function evaluation = closed_loop_evaluation(P, cfg, dataInformation, role, recordEvents)
initialConditions = banff_data('initial_conditions', cfg, role);
if role == "test"
    validationConditions = banff_data('initial_conditions', cfg, "validation");
    if any(ismember(initialConditions.', validationConditions.', 'rows'))
        error('banff:heldOutInitialConditions', ...
            'A test initial condition is also present in validation.');
    end
end
system = banff_data('system', cfg.task);
if role == "validation"
    duration = cfg.validation_time;
    warmup = cfg.validation_warmup_time;
else
    duration = cfg.test_time;
    warmup = cfg.test_warmup_time;
end
totalDuration = duration + warmup;
transitionCount = round(totalDuration / cfg.dt);
teacherForcing = false(1, transitionCount + 1);
teacherForcing(1) = true;
distances = inf(size(initialConditions, 2), 1);
predictions = cell(size(distances));
truths = cell(size(distances));
eventSets = cell(size(distances));

for index = 1:size(initialConditions, 2)
    referenceRaw = banff_data('trajectory', system, ...
        initialConditions(:, index), totalDuration, cfg);
    reference = single((referenceRaw - dataInformation.mean) ./ dataInformation.std);
    [~, ~, output, events] = banff_model('dynamics', ...
        P, reference, teacherForcing, false, recordEvents);
    output = gather(output);
    warmupSteps = round(warmup / cfg.dt);
    if warmupSteps > 0
        prediction = output(:, warmupSteps + 1:end).';
        truthInitialNormalised = output(:, warmupSteps);
        truthInitialRaw = truthInitialNormalised .* dataInformation.std ...
            + dataInformation.mean;
        truthRaw = banff_data('trajectory', system, ...
            truthInitialRaw, duration, cfg);
        truthNormalised = single((truthRaw - dataInformation.mean) ...
            ./ dataInformation.std);
        truth = truthNormalised(:, 2:end).';
    else
        prediction = output.';
        truth = reference(:, 2:end).';
    end
    if ~isequal(size(prediction), size(truth))
        error('banff:trajectoryShapeMismatch', ...
            ['Closed-loop prediction and reference must have identical ', ...
             'length and state dimension; received [%s] and [%s].'], ...
            num2str(size(prediction)), num2str(size(truth)));
    end
    distances(index) = banff_metrics('phase_distance', ...
        prediction, truth, cfg.phase_metric);
    predictions{index} = prediction;
    truths{index} = truth;
    eventSets{index} = events;
end
evaluation.phase_distance = single(mean(distances));
evaluation.phase_distance_by_initial_condition = single(distances);
evaluation.prediction = predictions;
evaluation.truth = truths;
evaluation.initial_conditions = initialConditions;
if recordEvents
    evaluation.events = eventSets;
end
end
