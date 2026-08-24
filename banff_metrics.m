function value = banff_metrics(metric, varargin)
%BANFF_METRICS Publication evaluation metrics in one auditable file.
switch lower(string(metric))
    case "classification"
        value = classification_metrics(varargin{:});
    case "regression"
        value = regression_metrics(varargin{:});
    case "phase_distance"
        value = phase_distance(varargin{:});
    otherwise
        error('banff:metric', 'Unknown metric "%s".', metric);
end
end

function result = classification_metrics(logits, targets)
[~, predicted] = max(logits, [], 1);
[~, truth] = max(targets, [], 1);
shifted = logits - max(logits, [], 1);
probability = exp(shifted) ./ sum(exp(shifted), 1);
result.accuracy_percent = single(100 * mean(predicted == truth));
result.cross_entropy = single(mean(-sum(targets .* log( ...
    max(probability, realmin('single'))), 1)));
result.predicted_class = uint8(predicted);
result.true_class = uint8(truth);
end

function result = regression_metrics(prediction, truth, targetMean, targetStd)
prediction = double(prediction) .* double(targetStd) + double(targetMean);
truth = double(truth) .* double(targetStd) + double(targetMean);
errorValue = prediction(:) - truth(:);
result.rmse = single(sqrt(mean(errorValue.^2)));
result.signed_error_mean = single(mean(errorValue));
result.signed_error_std = single(std(errorValue));
[correlation, pValue] = corrcoef(prediction(:), truth(:));
if numel(correlation) == 4
    result.pearson_r = single(correlation(1, 2));
    result.pearson_p = single(pValue(1, 2));
else
    result.pearson_r = single(NaN);
    result.pearson_p = single(NaN);
end
result.prediction = single(prediction);
result.truth = single(truth);
end

function distance = phase_distance(prediction, truth, options)
prediction = double(prediction);
truth = double(truth);
validate_phase_options(options);
if any(~isfinite(prediction), 'all') || any(~isfinite(truth), 'all') || ...
        size(prediction, 1) < 2 || size(truth, 1) < 2
    distance = Inf;
    return;
end
dimension = min(size(prediction, 2), size(truth, 2));
if dimension >= 2
    pairs = nchoosek(1:dimension, 2);
else
    pairs = [1 1];
end
scores = inf(size(pairs, 1), 1);
for index = 1:size(pairs, 1)
    predictedCloud = phase_cloud(prediction, pairs(index, :), options);
    trueCloud = phase_cloud(truth, pairs(index, :), options);
    scores(index) = sliced_distance(predictedCloud, trueCloud, ...
        options.projections, options.trim_fraction);
end
finiteScores = scores(isfinite(scores));
if isempty(finiteScores)
    distance = Inf;
else
    distance = mean(finiteScores);
end
end

function validate_phase_options(options)
required = {'projections','trim_fraction','subsample','transient_fraction','max_points'};
missing = required(~isfield(options, required));
if ~isempty(missing)
    error('banff:phaseOptions', 'Missing phase-metric option(s): %s.', strjoin(missing, ', '));
end
if options.projections < 2 || options.projections ~= round(options.projections)
    error('banff:phaseOptions', 'projections must be an integer >= 2.');
end
if options.trim_fraction < 0 || options.trim_fraction >= 0.5 || ...
        2 * floor(options.trim_fraction * options.projections) >= options.projections
    error('banff:phaseOptions', ...
        'trim_fraction must leave at least one projection after trimming each tail.');
end
if options.subsample < 1 || options.subsample ~= round(options.subsample) || ...
        options.max_points < 2 || options.max_points ~= round(options.max_points) || ...
        options.transient_fraction < 0 || options.transient_fraction >= 1
    error('banff:phaseOptions', 'Invalid subsampling, point-cap or transient settings.');
end
end

function cloud = phase_cloud(trajectory, pair, options)
first = 1 + floor(options.transient_fraction * size(trajectory, 1));
first = min(max(first, 1), size(trajectory, 1) - 1);
trajectory = trajectory(first:options.subsample:end, :);
if pair(1) == pair(2)
    cloud = [trajectory(1:end-1, pair(1)), trajectory(2:end, pair(1))];
else
    cloud = trajectory(:, pair);
end
if size(cloud, 1) > options.max_points
    selection = unique(round(linspace(1, size(cloud, 1), options.max_points)));
    cloud = cloud(selection, :);
end
end

function distance = sliced_distance(firstCloud, secondCloud, projections, trimFraction)
if size(firstCloud, 1) < 2 || size(secondCloud, 1) < 2
    distance = Inf;
    return;
end
angles = ((0:(projections - 1)).' + 0.5) .* pi ./ projections;
directions = [cos(angles), sin(angles)];
squaredDistances = zeros(projections, 1);
for index = 1:projections
    first = sort(firstCloud * directions(index, :).');
    second = sort(secondCloud * directions(index, :).');
    if numel(first) ~= numel(second)
        count = max(numel(first), numel(second));
        grid = linspace(0, 1, count).';
        first = interp1(linspace(0, 1, numel(first)).', first, grid);
        second = interp1(linspace(0, 1, numel(second)).', second, grid);
    end
    squaredDistances(index) = mean((first - second).^2);
end
squaredDistances = sort(squaredDistances);
trimCount = floor(trimFraction * projections);
squaredDistances = squaredDistances( ...
    (trimCount + 1):(end - trimCount));
distance = sqrt(mean(squaredDistances));
end
