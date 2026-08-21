% summarize_seed_metric_table.m
function S = summarize_seed_metric_table(T)
metric_names = ["Loss", "AccuracyPercent", "RMSE", "PearsonR", "PearsonP", ...
    "SignedErrorMean", "SignedErrorSD", "WassersteinDistance", ...
    "BestValidationLoss", "BestValidationWD", "PrimaryMetric"];
Metric = strings(0,1);
Mean = [];
Std = [];
N = [];
for ii = 1:numel(metric_names)
    name = metric_names(ii);
    if ~any(strcmp(T.Properties.VariableNames, char(name)))
        continue;
    end
    vals = double(T.(char(name)));
    vals = vals(isfinite(vals));
    if isempty(vals)
        continue;
    end
    Metric(end+1,1) = name; %#ok<AGROW>
    Mean(end+1,1) = mean(vals, 'omitnan'); %#ok<AGROW>
    Std(end+1,1) = std(vals, 0, 'omitnan'); %#ok<AGROW>
    N(end+1,1) = numel(vals); %#ok<AGROW>
end
S = table(Metric, Mean, Std, N);
end

