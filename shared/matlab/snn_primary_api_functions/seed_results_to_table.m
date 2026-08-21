% seed_results_to_table.m
function T = seed_results_to_table(results)
n = numel(results);
Seed = nan(n,1);
ModelFile = strings(n,1);
Loss = nan(n,1);
AccuracyPercent = nan(n,1);
RMSE = nan(n,1);
PearsonR = nan(n,1);
PearsonP = nan(n,1);
SignedErrorMean = nan(n,1);
SignedErrorSD = nan(n,1);
WassersteinDistance = nan(n,1);
BestValidationEpoch = nan(n,1);
BestValidationLoss = nan(n,1);
BestValidationWD = nan(n,1);
Count = nan(n,1);
PrimaryMetric = nan(n,1);
PrimaryMetricName = strings(n,1);
for ii = 1:n
    R = results(ii);
    Seed(ii) = double(get_result_field(R, 'init_seed', get_result_field(R, 'seed_index', ii)));
    ModelFile(ii) = string(get_result_field(R, 'model_file', ""));
    if isfield(R, 'test')
        test = R.test;
        Loss(ii) = scalar_field(test, 'loss');
        Count(ii) = scalar_field(test, 'count');
        if isfield(test, 'wasserstein_distance') || isfield(test, 'wd')
            WassersteinDistance(ii) = first_finite([scalar_field(test, 'wasserstein_distance'), scalar_field(test, 'wd')]);
            PrimaryMetric(ii) = WassersteinDistance(ii);
            PrimaryMetricName(ii) = "Wasserstein distance";
        elseif isfield(test, 'regression')
            stats = test.regression;
            RMSE(ii) = scalar_field(stats, 'rmse');
            PearsonR(ii) = first_finite([scalar_field(stats, 'pearson_r'), scalar_field(stats, 'r')]);
            PearsonP(ii) = first_finite([scalar_field(stats, 'pearson_p'), scalar_field(stats, 'p')]);
            SignedErrorMean(ii) = scalar_field(stats, 'signed_error_mean');
            SignedErrorSD(ii) = scalar_field(stats, 'signed_error_sd');
            Count(ii) = first_finite([Count(ii), scalar_field(stats, 'n')]);
            PrimaryMetric(ii) = RMSE(ii);
            PrimaryMetricName(ii) = "RMSE";
        elseif isfield(R, 'domain') && strcmpi(char(R.domain), 'classification')
            AccuracyPercent(ii) = scalar_field(test, 'metric');
            PrimaryMetric(ii) = AccuracyPercent(ii);
            PrimaryMetricName(ii) = "Accuracy [%]";
        elseif isfield(test, 'metric')
            PrimaryMetric(ii) = scalar_field(test, 'metric');
            PrimaryMetricName(ii) = "Task score";
        end
    end
    if isfield(R, 'best')
        BestValidationEpoch(ii) = scalar_field(R.best, 'epoch');
        BestValidationLoss(ii) = scalar_field(R.best, 'loss');
        BestValidationWD(ii) = scalar_field(R.best, 'wd');
        if ~isfinite(PrimaryMetric(ii))
            if isfinite(BestValidationWD(ii))
                PrimaryMetric(ii) = BestValidationWD(ii);
                PrimaryMetricName(ii) = "Best validation WD";
            elseif isfinite(scalar_field(R.best, 'metric'))
                PrimaryMetric(ii) = scalar_field(R.best, 'metric');
                PrimaryMetricName(ii) = "Best validation metric";
            elseif isfinite(BestValidationLoss(ii))
                PrimaryMetric(ii) = BestValidationLoss(ii);
                PrimaryMetricName(ii) = "Best validation loss";
            end
        end
    end
end
T = table(Seed, ModelFile, Loss, AccuracyPercent, RMSE, PearsonR, PearsonP, ...
    SignedErrorMean, SignedErrorSD, WassersteinDistance, BestValidationEpoch, ...
    BestValidationLoss, BestValidationWD, Count, PrimaryMetricName, PrimaryMetric);
end

