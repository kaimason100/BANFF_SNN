% has_valid_regression_stats.m
function tf = has_valid_regression_stats(summary)
tf = isfield(summary, 'regression') && isfield(summary.regression, 'rmse') && ...
    isfinite(double(summary.regression.rmse));
end

