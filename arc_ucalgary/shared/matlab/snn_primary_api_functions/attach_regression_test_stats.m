% attach_regression_test_stats.m
function summary = attach_regression_test_stats(summary, data, split)
%ATTACH_REGRESSION_TEST_STATS Compute regression summaries in task units.
%   Training uses standardized regression targets. Reporting converts both
%   predictions and targets back with the training-set target mean/SD so RMSE
%   and signed errors are in the original task units.
if ~isfield(summary, 'Z') || isempty(summary.Z)
    return;
end
if ~isfield(summary, 'Y') || isempty(summary.Y)
    [~, summary.Y] = split_arrays(data, split);
end
summary.regression = regression_test_stats_task_units(summary.Z, summary.Y, data);
summary.regression.source = 'reported_task_units';
end

