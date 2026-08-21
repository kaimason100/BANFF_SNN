% regression_test_stats.m
function stats = regression_test_stats(y_pred, y_true)
%REGRESSION_TEST_STATS Match the non-spiking regression test summaries.
%   The attached reference code reports held-out RMSE, Pearson r and Pearson
%   p. The ARC reference wrappers also print signed error mean and SD, so
%   those are included here for a complete regression test summary.
y_pred = double(y_pred(:));
y_true = double(y_true(:));
valid = isfinite(y_pred) & isfinite(y_true);
y_pred = y_pred(valid);
y_true = y_true(valid);
err = y_pred - y_true;
stats = struct();
stats.n = int32(numel(err));
if isempty(err)
    stats.rmse = single(NaN);
    stats.signed_error_mean = single(NaN);
    stats.signed_error_sd = single(NaN);
    stats.pearson_r = single(NaN);
    stats.pearson_p = single(NaN);
    return;
end
stats.rmse = single(sqrt(mean(err.^2)));
stats.signed_error_mean = single(mean(err));
if numel(err) > 1
    stats.signed_error_sd = single(std(err, 0));
else
    stats.signed_error_sd = single(NaN);
end
[r, p] = pearson_r_p(y_pred, y_true);
stats.pearson_r = single(r);
stats.pearson_p = single(p);
end

