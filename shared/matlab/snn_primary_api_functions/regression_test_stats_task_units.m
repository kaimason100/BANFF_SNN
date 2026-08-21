% regression_test_stats_task_units.m
function stats = regression_test_stats_task_units(y_pred, y_true, data)
mu_y = 0;
sigma_y = 1;
if isfield(data, 'mu_y') && ~isempty(data.mu_y), mu_y = double(data.mu_y(:)); end
if isfield(data, 'sigma_y') && ~isempty(data.sigma_y), sigma_y = double(data.sigma_y(:)); end
if numel(mu_y) == 1
    mu_y = repmat(mu_y, size(y_true,1), 1);
end
if numel(sigma_y) == 1
    sigma_y = repmat(sigma_y, size(y_true,1), 1);
end
mu_y = reshape(mu_y, [], 1);
sigma_y = reshape(sigma_y, [], 1);
sigma_y(~isfinite(sigma_y) | sigma_y == 0) = 1;
y_pred_task = bsxfun(@plus, bsxfun(@times, double(y_pred), sigma_y), mu_y);
y_true_task = bsxfun(@plus, bsxfun(@times, double(y_true), sigma_y), mu_y);
stats = regression_test_stats(y_pred_task, y_true_task);
end

