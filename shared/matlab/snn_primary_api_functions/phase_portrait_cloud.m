% phase_portrait_cloud.m
function cloud = phase_portrait_cloud(x, pair, options)
n_rows = size(x,1);
if n_rows < 2
    cloud = zeros(0,2);
    return;
end
first_row = 1 + floor(max(0, min(0.9, options.TransientFraction)) * n_rows);
first_row = min(max(first_row, 1), n_rows - 1);
x = x(first_row:end, :);
x = x(1:max(1, round(options.Subsample)):end, :);
if pair(1) == pair(2)
    cloud = [x(1:end-1,pair(1)), x(2:end,pair(1))];
else
    cloud = x(:,pair);
end
cloud = cloud(all(isfinite(cloud), 2), :);
if size(cloud,1) > options.MaxPoints
    idx = unique(round(linspace(1, size(cloud,1), options.MaxPoints)));
    cloud = cloud(idx, :);
end
end

