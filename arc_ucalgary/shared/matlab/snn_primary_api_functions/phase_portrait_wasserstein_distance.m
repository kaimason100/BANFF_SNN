% phase_portrait_wasserstein_distance.m
function distance_value = phase_portrait_wasserstein_distance(x_pred, x_true, options)
%PHASE_PORTRAIT_WASSERSTEIN_DISTANCE Geometry metric for closed-loop DS tests.
%   For each pair of state variables, trajectories are treated as 2-D point
%   clouds in phase space. A sliced Wasserstein distance is computed per pair
%   and then averaged, giving a metric of attractor/trajectory geometry.
if any(~isfinite(x_pred(:))) || any(~isfinite(x_true(:)))
    distance_value = Inf;
    return;
end
% Use only the common state dimensions if arrays differ in width.
n_states = min(size(x_pred,2), size(x_true,2));
pairs = phase_portrait_pairs(n_states);
pair_scores = nan(size(pairs,1), 1);
for pp = 1:size(pairs,1)
    % Build subsampled/trimmed 2-D phase portrait clouds for this state pair.
    pred_cloud = phase_portrait_cloud(x_pred(:,1:n_states), pairs(pp,:), options);
    true_cloud = phase_portrait_cloud(x_true(:,1:n_states), pairs(pp,:), options);
    if size(pred_cloud,1) >= 2 && size(true_cloud,1) >= 2
        % Sliced Wasserstein: project each cloud onto many 1-D directions,
        % compare sorted projected samples, then average across projections.
        pair_scores(pp) = sliced_wasserstein_2d(pred_cloud, true_cloud, ...
            options.NumProjections, options.TrimFraction);
    else
        pair_scores(pp) = Inf;
    end
end
finite_scores = pair_scores(isfinite(pair_scores));
if isempty(finite_scores)
    distance_value = Inf;
else
    % Mean over all finite pairwise phase-portrait distances.
    distance_value = mean(finite_scores);
end
end
