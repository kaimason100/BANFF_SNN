% sliced_wasserstein_2d.m
function d = sliced_wasserstein_2d(P, Q, num_projections, trim_fraction)
if size(P,1) < 2 || size(Q,1) < 2
    d = Inf;
    return;
end
theta = ((0:(num_projections-1)).' + 0.5) * pi / num_projections;
directions = [cos(theta), sin(theta)];
vals = nan(num_projections, 1);
for kk = 1:num_projections
    p = sort(P * directions(kk,:).');
    q = sort(Q * directions(kk,:).');
    if numel(p) ~= numel(q)
        n_interp = max(numel(p), numel(q));
        tt = linspace(0, 1, n_interp).';
        p = interp1(linspace(0, 1, numel(p)).', p, tt, 'linear', 'extrap');
        q = interp1(linspace(0, 1, numel(q)).', q, tt, 'linear', 'extrap');
    end
    vals(kk) = mean_finite((p - q).^2);
end
vals = sort(vals(isfinite(vals)));
if isempty(vals)
    d = Inf;
    return;
end
n_trim = floor(max(0, min(0.45, trim_fraction)) * numel(vals));
if 2*n_trim < numel(vals)
    vals = vals((1+n_trim):(end-n_trim));
end
d = sqrt(mean_finite(vals));
end

