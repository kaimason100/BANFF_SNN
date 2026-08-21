% compare_arrays.m
function cmp = compare_arrays(a, b)
size_a = size(a);
size_b = size(b);
n_a = numel(a);
n_b = numel(b);
if ~isequal(size_a, size_b)
    error('snn_primary_api:compareShapeMismatch', ...
        'Cannot compare arrays with different shapes: A size %s (%d elements), B size %s (%d elements).', ...
        mat2str(size_a), n_a, mat2str(size_b), n_b);
end
a = double(a(:));
b = double(b(:));
d = a - b;
finite_mask = isfinite(a) & isfinite(b) & isfinite(d);
cmp = struct();
cmp.n = numel(a);
cmp.has_nan = any(isnan(a) | isnan(b));
cmp.has_inf = any(isinf(a) | isinf(b));
cmp.n_finite = sum(finite_mask);
if cmp.n_finite == 0
    cmp.max_abs = inf;
    cmp.rms = inf;
    cmp.max_rel = inf;
    return;
end
d_finite = d(finite_mask);
a_finite = a(finite_mask);
cmp.max_abs = max(abs(d_finite));
cmp.rms = sqrt(mean(d_finite.^2));
cmp.max_rel = max(abs(d_finite) ./ max(1, abs(a_finite)));
end

