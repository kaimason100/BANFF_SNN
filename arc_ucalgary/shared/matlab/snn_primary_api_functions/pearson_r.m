% pearson_r.m
function r = pearson_r(a, b)
a = double(a(:)); b = double(b(:));
a = a - mean(a);
b = b - mean(b);
den = sqrt(sum(a.^2)) * sqrt(sum(b.^2));
if ~isfinite(den) || den <= eps
    r = single(NaN);
else
    r = single(sum(a.*b) / den);
end
end

