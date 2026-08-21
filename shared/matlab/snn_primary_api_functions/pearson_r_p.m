% pearson_r_p.m
function [r, p] = pearson_r_p(a, b)
valid = isfinite(a) & isfinite(b);
a = double(a(valid));
b = double(b(valid));
if numel(a) < 3 || std(a) == 0 || std(b) == 0
    r = NaN;
    p = NaN;
    return;
end
try
    [R, P] = corrcoef(a, b);
    r = R(1,2);
    p = P(1,2);
catch
    a0 = a - mean(a);
    b0 = b - mean(b);
    r = sum(a0 .* b0) / (sqrt(sum(a0.^2)) * sqrt(sum(b0.^2)));
    p = NaN;
end
end

