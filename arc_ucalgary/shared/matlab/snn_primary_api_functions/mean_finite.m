% mean_finite.m
function m = mean_finite(x)
x = double(x(:));
x = x(isfinite(x));
if isempty(x)
    m = NaN;
else
    m = mean(x);
end
end

