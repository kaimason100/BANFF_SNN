% first_finite.m
function value = first_finite(values)
value = NaN;
values = double(values(:));
idx = find(isfinite(values), 1, 'first');
if ~isempty(idx)
    value = values(idx);
end
end

