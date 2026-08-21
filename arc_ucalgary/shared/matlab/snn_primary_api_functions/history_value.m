% history_value.m
function value = history_value(x, ep)
value = NaN;
if isempty(x) || ep < 1 || ep > numel(x)
    return;
end
xd = double(x(:));
value = double(x(ep));
if ~isfinite(value) || value == 0
    nz = find(isfinite(xd) & xd ~= 0, 1, 'last');
    if ~isempty(nz)
        value = xd(nz);
    end
end
end

