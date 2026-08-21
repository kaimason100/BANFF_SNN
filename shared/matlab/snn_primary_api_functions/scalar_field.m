% scalar_field.m
function value = scalar_field(S, name)
value = NaN;
if isstruct(S) && isfield(S, name) && ~isempty(S.(name))
    raw = double(S.(name));
    if ~isempty(raw)
        value = raw(1);
    end
end
end

