% rmfield_if_present.m
function s = rmfield_if_present(s, key)
if isstruct(s) && isfield(s, key)
    s = rmfield(s, key);
end
end

