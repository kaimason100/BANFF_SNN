% get_result_field.m
function value = get_result_field(s, key, fallback)
if isstruct(s) && isfield(s, key)
    value = s.(key);
else
    value = fallback;
end
end

