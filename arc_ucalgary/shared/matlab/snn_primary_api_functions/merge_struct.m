% merge_struct.m
function opts = merge_struct(base, override)
opts = base;
if isempty(override), return; end
names = fieldnames(override);
for i = 1:numel(names)
    key = names{i};
    if isstruct(override.(key)) && isfield(opts, key) && isstruct(opts.(key))
        opts.(key) = merge_struct(opts.(key), override.(key));
    else
        opts.(key) = override.(key);
    end
end
end

