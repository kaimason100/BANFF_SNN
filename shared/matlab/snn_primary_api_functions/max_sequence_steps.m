% max_sequence_steps.m
function n = max_sequence_steps(x)
if iscell(x)
    n = 0;
    for ii = 1:numel(x), n = max(n, size(x{ii},2)); end
elseif isstruct(x) && isfield(x, 'steps')
    n = x.steps;
else
    n = size(x,2);
end
n = max(2, n);
end

