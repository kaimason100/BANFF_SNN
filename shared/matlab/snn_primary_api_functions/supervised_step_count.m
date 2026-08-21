% supervised_step_count.m
function n = supervised_step_count(x)
if iscell(x)
    n = 0;
    for ii = 1:numel(x), n = n + max(1, size(x{ii},2)-1); end
elseif isstruct(x) && isfield(x, 'steps')
    n = max(1, x.train_blocks * (x.steps - 1));
else
    n = max(1, size(x,2)-1);
end
end

