% make_lambda_sequence_for_data.m
function lambda = make_lambda_sequence_for_data(x, opts)
if iscell(x)
    lambda = cellfun(@(xx) make_lambda_sequence(size(xx,2), opts), x, 'UniformOutput', false);
elseif isstruct(x) && isfield(x, 'steps')
    lambda = make_lambda_sequence(x.steps, opts);
else
    lambda = make_lambda_sequence(size(x,2), opts);
end
end

