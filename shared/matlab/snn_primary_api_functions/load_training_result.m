% load_training_result.m
function result = load_training_result(model_file)
if exist(model_file, 'file') ~= 2
    error('snn_primary_api:modelFile', 'Could not find saved model file "%s".', model_file);
end
S = load(model_file);
if isfield(S, 'result')
    result = S.result;
elseif isfield(S, 'train_result')
    result = S.train_result;
else
    names = fieldnames(S);
    if numel(names) == 1 && isstruct(S.(names{1}))
        result = S.(names{1});
    else
        error('snn_primary_api:modelFile', 'Saved model file must contain result or train_result.');
    end
end
end

