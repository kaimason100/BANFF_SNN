% best_bias_from_result.m
function B = best_bias_from_result(result)
if isfield(result, 'best') && isfield(result.best, 'B')
    B = result.best.B;
elseif isfield(result, 'model') && isfield(result.model, 'B')
    B = result.model.B;
elseif isfield(result, 'final_B')
    B = result.final_B;
else
    error('snn_primary_api:modelFile', 'Saved result does not contain a trained bias vector.');
end
end

