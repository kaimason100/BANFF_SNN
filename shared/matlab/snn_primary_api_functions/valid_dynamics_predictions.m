% valid_dynamics_predictions.m
function Zvalid = valid_dynamics_predictions(Z, num_samples)
%VALID_DYNAMICS_PREDICTIONS Return one-step predictions only.
%   Dynamics inputs contain endpoint-inclusive samples, so a trajectory with
%   N samples has N-1 valid one-step predictions. Older diagnostic paths
%   returned an extra zero-padded final column; this helper removes it and
%   checks that the returned prediction matrix is meaningful.
if nargin < 2 || isempty(num_samples)
    num_samples = size(Z,2);
end
Z = single(Z);
if size(Z,2) < 1
    error('snn_primary_api:emptyDynamicsPrediction', 'Dynamics prediction output is empty.');
end
num_valid = max(1, round(double(num_samples)) - 1);
if size(Z,2) == num_valid
    Zvalid = Z;
elseif size(Z,2) == num_valid + 1
    Zvalid = Z(:,1:end-1);
else
    error('snn_primary_api:dynamicsPredictionShape', ...
        'Dynamics prediction output has %d columns, but %d samples imply %d valid one-step prediction columns.', ...
        size(Z,2), round(double(num_samples)), num_valid);
end
if size(Zvalid,2) > 1 && all(Zvalid(:,end) == 0) && any(Zvalid(:,1:end-1) ~= 0, 'all')
    error('snn_primary_api:zeroPaddedDynamicsPrediction', ...
        'Dynamics prediction output has an unexpected all-zero final valid column.');
end
end

