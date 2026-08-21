% split_arrays.m
function [X, Y] = split_arrays(data, split)
switch lower(char(split))
    case 'train'
        X = data.X_train; Y = data.Y_train;
    case {'val','validation'}
        X = data.X_val; Y = data.Y_val;
    case 'test'
        X = data.X_test; Y = data.Y_test;
    otherwise
        error('snn_primary_api:split', 'Unknown split "%s".', split);
end
end

