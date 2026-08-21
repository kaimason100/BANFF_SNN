% validate_static_data.m
function validate_static_data(data, domain, context)
% Validate split orientation and numerical sanity before training/testing.
domain = lower(string(domain));
splits = {'train','val','test'};
for ii = 1:numel(splits)
    split = splits{ii};
    X = data.(sprintf('X_%s', split));
    Y = data.(sprintf('Y_%s', split));
    if isempty(X) || isempty(Y)
        error('snn_primary_api:emptyStaticSplit', '%s has an empty %s split.', context, split);
    end
    if ndims(X) ~= 2 || ndims(Y) ~= 2
        error('snn_primary_api:badStaticShape', '%s %s split must be 2-D.', context, split);
    end
    if size(X,2) ~= size(Y,2)
        error('snn_primary_api:staticSampleMismatch', ...
            '%s %s split has %d input samples but %d target samples.', ...
            context, split, size(X,2), size(Y,2));
    end
    if any(~isfinite(X(:))) || any(~isfinite(Y(:)))
        error('snn_primary_api:nonfiniteStaticData', '%s %s split contains non-finite values.', context, split);
    end
    if domain == "classification"
        class_mass = sum(Y,1);
        if any(abs(class_mass - single(1)) > single(1e-5)) || any(Y(:) < 0)
            error('snn_primary_api:badClassificationTargets', ...
                '%s %s classification targets must be one-hot columns.', context, split);
        end
    elseif domain == "regression" && strcmp(split, 'train')
        if all(abs(Y(:) - Y(1)) <= single(1e-7))
            error('snn_primary_api:degenerateRegressionTargets', ...
                '%s %s regression targets are constant, so correlation/statistics are not meaningful.', context, split);
        end
    end
end
validate_split_indices(data, context);
end

