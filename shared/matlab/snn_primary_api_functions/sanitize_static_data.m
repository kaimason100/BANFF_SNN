% sanitize_static_data.m
function data = sanitize_static_data(data, domain, opts)
%SANITIZE_STATIC_DATA Apply an explicit non-finite data policy.
policy = get_nonfinite_policy(opts);
splits = {'train','val','test'};
for ii = 1:numel(splits)
    split = splits{ii};
    x_key = sprintf('X_%s', split);
    y_key = sprintf('Y_%s', split);
    X = single(data.(x_key));
    Y = single(data.(y_key));
    bad_x = ~isfinite(X);
    if any(bad_x(:))
        if policy == "zero_fill"
            warning('snn_primary_api:nonfiniteStaticFeaturesZeroFill', ...
                'Explicit nan_policy=zero_fill: replacing %d non-finite %s feature values with zero.', nnz(bad_x), split);
            X(bad_x) = single(0);
        else
            error('snn_primary_api:nonfiniteStaticFeatures', ...
                ['%s features contain %d non-finite values. Clean the data or set ', ...
                 'opts.nonfinite_policy explicitly; supported fill policy is ''zero_fill''. ', ...
                 'The legacy opts.nan_policy name is still accepted.'], split, nnz(bad_x));
        end
    end
    bad_y = ~isfinite(Y);
    if any(bad_y(:))
        error('snn_primary_api:nonfiniteStaticTargets', ...
            '%s %s targets contain %d non-finite values. Targets are never imputed automatically.', ...
            split, char(domain), nnz(bad_y));
    end
    data.(x_key) = X;
    data.(y_key) = Y;
end
end

