% assert_finite_classification_labels.m
function assert_finite_classification_labels(labels, context)
if isnumeric(labels) && any(~isfinite(double(labels(:))))
    error('snn_primary_api:nonfiniteClassificationLabels', ...
        '%s contain NaN, Inf or -Inf values. Classification labels must be finite.', context);
end
end

