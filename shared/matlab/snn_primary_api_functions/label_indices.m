% label_indices.m
function y = label_indices(labels)
%LABEL_INDICES Map numeric, categorical, or string labels to 1-based class ids.
labels = labels(:);
if isnumeric(labels)
    labels = double(labels);
    assert_finite_classification_labels(labels, 'numeric image labels');
    classes = unique(labels);
    if isequal(classes(:).', 0:max(classes))
        y = labels + 1;
    else
        [~, ~, y] = unique(labels);
    end
else
    if iscategorical(labels) && any(isundefined(labels(:)))
        error('snn_primary_api:nonfiniteClassificationLabels', ...
            'Classification labels contain undefined categorical values.');
    end
    [~, ~, y] = unique(string(labels));
end
y = double(y(:));
end

