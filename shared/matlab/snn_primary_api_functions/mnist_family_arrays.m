% mnist_family_arrays.m
function [X_train_pool, y_train_pool, X_test_raw, y_test_raw] = mnist_family_arrays(training, test, label)
if ~isstruct(training) || ~isstruct(test) || ~isfield(training, 'images') || ...
        ~isfield(training, 'labels') || ~isfield(test, 'images') || ~isfield(test, 'labels')
    error('Image dataset "%s" must have training.images, training.labels, test.images, and test.labels.', label);
end
X_train_pool = flatten_mnist_images(training.images);
X_test_raw = flatten_mnist_images(test.images);
y_train_pool = label_indices(training.labels);
y_test_raw = label_indices(test.labels);
if numel(y_train_pool) ~= size(X_train_pool,1) || numel(y_test_raw) ~= size(X_test_raw,1)
    error('Image dataset "%s" has mismatched image and label counts.', label);
end
end

