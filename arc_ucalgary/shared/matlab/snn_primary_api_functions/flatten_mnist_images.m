% flatten_mnist_images.m
function X = flatten_mnist_images(images)
images = single(images);
sz = size(images);
if ndims(images) == 4
    X = reshape(images, [], sz(4)).';
elseif ndims(images) == 3
    X = reshape(images, [], sz(3)).';
elseif ismatrix(images) && size(images,1) == 28*28
    X = images.';
elseif ismatrix(images) && size(images,2) == 28*28
    X = images;
else
    error('Unsupported MNIST-family image array shape %s.', mat2str(sz));
end
if size(X,2) ~= 28*28
    error('MNIST-family images must flatten to 784 pixels, got %d.', size(X,2));
end
if any(~isfinite(X(:)))
    error('snn_primary_api:nonfiniteImagePixels', ...
        'Image dataset contains non-finite pixel values after flattening.');
end
mx = max(X(:));
if isfinite(mx) && mx > 1
    X = X ./ single(255);
end
end

