% load_image_classification_data.m
function data = load_image_classification_data(task_tag, dataset_file, opts)
task_tag = lower(string(task_tag));
if task_tag == "mnist"
    if isempty(dataset_file), dataset_file = 'mnist.mat'; end
    dataset_file = resolve_dataset_path(dataset_file, 'mnist.mat');
elseif task_tag == "afro_mnist_vai" || task_tag == "afro-mnist-vai" || task_tag == "vai"
    if isempty(dataset_file), dataset_file = 'afro_mnist_vai.mat'; end
    dataset_file = resolve_dataset_path(dataset_file, 'afro_mnist_vai.mat');
else
    error('Unknown image classification task "%s".', task_tag);
end
S = load(dataset_file, 'training', 'test');
if ~isfield(S,'training') || ~isfield(S,'test')
    error('Image dataset "%s" must contain training and test structs.', dataset_file);
end
[X_train_pool, y_train_pool, X_test_raw, y_test_raw] = mnist_family_arrays(S.training, S.test, dataset_file);

N_pool = size(X_train_pool,1);
idx_pool = randperm(N_pool);
n_train = max(1, floor(0.8 * N_pool));
idx_train_pool = idx_pool(1:n_train);
idx_val_pool = idx_pool(n_train+1:end);
if isempty(idx_val_pool), idx_val_pool = idx_train_pool(end); end

X_all_raw = single([X_train_pool; X_test_raw]);
y_all = [double(y_train_pool(:)); double(y_test_raw(:))];
C = max(y_all);
Y_all = zeros(numel(y_all), C, 'single');
for ii = 1:numel(y_all), Y_all(ii, y_all(ii)) = 1; end

idx_train = idx_train_pool;
idx_val = idx_val_pool;
idx_test = N_pool + (1:size(X_test_raw,1));
mu_X = mean(X_all_raw(idx_train,:),1);
sigma_X = std(X_all_raw(idx_train,:),0,1);
sigma_X(sigma_X==0) = 1;
% Keep image tasks consistent with every other task: data are standardized
% using training-set statistics here, and the network applies the single
% input-amplitude factor P.INPUT_SCALE = 1/sqrt(N_in) inside make_primary_model.
% Do not apply another 1/sqrt(N_in) factor in the loader.
X_all = single((X_all_raw - mu_X) ./ sigma_X);

data = struct();
data.X_train = single(X_all(idx_train,:).');
data.Y_train = single(Y_all(idx_train,:).');
data.X_val = single(X_all(idx_val,:).');
data.Y_val = single(Y_all(idx_val,:).');
data.X_test = single(X_all(idx_test,:).');
data.Y_test = single(Y_all(idx_test,:).');
data.idx_train = uint32(idx_train(:));
data.idx_val = uint32(idx_val(:));
data.idx_test = uint32(idx_test(:));
data.mu_X = single(mu_X);
data.sigma_X = single(sigma_X);
data.mu_y = single(0);
data.sigma_y = single(1);
data.task = struct('task_tag', char(task_tag), 'dataset_file', dataset_file, ...
    'normalization_fitted_on', 'train', ...
    'input_scaling', 'model_INPUT_SCALE_only');
end
