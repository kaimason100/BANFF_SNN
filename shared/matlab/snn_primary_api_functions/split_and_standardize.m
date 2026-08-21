% split_and_standardize.m
function data = split_and_standardize(X_raw, Y_raw, opts)
N = size(X_raw,1);
duplicate_group_count = count_exact_duplicate_groups(X_raw, Y_raw);
group_duplicates = logical(get_opt(opts, 'group_exact_duplicate_rows', false)) && ...
    duplicate_group_count > 0;
if isfield(opts, 'idx_train') && isfield(opts, 'idx_val') && isfield(opts, 'idx_test')
    idx_train = double(opts.idx_train(:)).';
    idx_val = double(opts.idx_val(:)).';
    idx_test = double(opts.idx_test(:)).';
else
    rng(get_opt(opts, 'split_seed', get_opt(opts, 'seed', 42)), 'twister');
    n_train = max(1, floor(0.6*N));
    n_val = max(1, floor(0.2*N));
    if n_train + n_val >= N
        n_train = max(1, N-2);
        n_val = 1;
    end
    if group_duplicates
        [idx_train, idx_val, idx_test] = grouped_random_split( ...
            X_raw, Y_raw, [n_train n_val N-n_train-n_val]);
    else
        idx = randperm(N);
        idx_train = idx(1:n_train);
        idx_val = idx(n_train+1:n_train+n_val);
        idx_test = idx(n_train+n_val+1:end);
    end
end
mu_X = mean(X_raw(idx_train,:),1);
sigma_X = std(X_raw(idx_train,:),0,1);
sigma_X(sigma_X==0) = 1;
X = single((X_raw - mu_X) ./ sigma_X);
if size(Y_raw,2) == 1
    mu_y = mean(Y_raw(idx_train,:),1);
    sigma_y = std(Y_raw(idx_train,:),0,1);
    if sigma_y == 0, sigma_y = 1; end
    Y = single((Y_raw - mu_y) ./ sigma_y);
else
    mu_y = single(0); sigma_y = single(1); Y = single(Y_raw);
end
data = struct();
data.X_train = single(X(idx_train,:).');
data.Y_train = single(Y(idx_train,:).');
data.X_val = single(X(idx_val,:).');
data.Y_val = single(Y(idx_val,:).');
data.X_test = single(X(idx_test,:).');
data.Y_test = single(Y(idx_test,:).');
data.idx_train = uint32(idx_train(:));
data.idx_val = uint32(idx_val(:));
data.idx_test = uint32(idx_test(:));
if group_duplicates
    data.split_policy = 'grouped_exact_feature_target_rows';
    assert_no_exact_group_overlap(X_raw, Y_raw, idx_train, idx_val, idx_test);
elseif logical(get_opt(opts, 'group_exact_duplicate_rows', false))
    data.split_policy = 'independent_rows_no_exact_duplicate_groups';
else
    data.split_policy = 'independent_rows';
end
data.exact_duplicate_group_count = double(duplicate_group_count);
data.mu_X = single(mu_X);
data.sigma_X = single(sigma_X);
data.mu_y = single(mu_y);
data.sigma_y = single(sigma_y);
end

function count = count_exact_duplicate_groups(X, Y)
[~, ~, group_id] = unique([X Y], 'rows', 'stable');
group_sizes = accumarray(group_id, 1);
count = nnz(group_sizes > 1);
end

function [idx_train, idx_val, idx_test] = grouped_random_split(X, Y, targets)
% Keep every exact feature-plus-target row in one partition while remaining
% as close as possible to the requested 60/20/20 row counts.
rows = [X Y];
[~, ~, group_id] = unique(rows, 'rows', 'stable');
n_groups = max(group_id);
members = accumarray(group_id, (1:size(rows,1)).', [n_groups 1], @(v) {v});
group_sizes = cellfun(@numel, members);
order = randperm(n_groups);
counts = zeros(1, 3);
assignment = zeros(n_groups, 1);

for kk = 1:n_groups
    group = order(kk);
    group_size = group_sizes(group);
    remaining = targets - counts;
    fits = remaining >= group_size;
    if any(fits)
        candidates = find(fits);
        [~, pick] = max(remaining(candidates));
        partition = candidates(pick);
    else
        [~, partition] = max(remaining);
    end
    assignment(group) = partition;
    counts(partition) = counts(partition) + group_size;
end

idx_train = vertcat(members{assignment == 1}).';
idx_val = vertcat(members{assignment == 2}).';
idx_test = vertcat(members{assignment == 3}).';
if isempty(idx_train) || isempty(idx_val) || isempty(idx_test)
    error('snn_primary_api:emptyGroupedSplit', ...
        'Grouped duplicate-aware splitting produced an empty partition.');
end
end

function assert_no_exact_group_overlap(X, Y, idx_train, idx_val, idx_test)
rows = [X Y];
[~, ~, group_id] = unique(rows, 'rows', 'stable');
train_groups = unique(group_id(idx_train));
val_groups = unique(group_id(idx_val));
test_groups = unique(group_id(idx_test));
if ~isempty(intersect(train_groups, val_groups)) || ...
        ~isempty(intersect(train_groups, test_groups)) || ...
        ~isempty(intersect(val_groups, test_groups))
    error('snn_primary_api:duplicateGroupLeakage', ...
        'An exact feature-plus-target duplicate group crosses data partitions.');
end
end

function value = get_opt(S, name, default_value)
if isstruct(S) && isfield(S, name) && ~isempty(S.(name))
    value = S.(name);
else
    value = default_value;
end
end
