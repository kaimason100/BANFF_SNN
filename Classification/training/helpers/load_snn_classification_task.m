% Package orientation: Package helper or script. Use the surrounding folder and caller to interpret inputs, outputs, and expected side effects.

function TASK = load_snn_classification_task(task_tag, varargin)
%LOAD_SNN_CLASSIFICATION_TASK Load tabular classification data for SNN scripts.
% Splits are made before feature normalisation. Exact duplicate raw
% feature+label rows are grouped into the same split to avoid direct leakage.

opts = parse_options(varargin{:});
task_tag = normalise_task_tag(task_tag);

[X_all_raw, y_raw, dataset_id, display_name, source_name] = load_raw_task(task_tag, opts.DatasetFile);
X_all_raw = single(X_all_raw);
if any(~isfinite(X_all_raw(:)))
    error('load_snn_classification_task:nonfiniteFeatures', ...
        ['Task "%s" contains non-finite feature values. Clean the dataset before ', ...
         'classification splitting; silent zero-fill is disabled for scientific reliability.'], task_tag);
end
N = size(X_all_raw, 1);
if N < 3
    error('Task "%s" has too few samples for train/validation/test splitting.', task_tag);
end

[y_idx, label_values] = encode_labels(y_raw, opts.LabelValues);
C = numel(label_values);

SPLIT = struct('train', 0.6, 'val', 0.2, 'test', 0.2);
if isempty(opts.SplitIndices)
    [idx_train, idx_val, idx_test] = make_stratified_group_split(X_all_raw, y_idx, SPLIT);
    split_policy = 'stratified_group_disjoint_raw_feature_label_60_20_20';
else
    split_policy = 'repro_saved_indices';
    idx_train = double(opts.SplitIndices.idx_train(:)).';
    idx_val = double(opts.SplitIndices.idx_val(:)).';
    idx_test = double(opts.SplitIndices.idx_test(:)).';
end

check_split_indices(idx_train, idx_val, idx_test, N);
check_no_duplicate_row_leakage(X_all_raw, y_idx, idx_train, idx_val, idx_test);

if isempty(opts.Normalization)
    mu_X = mean(X_all_raw(idx_train, :), 1);
    sigma_X = std(X_all_raw(idx_train, :), 0, 1);
else
    mu_X = single(opts.Normalization.mu_X);
    sigma_X = single(opts.Normalization.sigma_X);
end
sigma_X(sigma_X == 0) = 1;
if numel(mu_X) ~= size(X_all_raw, 2) || numel(sigma_X) ~= size(X_all_raw, 2)
    error('Normalization statistics do not match task "%s" feature dimension.', task_tag);
end

X_all = single((X_all_raw - mu_X) ./ sigma_X);
Y_all = zeros(N, C, 'single');
for ii = 1:N
    Y_all(ii, y_idx(ii)) = 1;
end

TASK = struct();
TASK.task_tag = task_tag;
TASK.display_name = display_name;
TASK.source = source_name;
TASK.dataset_id = dataset_id;
TASK.X_all_raw = X_all_raw;
TASK.y_raw = y_raw;
TASK.y_idx = uint32(y_idx(:));
TASK.label_values = string(label_values(:));
TASK.X_all = X_all;
TASK.Y_all = Y_all;
TASK.X_train = X_all(idx_train, :);
TASK.Y_train = Y_all(idx_train, :);
TASK.X_val = X_all(idx_val, :);
TASK.Y_val = Y_all(idx_val, :);
TASK.X_test = X_all(idx_test, :);
TASK.Y_test = Y_all(idx_test, :);
TASK.idx_train = idx_train(:);
TASK.idx_val = idx_val(:);
TASK.idx_test = idx_test(:);
TASK.SPLIT = SPLIT;
TASK.split_policy = split_policy;
TASK.normalization_fitted_on = 'train';
TASK.mu_X = single(mu_X);
TASK.sigma_X = single(sigma_X);
end

function opts = parse_options(varargin)
opts = struct('DatasetFile', '', 'SplitIndices', [], 'Normalization', [], 'LabelValues', []);
if mod(numel(varargin), 2) ~= 0
    error('Options must be name/value pairs.');
end
for ii = 1:2:numel(varargin)
    name = char(varargin{ii});
    value = varargin{ii+1};
    switch lower(name)
        case 'datasetfile'
            opts.DatasetFile = value;
        case 'splitindices'
            opts.SplitIndices = value;
        case 'normalization'
            opts.Normalization = value;
        case 'labelvalues'
            opts.LabelValues = value;
        otherwise
            error('Unknown option "%s".', name);
    end
end
end

function task_tag = normalise_task_tag(task_tag)
task_tag = lower(strtrim(char(task_tag)));
switch task_tag
    case {'bc', 'breast_cancer', 'breast-cancer', 'iris_bc'}
        task_tag = 'iris_bc';
    otherwise
        error('Unknown classification task "%s".', task_tag);
end
end

function [X_all_raw, y_raw, dataset_id, display_name, source_name] = load_raw_task(task_tag, dataset_file)
switch task_tag
    case 'iris_bc'
        dataset_id = resolve_dataset_file(dataset_file, 'breast_cancer_dataset.mat');
        M = load_first_numeric_matrix(dataset_id);
        if size(M, 2) < 3
            error('Breast-cancer dataset must have columns [*, label, features...].');
        end
        y_raw = M(:, 2);
        X_all_raw = M(:, 3:end);
        display_name = 'Breast cancer';
        source_name = 'local breast_cancer_dataset.mat';
end
end

function dataset_file = resolve_dataset_file(dataset_file, default_name, alias_names)
if nargin < 3 || isempty(alias_names)
    alias_names = {};
end
if ischar(alias_names) || isstring(alias_names)
    alias_names = cellstr(string(alias_names(:)));
end
if nargin < 1 || isempty(dataset_file)
    dataset_file = '';
end
if ~isempty(dataset_file) && exist(dataset_file, 'file') == 2
    dataset_file = char(dataset_file);
    return;
end

root_dir = locate_project_root();
names = [{default_name}, alias_names(:).'];
candidates = {};
candidate_names = {};
for nn = 1:numel(names)
    name = char(names{nn});
    candidates = [candidates; { ...
        fullfile(root_dir, 'data', 'raw', name)
        fullfile(root_dir, 'data', 'external', name)
        fullfile(root_dir, 'data', name)
        fullfile(pwd, name)
        fullfile(pwd, 'data', name)
        fullfile(pwd, 'data', 'raw', name)}]; %#ok<AGROW>
    candidate_names = [candidate_names; repmat({name}, 6, 1)]; %#ok<AGROW>
end
if ~isempty(dataset_file)
    candidates{end+1} = char(dataset_file);
    candidate_names{end+1} = char(dataset_file);
end
for ii = 1:numel(candidates)
    if exist(candidates{ii}, 'file') == 2
        dataset_file = candidates{ii};
        if ~strcmp(candidate_names{ii}, default_name)
            warning('load_snn_classification_task:datasetAliasFallback', ...
                'Using dataset alias "%s" for canonical dataset "%s". Rename the file when possible.', ...
                candidate_names{ii}, default_name);
        end
        return;
    end
end
error('Could not find dataset "%s". Put it in data/raw or pass DatasetFile.', default_name);
end

function root_dir = locate_project_root()
here = fileparts(mfilename('fullpath'));
root_dir = here;
while true
    has_standard_root = exist(fullfile(root_dir, 'setup_project_paths.m'), 'file') == 2 && ...
        exist(fullfile(root_dir, 'data'), 'dir') == 7;
    has_standalone_arc_root = exist(fullfile(root_dir, 'shared', 'matlab', 'snn_primary_api.m'), 'file') == 2 && ...
        exist(fullfile(root_dir, 'data'), 'dir') == 7;
    if has_standard_root || has_standalone_arc_root
        return;
    end
    parent = fileparts(root_dir);
    if strcmp(parent, root_dir) || isempty(parent)
        break;
    end
    root_dir = parent;
end
root_dir = pwd;
end

function M = load_first_numeric_matrix(dataset_file)
S = load(dataset_file);
if isfield(S, 'data')
    M = S.data;
else
    fn = fieldnames(S);
    if isempty(fn)
        error('Dataset "%s" is empty.', dataset_file);
    end
    M = S.(fn{1});
end
if istable(M)
    M = table2array(M);
end
if ~isnumeric(M)
    error('Dataset "%s" must contain a numeric matrix or table.', dataset_file);
end
end

function [y_idx, label_values] = encode_labels(y_raw, saved_label_values)
assert_finite_labels(y_raw);
y_cat = categorical(y_raw);
y_str = string(y_cat);
if isempty(saved_label_values)
    label_values = string(categories(y_cat));
else
    label_values = string(saved_label_values(:));
end
y_idx = zeros(numel(y_str), 1);
for cc = 1:numel(label_values)
    y_idx(y_str == label_values(cc)) = cc;
end
if any(y_idx == 0)
    missing = unique(y_str(y_idx == 0));
    error('Labels not found in saved label set: %s', char(strjoin(missing, ', ')));
end
end

function assert_finite_labels(y_raw)
if isnumeric(y_raw) && any(~isfinite(double(y_raw(:))))
    error('load_snn_classification_task:nonfiniteLabels', ...
        'Classification labels contain NaN, Inf or -Inf values. Labels must be finite.');
end
if iscategorical(y_raw) && any(isundefined(y_raw(:)))
    error('load_snn_classification_task:nonfiniteLabels', ...
        'Classification labels contain undefined categorical values.');
end
end

function [idx_train, idx_val, idx_test] = make_stratified_group_split(X_all_raw, y_idx, SPLIT)
N = size(X_all_raw, 1);
C = max(y_idx);
row_key = [double(X_all_raw), double(y_idx(:))];
[~, ~, group_id] = unique(row_key, 'rows');

idx_train = [];
idx_val = [];
idx_test = [];
for cc = 1:C
    class_idx = find(y_idx == cc);
    class_groups = unique(group_id(class_idx));
    class_groups = class_groups(randperm(numel(class_groups)));
    n_class = numel(class_idx);
    target_train = floor(SPLIT.train * n_class);
    target_val = floor(SPLIT.val * n_class);

    class_train = [];
    class_val = [];
    class_test = [];
    for gg = class_groups(:).'
        members = find(group_id == gg).';
        if numel(class_train) < target_train
            class_train = [class_train, members]; %#ok<AGROW>
        elseif numel(class_val) < target_val
            class_val = [class_val, members]; %#ok<AGROW>
        else
            class_test = [class_test, members]; %#ok<AGROW>
        end
    end
    idx_train = [idx_train, class_train]; %#ok<AGROW>
    idx_val = [idx_val, class_val]; %#ok<AGROW>
    idx_test = [idx_test, class_test]; %#ok<AGROW>
end

idx_train = idx_train(randperm(numel(idx_train)));
idx_val = idx_val(randperm(numel(idx_val)));
idx_test = idx_test(randperm(numel(idx_test)));
if numel(unique([idx_train, idx_val, idx_test])) ~= N
    error('Internal split construction error: split indices do not cover all samples exactly once.');
end
end

function check_split_indices(idx_train, idx_val, idx_test, N)
if isempty(idx_train) || isempty(idx_val) || isempty(idx_test)
    error('Train/validation/test split cannot contain an empty split.');
end
all_idx = [idx_train(:); idx_val(:); idx_test(:)];
if any(all_idx < 1) || any(all_idx > N)
    error('Split indices are out of range.');
end
if numel(unique(all_idx)) ~= numel(all_idx) || numel(all_idx) ~= N
    error('Train/validation/test split indices overlap or do not cover every sample.');
end
end

function check_no_duplicate_row_leakage(X_all_raw, y_idx, idx_train, idx_val, idx_test)
leak_rows = [double(X_all_raw), double(y_idx(:))];
if ~isempty(intersect(leak_rows(idx_train, :), leak_rows(idx_val, :), 'rows')) || ...
        ~isempty(intersect(leak_rows(idx_train, :), leak_rows(idx_test, :), 'rows')) || ...
        ~isempty(intersect(leak_rows(idx_val, :), leak_rows(idx_test, :), 'rows'))
    error('Direct leakage detected: exact duplicate feature+label rows occur across train/validation/test splits.');
end
end
