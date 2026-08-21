% Package orientation: Shared MATLAB utility used across tasks. Read this with the caller open so input/output structs and saved-result fields are clear.

function [model_files, model_file, trained_model_backend] = snn_resolve_seed_model_files(repo_root, model_stem, test_backend, test_seeds, diagnostic_seed_index, trained_model_backend)
%SNN_RESOLVE_SEED_MODEL_FILES Resolve seed model files for CPU/GPU test scripts.
%   CPU and GPU test scripts keep their own evaluation backend. This helper
%   only chooses which trained backend's saved model files are loaded.
%
%   trained_model_backend can be:
%     'auto' : prefer models trained with test_backend, then try the other backend
%     'cpu'  : require CPU-trained saved models
%     'gpu'  : require GPU-trained saved models

if nargin < 6 || isempty(trained_model_backend)
    trained_model_backend = 'auto';
end

repo_root = char(repo_root);
model_stem = char(model_stem);
test_backend = lower(char(test_backend));
requested_backend = lower(char(trained_model_backend));
test_seeds = double(test_seeds(:).');
diagnostic_seed_index = max(1, min(numel(test_seeds), round(double(diagnostic_seed_index))));

switch requested_backend
    case 'auto'
        if strcmp(test_backend, 'cpu')
            candidates = {'cpu', 'gpu'};
        else
            candidates = {'gpu', 'cpu'};
        end
    case {'cpu', 'gpu'}
        candidates = {requested_backend};
    otherwise
        error('snn_resolve_seed_model_files:backend', ...
            'trained_model_backend must be ''auto'', ''cpu'' or ''gpu'', got "%s".', requested_backend);
end

missing_by_backend = struct();
for ii = 1:numel(candidates)
    backend = candidates{ii};
    files = build_seed_files(repo_root, model_stem, backend, test_seeds);
    missing = files(cellfun(@(f) exist(f, 'file') ~= 2, files));
    missing_by_backend.(backend) = missing;
    if isempty(missing)
        model_files = files;
        model_file = model_files{diagnostic_seed_index};
        trained_model_backend = backend;
        return;
    end
end

if strcmp(requested_backend, 'auto')
    detail = sprintf('CPU first missing: %s | GPU first missing: %s', ...
        first_missing(missing_by_backend, 'cpu'), first_missing(missing_by_backend, 'gpu'));
else
    detail = sprintf('%s first missing: %s', upper(requested_backend), ...
        first_missing(missing_by_backend, requested_backend));
end
error('snn_resolve_seed_model_files:missingModels', ...
    ['Saved model(s) not found for requested trained backend "%s". ', ...
     'Run the relevant CPU/GPU training script for all requested seeds, or set trained_model_backend to an available backend. %s'], ...
    requested_backend, detail);
end

function files = build_seed_files(repo_root, model_stem, backend, seeds)
files = arrayfun(@(s) fullfile(repo_root, 'outputs', 'models', ...
    sprintf('%s_%s_primary_seed%03d.mat', model_stem, backend, s)), ...
    seeds, 'UniformOutput', false);
end

function txt = first_missing(missing_by_backend, backend)
if isfield(missing_by_backend, backend) && ~isempty(missing_by_backend.(backend))
    txt = missing_by_backend.(backend){1};
else
    txt = '<none>';
end
end
