function value = banff_provenance(action, varargin)
%BANFF_PROVENANCE Define source provenance and trained-model compatibility.
%   ALL = BANFF_PROVENANCE("all") hashes the complete readable scientific
%   source set for audit records. TRAINING = BANFF_PROVENANCE("training")
%   hashes only files whose execution can alter optimisation, validation
%   selection, or the resulting trained parameters.
%
%   BANFF_PROVENANCE("assert_training_compatible", SAVED) verifies a saved
%   provenance structure against the current training implementation. Test
%   orchestration, publication export, Live Scripts, and plotting code are
%   intentionally outside this compatibility boundary so they may be improved
%   after a network has been trained.

switch lower(string(action))
    case "all"
        value = source_hashes(all_source_files());
    case "training"
        value = source_hashes(training_source_files());
    case "assert_training_compatible"
        assert_training_compatible(varargin{1});
        value = true;
    otherwise
        error('banff:provenanceAction', 'Unknown provenance action "%s".', action);
end
end

function files = all_source_files()
files = {'banff.m','banff_train.m','banff_test.m','banff_eval.m', ...
    'banff_model.m','banff_data.m','banff_metrics.m','banff_publication.m', ...
    'banff_provenance.m','run_experiment.m'};
end

function files = training_source_files()
% Configuration values are already captured by the scientific configuration
% fingerprint. These files contain the executable training/selection rules.
files = {'banff_train.m','banff_eval.m','banff_model.m','banff_data.m', ...
    'banff_metrics.m'};
end

function assert_training_compatible(saved)
current = source_hashes(training_source_files());
if isfield(saved, 'training_source_sha256')
    compare_hash_sets(saved.training_source_sha256, current);
    return;
end

% Legacy v3 results predate the explicit training-only signature. Preserve
% their original conservative compatibility rule. This deliberately includes
% the shared evaluation/metric sources because they performed validation and
% model selection during training.
if ~isfield(saved, 'core_source_sha256')
    error('banff:modelProvenanceMissing', ...
        'The trained result does not contain source-code provenance.');
end
legacyFields = {'banff_model_m','banff_data_m','banff_eval_m','banff_metrics_m'};
compare_hash_sets(saved.core_source_sha256, current, legacyFields);
end

function compare_hash_sets(saved, current, fields)
if nargin < 3
    fields = fieldnames(current);
end
for index = 1:numel(fields)
    field = fields{index};
    if ~isfield(saved, field) || ~isfield(current, field) || ...
            ~strcmp(saved.(field), current.(field))
        error('banff:modelSourceMismatch', ...
            ['The trained result was produced with a different training ', ...
             'implementation (mismatch in %s). Use the source that trained ', ...
             'the model or retrain it with this release.'], field);
    end
end
end

function hashes = source_hashes(files)
root = fileparts(mfilename('fullpath'));
hashes = struct();
for index = 1:numel(files)
    field = matlab.lang.makeValidName(files{index});
    hashes.(field) = file_sha256(fullfile(root, files{index}));
end
end

function hash = file_sha256(file)
engine = javaMethod('getInstance','java.security.MessageDigest','SHA-256');
fileId = fopen(file,'r');
if fileId < 0
    error('banff:sourceHash','Could not open %s.',file);
end
cleanup = onCleanup(@() fclose(fileId)); %#ok<NASGU>
while true
    bytes = fread(fileId,1024*1024,'*uint8');
    if isempty(bytes), break; end
    engine.update(bytes);
end
digest = typecast(engine.digest(),'uint8');
hash = lower(reshape(dec2hex(digest).',1,[]));
end
