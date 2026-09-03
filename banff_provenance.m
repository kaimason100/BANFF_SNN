function value = banff_provenance(action, varargin)
%BANFF_PROVENANCE Define source provenance and trained-model compatibility.
%   ALL = BANFF_PROVENANCE("all") hashes the complete readable scientific
%   source set for audit records. TRAINING = BANFF_PROVENANCE("training")
%   hashes only files whose execution can alter optimisation, validation
%   selection, or the resulting trained parameters.
%
%   BANFF_PROVENANCE("assert_training_compatible", SAVED, CONFIG) verifies a
%   saved provenance structure against the current training implementation.
%   CONFIG permits narrowly audited, configuration-dependent compatibility
%   transitions; omission retains strict source equality. Test orchestration,
%   publication export, Live Scripts, and plotting code are intentionally
%   outside this boundary so they may be improved after network training.
%   Source hashes use canonical LF line endings so that an unchanged Git tree
%   has the same identity on Windows and Linux.

switch lower(string(action))
    case "all"
        value = source_hashes(all_source_files());
    case "training"
        value = source_hashes(training_source_files());
    case "assert_training_compatible"
        assert_training_compatible(varargin{:});
        value = true;
    case "hash_file"
        % Exposed for the cross-platform provenance regression test.
        value = file_sha256(varargin{1});
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

function assert_training_compatible(saved, config)
if nargin < 2
    config = struct();
end
current = source_hashes(training_source_files());
if isfield(saved, 'training_source_sha256')
    if compatible_additive_training_extensions( ...
            saved.training_source_sha256, current, config)
        return;
    end
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

function compatible = compatible_additive_training_extensions(saved,current,config)
% Two audited additive transitions leave all previously registered tasks
% unchanged: support for an explicit neuron-wise initial-bias vector, and the
% addition of a separate delayed-cue temporal task. Whole-file hashes change
% because the new dispatch branches live in shared files. Accept only the
% exact predecessor/current signatures below, never a delayed-cue result, and
% require a scalar bias for the predecessor that supported scalars only.
old = struct( ...
    'banff_train_m','fe2c766b0cc005406d4d033f2df3e0f292cab95700f977db9d9235cc616ec627', ...
    'banff_eval_m','0faf252c04cd0536760d72424795349cfd7fbd056884d16f4a1ca5b2c43c9111', ...
    'banff_data_m','e3a742283397c5ceb2cff061abe73f01039cb9ab9b1b2f65ba87fe3deffdc44a', ...
    'banff_metrics_m','c3f34d81f3e84bd07f423896a27ee4b98c9296e110461e85b10babd5dd9b47c4');
oldScalarModel = ...
    'a98b1dcc3b1fcd2b35e65329fbabdb777e4fd0b1d37e5c7e1613c1a2cf509d84';
oldVectorModel = ...
    '670af95eca96e45aff643536b0273342824373b5a41ccd72917455c8deb8a0af';
expectedCurrent = struct( ...
    'banff_train_m','511fe48092d61a0d038f92f5f4d768492680a493597f11534c73f1e091d9c682', ...
    'banff_eval_m','39e2ad11af46c92535eba41fd346287a6b896e12d5afd3c295dfdf996c06ab1c', ...
    'banff_model_m','cf20ccc3e358a54ccac7fcd4ae976f766dfd29bf1eb2dc3cd8a90601f701a02f', ...
    'banff_data_m','d92e55251e1d4bfbf8d26cddf11e4229ea1956b6d3690a9e2d79a8d134959adc', ...
    'banff_metrics_m','c3f34d81f3e84bd07f423896a27ee4b98c9296e110461e85b10babd5dd9b47c4');

compatible = isstruct(config) && isfield(config,'task') && ...
    string(config.task)~="delayed_cue" && ...
    hash_fields_equal(current,expectedCurrent) && ...
    hash_fields_equal(saved,old) && isfield(saved,'banff_model_m');
if ~compatible, return; end
if strcmp(saved.banff_model_m,oldScalarModel)
    compatible=isfield(config,'initial_bias') && isscalar(config.initial_bias);
else
    compatible=strcmp(saved.banff_model_m,oldVectorModel);
end
end

function equal=hash_fields_equal(first,second)
fields=fieldnames(second);
equal=all(isfield(first,fields));
for index=1:numel(fields)
    field=fields{index};
    if ~isfield(first,field) || ~strcmp(first.(field),second.(field))
        equal=false;
        return;
    end
end
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
bytes = fread(fileId,Inf,'*uint8');
if ~isempty(bytes)
    % Git stores the publication sources with LF endings, whereas a Windows
    % checkout may contain CRLF or even mixed endings. Hash the canonical text
    % representation: remove CR from CRLF and map any lone CR to LF. Reading
    % the complete (small) source file also handles a CRLF pair at what would
    % otherwise be a streaming-block boundary.
    remove = false(size(bytes));
    remove(1:end-1) = bytes(1:end-1) == 13 & bytes(2:end) == 10;
    bytes(remove) = [];
    bytes(bytes == 13) = 10;
    engine.update(bytes);
end
digest = typecast(engine.digest(),'uint8');
hash = lower(reshape(dec2hex(digest).',1,[]));
end
