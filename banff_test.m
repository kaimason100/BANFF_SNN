function result = banff_test(cfg)
if exist(cfg.model_file, 'file') ~= 2
    error('banff:modelMissing', 'Could not find trained result %s.', cfg.model_file);
end
loaded = load(cfg.model_file, 'result');
trained = loaded.result;
cfg = trained.config;
if ~isfield(trained, 'provenance') || ...
        ~isfield(trained.provenance, 'core_source_sha256')
    error('banff:modelProvenanceMissing', ...
        'The trained result does not contain source-code provenance. Retrain with this release.');
end
currentSource = core_source_hashes();
trainedSource = trained.provenance.core_source_sha256;
criticalFields = {'banff_model_m','banff_data_m','banff_eval_m','banff_metrics_m'};
for sourceIndex = 1:numel(criticalFields)
    field = criticalFields{sourceIndex};
    if ~isfield(trainedSource, field) || ...
            ~strcmp(trainedSource.(field), currentSource.(field))
        error('banff:modelSourceMismatch', ...
            ['The trained result was produced with different mathematical source code ', ...
             '(mismatch in %s). Retrain or test with the source that created the model.'], field);
    end
end
if cfg.kind == "dynamics"
    dimension = numel(trained.data_information.mean);
    P = banff_model('gpu', banff_model('create', dimension, dimension, cfg));
    P.B = gpuArray(single(trained.best.B));
    test = banff_eval('closed_loop', P, cfg, trained.data_information, "test", true);
else
    [data, ~] = banff_data('static', cfg, trained.data_information);
    P = banff_model('gpu', banff_model('create', ...
        size(data.X_train, 1), size(data.Y_train, 1), cfg));
    P.B = gpuArray(single(trained.best.B));
    test = banff_eval('static', P, data.X_test, data.Y_test, cfg, true);
    if cfg.kind == "classification"
        test.statistics = banff_metrics('classification', ...
            test.output, data.Y_test);
    else
        test.statistics = banff_metrics('regression', test.output, data.Y_test, ...
            data.target_mean, data.target_std);
    end
end
result = trained;
result.test = test;
result.tested_at = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss Z'));
end

function sourceHashes = core_source_hashes()
root = fileparts(mfilename('fullpath'));
files = {'banff.m','banff_train.m','banff_test.m','banff_eval.m', ...
    'banff_model.m','banff_data.m','banff_metrics.m'};
sourceHashes = struct();
for index = 1:numel(files)
    field = matlab.lang.makeValidName(files{index});
    sourceHashes.(field) = sha256_file(fullfile(root,files{index}));
end
end

function hash = sha256_file(file)
engine = javaMethod('getInstance','java.security.MessageDigest','SHA-256');
fid = fopen(file,'r');
if fid < 0, error('banff:sourceHash','Could not open %s.',file); end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
while true
    bytes = fread(fid,1024*1024,'*uint8');
    if isempty(bytes), break; end
    engine.update(bytes);
end
digest = typecast(engine.digest(),'uint8');
hash = lower(reshape(dec2hex(digest).',1,[]));
end
