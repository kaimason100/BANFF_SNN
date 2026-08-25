function result = banff_test(cfg)
if exist(cfg.model_file, 'file') ~= 2
    error('banff:modelMissing', 'Could not find trained result %s.', cfg.model_file);
end
loaded = load(cfg.model_file, 'result');
trained = loaded.result;
cfg = trained.config;
if ~isfield(trained, 'provenance')
    error('banff:modelProvenanceMissing', ...
        'The trained result does not contain source-code provenance. Retrain with this release.');
end
banff_provenance("assert_training_compatible", trained.provenance);
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
% Record the exact test/publication-era source without making it part of the
% trained-model compatibility decision.
result.test_provenance = struct( ...
    'current_source_sha256', banff_provenance("all"), ...
    'training_source_sha256', banff_provenance("training"));
end
