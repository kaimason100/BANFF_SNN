% package_static_result.m
function result = package_static_result(domain, backend, P, best, hist, test, opts)
Pbest = P;
Pbest.B = best.B;
result = struct();
result.domain = char(domain);
result.backend = char(backend);
result.history = hist;
result.best = best;
result.test = test;
result.final_B = P.B;
result.model = Pbest;
result.options = opts;
result.model.recurrent_mode = Pbest.recurrent_mode;
result.model.decoder_mode = Pbest.decoder_mode;
result.training_metadata = primary_bias_training_metadata();
result.training_metadata.mex = mex_runtime_metadata(backend, domain);
result.training = struct('trainable_parameters', 'hidden_bias_only');
result = attach_architecture_metadata(result, Pbest, opts);
end
