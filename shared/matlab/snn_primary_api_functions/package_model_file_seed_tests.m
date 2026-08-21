% package_model_file_seed_tests.m
function result = package_model_file_seed_tests(out, opts, files, seeds)
result = struct();
if isfield(out(1), 'domain'), result.domain = out(1).domain; end
result.train_backend = get_result_field(out(1), 'train_backend', 'unknown');
result.test_backend = get_result_field(out(1), 'test_backend', 'unknown');
result.model_file = files{1};
result.model_files = files;
result.seed_list = seeds;
result.seed_results = out;
result.options = opts;
result.summary = summarize_seed_results(out);
result.seed_table = result.summary.seed_table;
[result.bias_by_seed, result.network_seed_audit] = audit_seed_networks(files, seeds, files);
if isfield(out(1), 'test'), result.test = out(1).test; end
end
