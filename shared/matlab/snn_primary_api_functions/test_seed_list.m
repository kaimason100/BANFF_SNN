% test_seed_list.m
function result = test_seed_list(test_fn, train_result, opts)
seed_results = train_result.seed_results;
out = [];
for ii = 1:numel(seed_results)
    one_result = test_fn(seed_results(ii), opts);
    one_result.seed_index = ii;
    one_result.init_seed = get_result_field(seed_results(ii), 'init_seed', get_result_field(seed_results(ii).options, 'init_seed', get_result_field(seed_results(ii).options, 'seed', NaN)));
    if ii == 1
        out = repmat(one_result, 1, numel(seed_results));
    else
        out(ii) = one_result;
    end
end
result = struct();
if isfield(out(1), 'domain'), result.domain = out(1).domain; end
result.train_backend = get_result_field(out(1), 'train_backend', 'unknown');
result.test_backend = get_result_field(out(1), 'test_backend', 'unknown');
result.model_file = char(opts.model_file);
result.seed_results = out;
result.seed_list = get_result_field(train_result, 'seed_list', 1:numel(out));
if isfield(out(1), 'arch')
    result.arch = out(1).arch;
elseif isfield(train_result, 'arch')
    result.arch = train_result.arch;
end
result.summary = summarize_seed_results(out);
result.seed_table = result.summary.seed_table;
[result.bias_by_seed, result.network_seed_audit] = audit_seed_networks(seed_results, result.seed_list);
if isfield(out(1), 'test'), result.test = out(1).test; end
end
