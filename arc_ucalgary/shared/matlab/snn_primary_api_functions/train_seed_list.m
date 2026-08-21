% train_seed_list.m
function result = train_seed_list(train_fn, opts)
seeds = double(opts.seed_list(:).');
if isempty(seeds)
    error('snn_primary_api:emptySeedList', 'opts.seed_list must contain at least one seed.');
end
seed_results = [];
seed_model_files = strings(1, numel(seeds));
for ii = 1:numel(seeds)
    one_opts = opts;
    one_opts.seed = seeds(ii);
    one_opts.init_seed = seeds(ii);
    one_opts.split_seed = get_opt(opts, 'split_seed', 42);
    fprintf('[seed %d/%d] init_seed=%d split_seed=%d%s', ...
        ii, numel(seeds), one_opts.init_seed, one_opts.split_seed, newline);
    one_result = train_fn(one_opts);
    one_result.seed_index = ii;
    one_result.init_seed = one_opts.init_seed;
    one_result.split_seed = one_opts.split_seed;
    if isfield(opts, 'model_file') && ~isempty(opts.model_file)
        seed_model_files(ii) = save_seed_training_result(one_result, opts.model_file, seeds(ii));
    end
    if ii == 1
        seed_results = repmat(one_result, 1, numel(seeds));
    else
        seed_results(ii) = one_result;
    end
end
result = struct();
result.backend = get_result_field(seed_results(1), 'backend', 'unknown');
if isfield(seed_results(1), 'domain'), result.domain = seed_results(1).domain; end
result.seed_list = seeds;
result.seed_results = seed_results;
if any(seed_model_files ~= "")
    result.seed_model_files = cellstr(seed_model_files);
end
result.options = opts;
result.options.seed = seeds(1);
result.options.init_seed = seeds(1);
result.options.split_seed = get_opt(opts, 'split_seed', 42);
if isfield(seed_results(1), 'arch')
    result.arch = seed_results(1).arch;
elseif isfield(opts, 'arch')
    result.arch = opts.arch;
end
result.summary = summarize_seed_results(seed_results);
if isfield(seed_results(1), 'test'), result.test = seed_results(1).test; end
if isfield(seed_results(1), 'best'), result.best = seed_results(1).best; end
if isfield(seed_results(1), 'closed_loop_validation')
    result.closed_loop_validation = seed_results(1).closed_loop_validation;
end
end
