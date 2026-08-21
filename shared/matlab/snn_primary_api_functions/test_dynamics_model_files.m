% test_dynamics_model_files.m
function result = test_dynamics_model_files(backend, opts)
files = normalize_model_file_list(opts.model_files);
seeds = normalize_seed_list(get_opt(opts, 'seed_list', 1:numel(files)), numel(files));
out = [];
for ii = 1:numel(files)
    if exist(files{ii}, 'file') ~= 2
        error('snn_primary_api:modelFileMissing', 'Saved dynamics model for seed %g was not found: %s', seeds(ii), files{ii});
    end
    one_opts = rmfield_if_present(opts, 'model_files');
    one_opts = rmfield_if_present(one_opts, 'seed_list');
    one_opts.save_publication_analysis = false;
    one_opts.model_file = files{ii};
    one_result = test_dynamics(backend, one_opts);
    one_result.seed_index = ii;
    one_result.init_seed = seeds(ii);
    one_result.model_file = files{ii};
    if ii == 1
        out = repmat(one_result, 1, numel(files));
    else
        out(ii) = one_result;
    end
end
result = package_model_file_seed_tests(out, opts, files, seeds);
result.domain = 'dynamical_systems';
if logical(get_opt(opts, 'save_publication_analysis', true))
    save_publication_test_analysis(result, 'dynamical_systems', opts);
end
end
