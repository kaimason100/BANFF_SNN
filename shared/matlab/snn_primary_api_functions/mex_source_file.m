% mex_source_file.m
function source_file = mex_source_file(domain)
root_dir = project_root();
domain = lower(string(domain));
switch domain
    case "classification"
        source_file = fullfile(root_dir, 'Classification', 'src', 'cuda', 'snn_classify_time_loop_gpu_mex.cu');
    case "regression"
        source_file = fullfile(root_dir, 'Regression', 'src', 'cuda', 'snn_regress_time_loop_gpu_mex.cu');
    case {"dynamics","dynamical_systems"}
        source_file = fullfile(root_dir, 'dynamical_systems', 'src', 'cuda', 'snn_time_loop_gpu_mex.cu');
    otherwise
        source_file = '';
end
end

