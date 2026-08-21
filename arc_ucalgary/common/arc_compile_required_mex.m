function arc_compile_required_mex(repo_root, domain)
%ARC_COMPILE_REQUIRED_MEX Compile the active GPU MEX for an ARC job if needed.
%   This keeps ARC jobs self-contained: if the correct MEX is absent or older
%   than its CUDA source, mexcuda rebuilds it in the active bin/<mexext> folder.

domain = lower(string(domain));
fprintf('[ARC compile] helper version: 20260601_210000%s', newline);
switch domain
    case "classification"
        src_file = fullfile(repo_root, 'Classification', 'src', 'cuda', 'snn_classify_time_loop_gpu_mex.cu');
        output_dir = fullfile(repo_root, 'Classification', 'bin', mexext);
        output_name = 'snn_classify_time_loop_gpu_mex';
    case "regression"
        src_file = fullfile(repo_root, 'Regression', 'src', 'cuda', 'snn_regress_time_loop_gpu_mex.cu');
        output_dir = fullfile(repo_root, 'Regression', 'bin', mexext);
        output_name = 'snn_regress_time_loop_gpu_mex';
    case "dynamical_systems"
        src_file = fullfile(repo_root, 'dynamical_systems', 'src', 'cuda', 'snn_time_loop_gpu_mex.cu');
        output_dir = fullfile(repo_root, 'dynamical_systems', 'bin', mexext);
        output_name = 'snn_time_loop_gpu_mex';
    otherwise
        error('arc_compile_required_mex:domain', 'Unknown ARC MEX domain "%s".', domain);
end

if exist(src_file, 'file') ~= 2
    error('arc_compile_required_mex:missingSource', 'CUDA source not found: %s', src_file);
end
if ~isfolder(output_dir)
    [ok, msg] = mkdir(output_dir);
    if ~ok
        error('arc_compile_required_mex:mkdir', 'Could not create MEX output directory: %s', msg);
    end
end
mex_file = fullfile(output_dir, [output_name '.' mexext]);
needs_compile = exist(mex_file, 'file') ~= 3 && exist(mex_file, 'file') ~= 2;
if ~needs_compile
    src_info = dir(src_file);
    mex_info = dir(mex_file);
    needs_compile = ~isempty(src_info) && ~isempty(mex_info) && src_info.datenum > mex_info.datenum;
end
if ~needs_compile
    addpath(output_dir);
    fprintf('[ARC compile] using current MEX: %s%s', mex_file, newline);
    return;
end

fprintf('[ARC compile] compiling %s from %s%s', output_name, src_file, newline);
args = {'-R2018a', '-O', '-outdir', output_dir, ...
    'NVCCFLAGS="$NVCCFLAGS --allow-unsupported-compiler --fmad=false --prec-div=true --prec-sqrt=true"'};
if any(domain == ["classification", "regression"])
    % Classification and regression kernels use cuBLAS SGEMM for batched
    % readout operations, so ARC must link libcuBLAS explicitly at MEX build
    % time. The dynamical-systems kernel does not call cuBLAS.
    fprintf('[ARC compile] linking cuBLAS for %s%s', output_name, newline);
    if ispc
        args = [args, {'LINKLIBS="$LINKLIBS cublas.lib"'}];
    else
        args = [args, {'-lcublas'}];
    end
end
mexcuda(args{:}, '-output', output_name, src_file);
if exist(mex_file, 'file') ~= 3 && exist(mex_file, 'file') ~= 2
    error('arc_compile_required_mex:missingOutput', 'mexcuda finished but output was not found: %s', mex_file);
end
addpath(output_dir);
clear(output_name);
fprintf('[ARC compile] built MEX: %s%s', mex_file, newline);
end
