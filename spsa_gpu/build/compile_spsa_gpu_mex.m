% Package orientation: Build helper. It locates CUDA/MEX source files, configures compiler flags, and writes compiled binaries into the package bin folder.

function compile_spsa_gpu_mex(varargin)
%COMPILE_SPSA_GPU_MEX Build the separate SPSA GPU MEX binaries.
%   The CUDA sources in spsa_gpu/src/cuda are copies of the active release
%   sources compiled under separate MEX names. This avoids overwriting or
%   modifying the current e-prop backend binaries.

cfg = parse_options(varargin{:});
module_root = fileparts(fileparts(mfilename('fullpath')));
repo_root = fileparts(module_root);
output_dir = fullfile(module_root, 'bin', mexext);
if exist(output_dir, 'dir') ~= 7
    mkdir(output_dir);
end

if exist(fullfile(repo_root, 'shared', 'matlab'), 'dir') == 7
    addpath(fullfile(repo_root, 'shared', 'matlab'), '-begin');
    addpath(fullfile(repo_root, 'shared', 'matlab', 'snn_primary_api_functions'), '-end');
end

fastflag = '';
if cfg.use_fast_math
    fastflag = ' -use_fast_math';
end
base_args = {'-R2018a', '-O', '-outdir', output_dir, ...
    ['NVCCFLAGS="$NVCCFLAGS --allow-unsupported-compiler --fmad=false --prec-div=true --prec-sqrt=true', fastflag, '"']};

preflight_mexcuda_toolchain();
targets = {
    fullfile(module_root, 'src', 'cuda', 'spsa_classify_time_loop_gpu_mex.cu'), 'spsa_classify_time_loop_gpu_mex', {'-lcublas'}
    fullfile(module_root, 'src', 'cuda', 'spsa_regress_time_loop_gpu_mex.cu'),  'spsa_regress_time_loop_gpu_mex',  {'-lcublas'}
    fullfile(module_root, 'src', 'cuda', 'spsa_dynamics_time_loop_gpu_mex.cu'), 'spsa_dynamics_time_loop_gpu_mex', {}
    };

for ii = 1:size(targets, 1)
    src = targets{ii, 1};
    out = targets{ii, 2};
    link_args = targets{ii, 3};
    out_file = fullfile(output_dir, [out '.' mexext]);
    if exist(src, 'file') ~= 2
        error('spsa_gpu:compileMissingSource', 'Missing SPSA CUDA source: %s', src);
    end
    if ~cfg.force && ~needs_rebuild(src, out_file)
        fprintf('[compile_spsa_gpu_mex] Up to date, skipping: %s\n', out);
        continue;
    end
    fprintf('[compile_spsa_gpu_mex] Building %s from %s\n', out, src);
    try
        mexcuda(base_args{:}, link_args{:}, '-output', out, src);
    catch ME
        explain_mexcuda_failure(ME);
        rethrow(ME);
    end
end
addpath(output_dir, '-begin');
fprintf('[compile_spsa_gpu_mex] Output path: %s\n', output_dir);
end

function tf = needs_rebuild(src, out_file)
if exist(out_file, 'file') ~= 3
    tf = true;
    return;
end
src_info = dir(src);
out_info = dir(out_file);
tf = isempty(src_info) || isempty(out_info) || src_info.datenum > out_info.datenum;
end

function cfg = parse_options(varargin)
cfg = struct('force', false, 'use_fast_math', false);
if mod(numel(varargin), 2) ~= 0
    error('spsa_gpu:compileOptions', 'Options must be name/value pairs.');
end
for ii = 1:2:numel(varargin)
    key = lower(string(varargin{ii}));
    val = varargin{ii+1};
    switch key
        case "force"
            cfg.force = logical(val);
        case "use_fast_math"
            cfg.use_fast_math = logical(val);
        otherwise
            error('spsa_gpu:compileOptions', 'Unknown option "%s".', key);
    end
end
end

function preflight_mexcuda_toolchain()
fprintf('[spsa_gpu mexcuda preflight] MATLAB: %s\n', version);
fprintf('[spsa_gpu mexcuda preflight] Platform: %s\n', computer);
try
    cpp = mex.getCompilerConfigurations('C++', 'Selected');
catch ME
    warning('Could not query selected C++ MEX compiler: %s', ME.message);
    cpp = [];
end
if isempty(cpp)
    fprintf('[spsa_gpu mexcuda preflight] No selected C++ MEX compiler was reported.\n');
else
    fprintf('[spsa_gpu mexcuda preflight] C++ compiler: %s (%s)\n', cpp.Name, cpp.Manufacturer);
end
try
    status = system('nvcc --version');
    if status ~= 0
        warning('nvcc --version did not run successfully from MATLAB. Check CUDA Toolkit PATH.');
    end
catch ME
    warning('Could not run nvcc --version: %s', ME.message);
end
end

function explain_mexcuda_failure(ME)
msg = string(ME.message);
if contains(msg, 'Supported compiler not detected', 'IgnoreCase', true)
    fprintf('\n[spsa_gpu mexcuda help] MATLAB did not find a supported host C++ compiler.\n');
    fprintf('Configure a MATLAB-supported compiler, then rerun this compile script.\n\n');
elseif contains(msg, 'nvcc', 'IgnoreCase', true) || contains(msg, 'CUDA', 'IgnoreCase', true)
    fprintf('\n[spsa_gpu mexcuda help] CUDA/NVCC was not found or is incompatible with this MATLAB release.\n\n');
end
end
