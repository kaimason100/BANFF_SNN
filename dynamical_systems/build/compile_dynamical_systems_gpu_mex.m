% Package orientation: Build helper. It locates CUDA/MEX source files, configures compiler flags, and writes compiled binaries into the package bin folder.

% compile_dynamical_systems_gpu_mex.m
% Build the dynamical-systems GPU CUDA MEX into bin/<mexext>.

clc;
use_fast_math = false;
src_name = 'snn_time_loop_gpu_mex.cu';
[src_file, domain_root] = find_cuda_source(src_name, 'dynamical_systems');
output_dir = fullfile(domain_root, 'bin', mexext);
if exist(output_dir, 'dir') ~= 7, mkdir(output_dir); end

fastflag = '';
if use_fast_math, fastflag = ' -use_fast_math'; end
base_args = {'-R2018a', '-O', '-outdir', output_dir, ...
    ['NVCCFLAGS="$NVCCFLAGS --allow-unsupported-compiler --fmad=false --prec-div=true --prec-sqrt=true', fastflag, '"']};

preflight_mexcuda_toolchain();
fprintf('[compile_dynamical_systems_gpu_mex] Output: %s\n', output_dir);
targets = {
    'snn_time_loop_gpu_mex.cu', 'snn_time_loop_gpu_mex'
    };
try
    for ii = 1:size(targets,1)
        [src_i, ~] = find_cuda_source(targets{ii,1}, 'dynamical_systems');
        fprintf('[compile_dynamical_systems_gpu_mex] Source: %s\n', src_i);
        mexcuda(base_args{:}, '-output', targets{ii,2}, src_i);
    end
catch ME
    explain_mexcuda_failure(ME);
    rethrow(ME);
end
addpath(output_dir);
fprintf('[compile_dynamical_systems_gpu_mex] Built in %s\n', output_dir);

function [src_file, domain_root] = find_cuda_source(src_name, domain_name)
starts = {};
this_file = mfilename('fullpath');
if ~isempty(this_file), starts{end+1} = fileparts(this_file); end %#ok<AGROW>
w = which([mfilename '.mlx']);
if ~isempty(w), starts{end+1} = fileparts(w); end %#ok<AGROW>
w = which([mfilename '.m']);
if ~isempty(w), starts{end+1} = fileparts(w); end %#ok<AGROW>
w = which(mfilename);
if ~isempty(w), starts{end+1} = fileparts(w); end %#ok<AGROW>
try
    active_file = matlab.desktop.editor.getActiveFilename;
    if ~isempty(active_file), starts{end+1} = fileparts(active_file); end %#ok<AGROW>
catch
end
starts{end+1} = pwd; %#ok<AGROW>
starts{end+1} = fullfile(pwd, domain_name, 'build'); %#ok<AGROW>
starts{end+1} = fullfile(fileparts(pwd), domain_name, 'build'); %#ok<AGROW>
starts = unique(starts, 'stable');
for ss = 1:numel(starts)
    base = starts{ss};
    for up = 0:12
        root = ascend_dir(base, up);
        if isempty(root), continue; end
        candidates = {fullfile(root, 'src', 'cuda', src_name), fullfile(root, domain_name, 'src', 'cuda', src_name)};
        for cc = 1:numel(candidates)
            if exist(candidates{cc}, 'file') == 2
                src_file = candidates{cc};
                domain_root = fileparts(fileparts(fileparts(src_file)));
                return;
            end
        end
    end
end
fprintf('Searched from these folders:\n');
for ss = 1:numel(starts), fprintf('  %s\n', starts{ss}); end
error('Dynamical-systems GPU CUDA source not found by name: %s. Open this script from the project folder or set MATLAB current folder to the GitHub repo root and rerun.', src_name);
end

function out = ascend_dir(in, n)
out = in;
for ii = 1:n
    parent = fileparts(out);
    if isempty(parent) || strcmp(parent, out), out = ''; return; end
    out = parent;
end
end

function preflight_mexcuda_toolchain()
fprintf('[mexcuda preflight] MATLAB: %s\n', version);
fprintf('[mexcuda preflight] Platform: %s\n', computer);
try
    cpp = mex.getCompilerConfigurations('C++','Selected');
catch ME
    warning('Could not query selected C++ MEX compiler: %s', ME.message);
    cpp = [];
end
if isempty(cpp)
    fprintf('[mexcuda preflight] No selected C++ MEX compiler was reported.\n');
else
    fprintf('[mexcuda preflight] C++ compiler: %s (%s)\n', cpp.Name, cpp.Manufacturer);
end
try
    nvcc_status = system('nvcc --version');
    if nvcc_status ~= 0
        warning('nvcc --version did not run successfully from MATLAB. Check CUDA Toolkit PATH.');
    end
catch ME
    warning('Could not run nvcc --version: %s', ME.message);
end
end

function explain_mexcuda_failure(ME)
msg = string(ME.message);
if contains(msg, 'Supported compiler not detected', 'IgnoreCase', true)
    fprintf('\n[mexcuda help] MATLAB did not find a supported host C++ compiler.\n');
    fprintf('On Linux, install/configure a MATLAB-supported GCC version, then run: mex -setup C++\n');
    fprintf('On Windows, install a supported Visual Studio Build Tools version, then run: mex -setup C++\n\n');
elseif contains(msg, 'nvcc', 'IgnoreCase', true) || contains(msg, 'CUDA', 'IgnoreCase', true)
    fprintf('\n[mexcuda help] CUDA/NVCC was not found or is incompatible with this MATLAB release.\n');
    fprintf('Check NVIDIA driver, CUDA Toolkit, and MATLAB Parallel Computing Toolbox support.\n\n');
end
end
