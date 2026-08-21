% Package orientation: Package helper or script. Use the surrounding folder and caller to interpret inputs, outputs, and expected side effects.

function result = run_architecture_sanity_tests(varargin)
%RUN_ARCHITECTURE_SANITY_TESTS Small architecture checks for local/CI use.
%   CPU tests always run. CPU/GPU numerical equivalence checks run only when
%   the relevant current CUDA MEX files are compiled and on the MATLAB path.
%   run_architecture_sanity_tests("RequireGPU", true) fails if GPU checks
%   cannot run.

repo_root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo_root, 'shared', 'matlab'), '-begin');
add_project_paths(repo_root);

opts = parse_options(varargin{:});
result = struct();
result.cpu_architecture = snn_primary_api('check_architecture_modes', 'classification', 'cpu', struct());
result.gpu_equivalence = run_gpu_equivalence_checks();
result.summary = summarise_architecture_sanity(result);
if opts.require_gpu && result.summary.gpu.skipped > 0
    error('run_architecture_sanity_tests:gpuSkipped', ...
        'RequireGPU is true, but %d GPU architecture checks were skipped.', result.summary.gpu.skipped);
end
end

function checks = run_gpu_equivalence_checks()
modes = sanity_architecture_modes();
checks = repmat(struct('name', "", 'classification', struct(), 'regression', struct(), ...
    'dynamics', struct()), numel(modes), 1);
for ii = 1:numel(modes)
    opts = struct();
    opts.arch = modes(ii).arch;
    opts.N_hidden = 32;
    opts.N_rec = 4;
    opts.epochs = 1;
    opts.batch_size = 4;
    opts.check_tolerance = struct('abs', 1e-3, 'rel', 1e-3);
    checks(ii).name = modes(ii).name;
    checks(ii).classification = run_one_gpu_check('check_static', 'classification', opts, 'snn_classify_time_loop_gpu_mex');
    checks(ii).regression = run_one_gpu_check('check_static', 'regression', opts, 'snn_regress_time_loop_gpu_mex');
    checks(ii).dynamics = run_one_gpu_check('check_dynamics', 'dynamical_systems', opts, 'snn_time_loop_gpu_mex');
end
end

function out = run_one_gpu_check(action, domain, opts, mex_name)
out = struct('status', "skipped", 'reason', "", 'result', []);
if exist(mex_name, 'file') ~= 3
    out.reason = string(mex_name) + " is not compiled or not on path";
    return;
end
try
    out.result = snn_primary_api(action, domain, 'gpu', opts);
    out.status = "ran";
catch ME
    out.status = "failed";
    out.reason = string(ME.message);
end
end

function opts = parse_options(varargin)
opts = struct('require_gpu', false);
if mod(numel(varargin), 2) ~= 0
    error('run_architecture_sanity_tests:options', 'Options must be name/value pairs.');
end
for ii = 1:2:numel(varargin)
    name = lower(string(varargin{ii}));
    value = varargin{ii+1};
    switch name
        case {"requiregpu", "require_gpu"}
            opts.require_gpu = logical(value);
        otherwise
            error('run_architecture_sanity_tests:options', 'Unknown option "%s".', char(name));
    end
end
end

function summary = summarise_architecture_sanity(result)
gpu = result.gpu_equivalence;
statuses = strings(0, 1);
for ii = 1:numel(gpu)
    statuses(end+1,1) = string(gpu(ii).classification.status); %#ok<AGROW>
    statuses(end+1,1) = string(gpu(ii).regression.status); %#ok<AGROW>
    statuses(end+1,1) = string(gpu(ii).dynamics.status); %#ok<AGROW>
end
summary = struct();
summary.cpu = struct('passed', double(result.cpu_architecture.all_cpu_ran), ...
    'failed', double(~result.cpu_architecture.all_cpu_ran), 'skipped', 0);
summary.gpu = struct('passed', sum(statuses == "ran"), ...
    'failed', sum(statuses == "failed"), 'skipped', sum(statuses == "skipped"));
summary.gpu_mex_available = struct( ...
    'classification', exist('snn_classify_time_loop_gpu_mex', 'file') == 3, ...
    'regression', exist('snn_regress_time_loop_gpu_mex', 'file') == 3, ...
    'dynamical_systems', exist('snn_time_loop_gpu_mex', 'file') == 3);
summary.gpu_architecture_modes = string({gpu.name}).';
fprintf('[architecture sanity] CPU passed=%d failed=%d skipped=%d%s', ...
    summary.cpu.passed, summary.cpu.failed, summary.cpu.skipped, newline);
fprintf('[architecture sanity] GPU passed=%d failed=%d skipped=%d%s', ...
    summary.gpu.passed, summary.gpu.failed, summary.gpu.skipped, newline);
disp('[architecture sanity] GPU MEX availability:');
disp(summary.gpu_mex_available);
disp('[architecture sanity] GPU architecture modes requested:');
disp(summary.gpu_architecture_modes);
end

function modes = sanity_architecture_modes()
base = default_arch_options();
modes = repmat(struct('name', "", 'arch', base), 9, 1);
modes(1).name = "low_rank_shared";
modes(1).arch.recurrent_mode = "low_rank"; modes(1).arch.decoder_mode = "shared";
modes(2).name = "low_rank_signed_gaussian";
modes(2).arch = modes(1).arch; modes(2).arch.decoder_mode = "signed"; modes(2).arch.signed_decoder_distribution = "gaussian";
modes(3).name = "low_rank_signed_uniform";
modes(3).arch = modes(2).arch; modes(3).arch.signed_decoder_distribution = "uniform";
modes(4).name = "full_rank_shared_dense";
modes(4).arch = base; modes(4).arch.recurrent_mode = "full_rank"; modes(4).arch.decoder_mode = "shared"; modes(4).arch.full_rank_storage = "dense";
modes(5).name = "full_rank_signed_gaussian_dense";
modes(5).arch = modes(4).arch; modes(5).arch.decoder_mode = "signed"; modes(5).arch.signed_decoder_distribution = "gaussian";
modes(6).name = "full_rank_signed_uniform_dense";
modes(6).arch = modes(5).arch; modes(6).arch.signed_decoder_distribution = "uniform";
modes(7).name = "full_rank_shared_sparse";
modes(7).arch = modes(4).arch; modes(7).arch.full_rank_storage = "sparse"; modes(7).arch.full_rank_p_rec = single(0.25);
modes(8).name = "full_rank_signed_gaussian_sparse";
modes(8).arch = modes(7).arch; modes(8).arch.decoder_mode = "signed"; modes(8).arch.signed_decoder_distribution = "gaussian";
modes(9).name = "full_rank_signed_uniform_sparse";
modes(9).arch = modes(8).arch; modes(9).arch.signed_decoder_distribution = "uniform";
end
