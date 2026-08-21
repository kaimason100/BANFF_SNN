% Package orientation: Shared MATLAB utility used across tasks. Read this with the caller open so input/output structs and saved-result fields are clear.

function out_dir = prepare_run_environment(output_subdir)
%PREPARE_RUN_ENVIRONMENT Set package paths and optionally enter an output dir.
%   OUT_DIR = PREPARE_RUN_ENVIRONMENT() only adds package paths.
%   OUT_DIR = PREPARE_RUN_ENVIRONMENT('repro') changes into outputs/repro.

root = setup_project_paths();
out_dir = root;

if nargin < 1 || isempty(output_subdir)
    return;
end

out_dir = fullfile(root, 'outputs', output_subdir);
if exist(out_dir, 'dir') ~= 7
    mkdir(out_dir);
end
cd(out_dir);
end
