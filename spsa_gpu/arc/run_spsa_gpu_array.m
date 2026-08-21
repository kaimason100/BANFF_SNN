% Package orientation: SPSA GPU support. This path is separate from the primary e-prop-style bias-training API and should be reviewed as an alternative optimizer implementation.

% ARC entry point for the standalone GPU SPSA proof-of-principle jobs.
%
% This script intentionally lives inside spsa_gpu and calls only the public
% SPSA GPU wrapper. It does not edit or shadow the existing e-prop back-end.
%
% SLURM_ARRAY_TASK_ID mapping:
%   1 -> breast cancer classification
%   2 -> Yacht hydrodynamics regression
%   3 -> Van der Pol dynamical system

repo_root = spsa_gpu_arc_find_repo_root();
spsa_gpu_arc_add_paths(repo_root);

task_ids = {'bc', 'yacht', 'vanderpol'};
task_labels = {'breast_cancer', 'yacht_hydrodynamics', 'vanderpol'};

array_id_text = getenv('SLURM_ARRAY_TASK_ID');
if isempty(array_id_text)
    array_id_text = '1';
end
array_id = str2double(array_id_text);
if ~isfinite(array_id) || array_id < 1 || array_id > numel(task_ids) || array_id ~= floor(array_id)
    error('run_spsa_gpu_array:BadTaskId', ...
        'SLURM_ARRAY_TASK_ID must be an integer from 1 to %d.', numel(task_ids));
end

task_id = task_ids{array_id};
task_label = task_labels{array_id};
submit_script = fullfile(repo_root, 'spsa_gpu', 'arc', 'submit_spsa_gpu_array.slurm');

fprintf('GPU SPSA ARC job starting for task %d (%s).\n', array_id, task_label);
fprintf('Repository root: %s\n', repo_root);

if exist('compile_spsa_gpu_mex', 'file') == 2
    fprintf('Checking standalone SPSA GPU MEX build products.\n');
    compile_spsa_gpu_mex('Force', false);
else
    warning('run_spsa_gpu_array:NoCompilerHelper', ...
        'compile_spsa_gpu_mex is not on the MATLAB path; assuming MEX files already exist.');
end

run_checks = spsa_gpu_arc_env_bool('SPSA_RUN_CHECKS', true);
save_model = spsa_gpu_arc_env_bool('SPSA_SAVE_MODEL', true);
model_file = getenv('SPSA_MODEL_FILE');
continue_from = getenv('SPSA_CONTINUE_FROM');
additional_epochs = getenv('SPSA_ADDITIONAL_EPOCHS');
epochs_override = getenv('SPSA_EPOCHS');
if xor(isempty(continue_from), isempty(additional_epochs))
    error('run_spsa_gpu_array:ContinuationOptions', ...
        'Set both SPSA_CONTINUE_FROM and SPSA_ADDITIONAL_EPOCHS for continuation jobs.');
end
if ~isempty(continue_from) && ~isempty(epochs_override)
    error('run_spsa_gpu_array:EpochOptions', ...
        'SPSA_EPOCHS is for fresh runs and cannot be combined with SPSA_CONTINUE_FROM.');
end

args = {'EnableCheckpoint', true, ...
        'SubmitScript', submit_script, ...
        'RunChecks', run_checks, ...
        'RunFinalTest', false, ...
        'SaveModel', save_model};
if ~isempty(model_file)
    args = [args, {'ModelFile', model_file}]; %#ok<AGROW>
end
if ~isempty(continue_from)
    args = [args, {'ContinueFrom', continue_from, 'AdditionalEpochs', str2double(additional_epochs)}]; %#ok<AGROW>
end
if ~isempty(epochs_override)
    args = [args, {'Epochs', str2double(epochs_override)}]; %#ok<AGROW>
end

result = spsa_gpu_run_task(task_id, args{:});

if isfield(result, 'checkpoint') && isfield(result.checkpoint, 'needs_resubmit') ...
        && logical(result.checkpoint.needs_resubmit)
    spsa_gpu_arc_resubmit_current_task(result, array_id, submit_script, continue_from, additional_epochs, epochs_override);
    fprintf('Checkpoint saved and checkpoint-resumed job submitted for task %d (%s).\n', ...
        array_id, task_label);
else
    fprintf('GPU SPSA ARC job finished for task %d (%s).\n', array_id, task_label);
end

function repo_root = spsa_gpu_arc_find_repo_root()
    here = fileparts(mfilename('fullpath'));
    candidates = {pwd, here, fileparts(here), fileparts(fileparts(here))};
    for idx = 1:numel(candidates)
        candidate = candidates{idx};
        while ~isempty(candidate)
            has_spsa = exist(fullfile(candidate, 'spsa_gpu', 'matlab', 'spsa_gpu_run_task.m'), 'file') == 2;
            has_shared = exist(fullfile(candidate, 'shared', 'matlab', 'snn_primary_api.m'), 'file') == 2;
            if has_spsa && has_shared
                repo_root = candidate;
                return
            end
            parent = fileparts(candidate);
            if strcmp(parent, candidate)
                break
            end
            candidate = parent;
        end
    end
    error('run_spsa_gpu_array:RepoRootNotFound', ...
        'Could not find a repository root containing spsa_gpu/matlab and shared/matlab.');
end

function spsa_gpu_arc_add_paths(repo_root)
    addpath(fullfile(repo_root, 'spsa_gpu', 'matlab'), '-begin');
    addpath(fullfile(repo_root, 'spsa_gpu', 'build'), '-begin');
    addpath(fullfile(repo_root, 'spsa_gpu', 'bin', mexext), '-begin');
    if exist(fullfile(repo_root, 'spsa_gpu', 'matlab', 'spsa_gpu_add_paths.m'), 'file') == 2
        spsa_gpu_add_paths(repo_root);
    end
end

function value = spsa_gpu_arc_env_bool(name, default_value)
    text = strtrim(lower(getenv(name)));
    if isempty(text)
        value = default_value;
        return
    end
    value = any(strcmp(text, {'1', 'true', 'yes', 'y', 'on'}));
end

function spsa_gpu_arc_resubmit_current_task(result, array_id, submit_script, continue_from, additional_epochs, epochs_override)
    partition = getenv('SLURM_JOB_PARTITION');
    if isempty(partition)
        partition = getenv('SLURM_JOB_PARTITION_LIST');
    end
    if isempty(partition)
        partition = 'gpu-h100,gpu-a100,gpu-l40,gpu-v100';
    end

    export_vars = sprintf('ALL,SPSA_RUN_CHECKS=0,SPSA_MODEL_FILE=%s', ...
        spsa_gpu_arc_shell_quote(result.model_file));
    if ~isempty(continue_from)
        export_vars = sprintf('%s,SPSA_CONTINUE_FROM=%s,SPSA_ADDITIONAL_EPOCHS=%s', ...
            export_vars, spsa_gpu_arc_shell_quote(continue_from), ...
            spsa_gpu_arc_shell_quote(additional_epochs));
    end
    if ~isempty(epochs_override)
        export_vars = sprintf('%s,SPSA_EPOCHS=%s', export_vars, ...
            spsa_gpu_arc_shell_quote(epochs_override));
    end
    command = sprintf('sbatch --array=%d --partition=%s --export=%s', ...
        array_id, spsa_gpu_arc_shell_quote(partition), export_vars);
    command = sprintf('%s %s', command, spsa_gpu_arc_shell_quote(submit_script));
    [status, output] = system(command);
    fprintf('%s\n', output);
    if status ~= 0
        error('run_spsa_gpu_array:ResubmitFailed', ...
            'Checkpoint was saved, but resubmission failed with status %d.', status);
    end
end

function quoted = spsa_gpu_arc_shell_quote(text)
    text = char(text);
    quoted = ['''', strrep(text, '''', '''"''"'''), ''''];
end
