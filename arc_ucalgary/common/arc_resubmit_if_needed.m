function did_resubmit = arc_resubmit_if_needed(result, repo_root)
%ARC_RESUBMIT_IF_NEEDED Submit the same ARC array item after checkpointing.
%   Returns true when the current MATLAB process should exit without writing
%   a final model. The next SLURM job resumes from the checkpoint file.

did_resubmit = false;
if ~isstruct(result) || ~isfield(result, 'checkpoint') || ...
        ~isfield(result.checkpoint, 'needs_resubmit') || ~result.checkpoint.needs_resubmit
    return;
end

array_id = getenv('SLURM_ARRAY_TASK_ID');
if isempty(array_id) && isfield(result.checkpoint, 'array_id')
    array_id = char(result.checkpoint.array_id);
end
if isempty(array_id)
    fprintf('[ARC checkpoint] not running inside a SLURM array; checkpoint saved but no job was submitted.%s', newline);
    did_resubmit = true;
    return;
end

partition = getenv('SLURM_JOB_PARTITION');
if isempty(partition)
    partition = getenv('SLURM_JOB_PARTITION_LIST');
end
if isempty(partition)
    partition = 'gpu-h100,gpu-a100,gpu-l40,gpu-v100';
end

submit_script = '';
if isfield(result.checkpoint, 'submit_script') && ~isempty(result.checkpoint.submit_script)
    submit_script = char(result.checkpoint.submit_script);
end
if isempty(submit_script)
    submit_script = fullfile(repo_root, 'submit_arc_training_array.slurm');
    if exist(submit_script, 'file') ~= 2
        submit_script = fullfile(repo_root, 'arc_ucalgary', 'submit_arc_training_array.slurm');
    end
end
if exist(submit_script, 'file') ~= 2
    error('arc_resubmit_if_needed:missingSubmitScript', 'Could not find ARC checkpoint-resubmit script: %s', submit_script);
end

% Preserve the partition selected by SLURM, but allow the scheduler to use
% any node in that partition when the checkpoint-resumed job starts.
cmd = sprintf('sbatch --array=%s --partition=%s --export=ALL', ...
    shell_quote(array_id), shell_quote(partition));
cmd = sprintf('%s %s', cmd, shell_quote(submit_script));

fprintf('[ARC checkpoint] submitting checkpoint-resumed job with command:%s%s%s', newline, cmd, newline);
[status, out] = system(cmd);
fprintf('%s', out);
if status ~= 0
    error('arc_resubmit_if_needed:sbatchFailed', 'Checkpoint-resubmit sbatch command failed with status %d.', status);
end
did_resubmit = true;
end

function out = shell_quote(in)
%SHELL_QUOTE Quote a scalar string for POSIX shell commands.
in = char(in);
out = ['''' strrep(in, '''', '''"''"''') ''''];
end
