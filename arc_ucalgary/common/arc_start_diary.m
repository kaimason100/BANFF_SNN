function diary_file = arc_start_diary(repo_root, task_name)
%ARC_START_DIARY Start a timestamped text log for an ARC training run.
%   The log is written under outputs/arc_logs so SLURM stdout and MATLAB
%   diary output can both be inspected after the job finishes.

logs_dir = fullfile(repo_root, 'outputs', 'arc_logs');
if exist(logs_dir, 'dir') ~= 7
    mkdir(logs_dir);
end
stamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
job_id = getenv('SLURM_JOB_ID');
array_id = getenv('SLURM_ARRAY_TASK_ID');
if isempty(job_id), job_id = 'local'; end
if isempty(array_id), array_id = 'single'; end
safe_task = matlab.lang.makeValidName(char(task_name));
diary_file = fullfile(logs_dir, sprintf('%s_job%s_task%s_%s.log', safe_task, job_id, array_id, stamp));
diary(diary_file);
cleanup_obj = onCleanup(@() diary('off')); %#ok<NASGU>
assignin('base', ['arcDiaryCleanup_' safe_task], cleanup_obj);
fprintf('[ARC %s] diary: %s%s', safe_task, diary_file, newline);
end
