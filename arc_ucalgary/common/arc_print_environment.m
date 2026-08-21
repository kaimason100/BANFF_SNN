function arc_print_environment(task_name)
%ARC_PRINT_ENVIRONMENT Print reproducibility context at ARC job start.

fprintf('[ARC %s] MATLAB: %s%s', task_name, version, newline);
host_name = getenv('HOSTNAME');
if isempty(host_name)
    [status, host_text] = system('hostname');
    if status == 0
        host_name = strtrim(host_text);
    else
        host_name = 'unknown';
    end
end
fprintf('[ARC %s] host: %s%s', task_name, host_name, newline);
fprintf('[ARC %s] pwd: %s%s', task_name, pwd, newline);
fprintf('[ARC %s] SLURM_JOB_ID: %s%s', task_name, getenv('SLURM_JOB_ID'), newline);
fprintf('[ARC %s] SLURM_ARRAY_TASK_ID: %s%s', task_name, getenv('SLURM_ARRAY_TASK_ID'), newline);
try
    g = gpuDevice(1);
    fprintf('[ARC %s] GPU: %s | %.2f GB total | compute capability %s%s', ...
        task_name, g.Name, double(g.TotalMemory)/1024^3, g.ComputeCapability, newline);
catch ME
    fprintf('[ARC %s] GPU query failed before training: %s%s', task_name, ME.message, newline);
end
end
