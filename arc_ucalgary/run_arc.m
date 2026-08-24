% Run one BANFF training job selected by the Slurm array index.
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);

job = max(1, str2double(getenv('SLURM_ARRAY_TASK_ID')));
if ~isfinite(job), job = 1; end
architecture = lower(string(getenv('BANFF_ARCHITECTURE')));
if strlength(architecture) == 0, architecture = "low_rank"; end

options = struct('checkpoint_hours', 23, 'eligibility_mode', "hard_spike");
switch architecture
    case "low_rank"
        tasks = ["breast_cancer" "mnist" "afro_mnist_vai" "abalone" ...
            "toyota" "yacht" "lorenz" "sprott_s" "vanderpol"];
        task = tasks(mod(job - 1, numel(tasks)) + 1);
        options.seed = floor((job - 1) / numel(tasks)) + 1;
    case "full_rank"
        tasks = ["breast_cancer" "yacht" "vanderpol"];
        task = tasks(mod(job - 1, numel(tasks)) + 1);
        options.seed = 1;
        options.recurrent_mode = "full_rank";
        options.N_hidden = 6000;
    case "spsa"
        tasks = ["breast_cancer" "yacht" "vanderpol"];
        task = tasks(mod(job - 1, numel(tasks)) + 1);
        options.seed = 1;
        options.method = "spsa";
    case "neuron_sweep"
        tasks = ["breast_cancer" "yacht" "lorenz"];
        counts = [1000 2000 4000 8000 16000];
        task = tasks(floor((job - 1) / numel(counts)) + 1);
        options.N_hidden = counts(mod(job - 1, numel(counts)) + 1);
        options.seed = 1;
        options.training_profile = "neuron_sweep";
    otherwise
        error('banff:arcArchitecture', ...
            'BANFF_ARCHITECTURE must be low_rank, full_rank, spsa or neuron_sweep.');
end

result = banff("train", task, options);
if ~result.complete
    fprintf('Checkpoint saved; asking Slurm to requeue this array task.\n');
    exit(75);
end
