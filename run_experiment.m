function results = run_experiment(action, task, profile, seeds, overrides)
%RUN_EXPERIMENT One readable launcher for every BANFF experiment family.
%   RESULTS = RUN_EXPERIMENT(ACTION,TASK,PROFILE,SEEDS,OVERRIDES)
%   replaces the former collection of train/test Live Scripts.
%
%   PROFILE is one of:
%       "main"         principal low-rank e-prop experiment
%       "full_rank"    sparse 6k full-rank control
%       "spsa"         low-rank SPSA control
%       "neuron_sweep" low-rank size sweep; provide overrides.N_hidden
%
%   Examples:
%       run_experiment("train","lorenz","main",1:3)
%       R = run_experiment("test","lorenz","main",1:3);
%       banff_publication(R);  % explicit publication export
%       run_experiment("train","yacht","full_rank",1)
%       run_experiment("train","breast_cancer","spsa",1)
%       run_experiment("train","vanderpol","neuron_sweep",1,struct('N_hidden',8000))
%       run_experiment("train","delayed_cue","main",1:3)
%       evaluate_delayed_cue_models(1:3)
%
%   Each seed is dispatched as an independent complete experiment. This wrapper
%   performs no learning or simulation itself; it standardises profile overrides
%   and preserves the public BANFF configuration/provenance path.

if nargin < 3 || isempty(profile), profile = "main"; end
if nargin < 4 || isempty(seeds)
    if lower(string(profile)) == "main", seeds = 1:3; else, seeds = 1; end
end
if nargin < 5, overrides = struct(); end

action = lower(string(action));
profile = lower(string(profile));
options = overrides;
switch profile
    case "main"
        options.method = "eprop";
        options.recurrent_mode = "low_rank";
        options.training_profile = "main";
    case "full_rank"
        options.method = "eprop";
        options.recurrent_mode = "full_rank";
        if ~isfield(options, 'N_hidden'), options.N_hidden = 6000; end
    case "spsa"
        options.method = "spsa";
        options.recurrent_mode = "low_rank";
    case "neuron_sweep"
        options.method = "eprop";
        options.recurrent_mode = "low_rank";
        options.training_profile = "neuron_sweep";
        if ~isfield(options, 'N_hidden')
            error('banff:runExperimentSize', ...
                'neuron_sweep requires overrides.N_hidden.');
        end
    otherwise
        error('banff:runExperimentProfile', ...
            'profile must be main, full_rank, spsa or neuron_sweep.');
end

results = struct([]);
for index = 1:numel(seeds)
    oneOptions = options;
    oneOptions.seed = seeds(index);
    try
        oneResult = banff(action, task, oneOptions);
    catch exception
        % MATLAB can otherwise report only this launcher line for failures
        % raised by GPU kernels or functions called beneath BANFF.
        fprintf(2, '%s\n', getReport(exception, 'extended', ...
            'hyperlinks', 'off'));
        rethrow(exception);
    end
    if index == 1
        % A fieldless struct cannot receive a structure with fields by
        % subscripted assignment in MATLAB. Initialise from the first run.
        results = oneResult;
    else
        results(index) = oneResult; %#ok<AGROW>
    end
end
end
