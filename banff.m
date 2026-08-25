function result = banff(action, task, overrides)
%BANFF Public entry point for one BANFF experiment.
%   CFG = BANFF("config", TASK, OPTIONS) resolves all scientific settings.
%   RESULT = BANFF("train", TASK, OPTIONS) trains the task.
%   RESULT = BANFF("test",  TASK, OPTIONS) evaluates the validation-selected model.
%
%   Training and testing both use BANFF_MODEL as the single forward simulator.
%   BANFF_TRAIN and BANFF_TEST contain the short orchestration code; BANFF_EVAL
%   contains losses and held-out evaluation only.

if nargin < 3
    overrides = struct();
end
action = lower(string(action));
cfg = experiment_config(task, overrides);

if action == "config"
    result = cfg;
    return;
end
if ~canUseGPU
    error('banff:noGPU', 'A supported NVIDIA GPU is required for training/testing.');
end

switch action
    case "train"
        result = banff_train(cfg);
    case "test"
        result = banff_test(cfg);
    otherwise
        error('banff:action', 'Action must be "config", "train" or "test".');
end
end

%% Scientific configuration
function cfg = experiment_config(task, overrides)
task = canonical_task(task);
cfg = struct();
cfg.task = task;
cfg.seed = 1;
cfg.split_seed = 42;
cfg.training_seed = 4242; % fixed across network seeds for controlled comparisons
cfg.N_hidden = 32000;
cfg.N_recurrent = 10;
cfg.recurrent_mode = "low_rank";
cfg.method = "eprop";
cfg.training_profile = "main";
cfg.full_rank_probability = single(0.10);

cfg.dt = single(1e-3);
cfg.tau_membrane = single(50e-3);
cfg.tau_adaptation = single(500e-3);
cfg.tau_synapse_rise = single(2e-3);
cfg.tau_synapse_decay = single(50e-3);
cfg.resting_voltage = single(-70);
cfg.threshold_voltage = single(-50);
cfg.reset_voltage = single(-65);
cfg.adaptation_jump = single(0.5);
cfg.initial_bias = []; % resolved after overrides; defaults to Vth-EL

% Eligibility mode:
%   "surrogate"  = triangular pseudo-derivative on every timestep
%   "hard_spike" = principal method: event-time local sensitivity is
%                  injected only at real spikes.
cfg.eligibility_mode = "hard_spike";
cfg.hard_event_gain = single(1);       % 1/mV; explicit hard-event gain

% Optional continuous-surrogate ablation. These values are ignored in
% hard_spike mode and must be reported whenever surrogate mode is used.
cfg.surrogate_peak = single(0.7);      % 1/mV
cfg.surrogate_half_width = single(10); % mV
cfg.encoder_gain = single(2);
cfg.recurrent_gain = single(0.05);
cfg.decoder_gain = single(0.1);
cfg.excitatory_fraction = single(0.5);

% Every task uses conventional bias-corrected AMSGrad: the running maximum
% is taken over the raw second-moment accumulator, and the current Adam bias
% correction is applied to that maximum in the parameter update.
cfg.optimizer = "amsgrad";
cfg.learning_rate_start = single(5e-2);
cfg.learning_rate_end = single(1e-3);
cfg.adam_beta1 = single(0.9);
cfg.adam_beta2 = single(0.999);
cfg.adam_epsilon = single(1e-8);
cfg.spsa_c_start = single(0.75);
cfg.spsa_c_end = single(0.05);
cfg.spsa_schedule_epochs = [];
cfg.learning_rate_schedule_epochs = [];

cfg.presentation_time = single(0.300);
cfg.average_fraction = single(0.5);
cfg.presentation_steps = round(cfg.presentation_time / cfg.dt);
cfg.average_steps = round(cfg.average_fraction * cfg.presentation_steps);
cfg.average_start_step = cfg.presentation_steps - cfg.average_steps + 1;
cfg.batch_size = 256;
cfg.validate_every = 5;

cfg.long_simulation_time = single(2000);
cfg.burn_in_time = single(10);
cfg.training_window = single(20);
cfg.teacher_steps = 30;
cfg.closed_loop_steps = 55;
cfg.validate_dynamics_every = 100;
cfg.validation_time = single(50);
cfg.validation_warmup_time = single(0);
cfg.test_time = single(50);
cfg.test_warmup_time = single(5);
cfg.validation_initial_conditions = 5;
cfg.test_initial_conditions = 5;
cfg.validation_initial_condition_seed = 1001;
cfg.test_initial_condition_seed = 123;
cfg.initial_condition_jitter = single(0.01);
cfg.phase_metric = struct('projections', 128, 'trim_fraction', 0.10, ...
    'subsample', 5, 'transient_fraction', 0.10, 'max_points', 1250);

cfg.checkpoint_hours = inf;
cfg.output_directory = fullfile(fileparts(mfilename('fullpath')), 'outputs');
cfg.model_file = '';
cfg.verbose_every = 10;

switch task
    case "breast_cancer"
        cfg.kind = "classification";
        cfg.dataset_file = 'breast_cancer_dataset.mat';
        cfg.epochs = 5000;
    case "mnist"
        cfg.kind = "classification";
        cfg.dataset_file = 'mnist.mat';
        cfg.epochs = 1000;
    case "afro_mnist_vai"
        cfg.kind = "classification";
        cfg.dataset_file = 'afro_mnist_vai.mat';
        cfg.epochs = 1000;
    case "abalone"
        cfg.kind = "regression";
        cfg.dataset_file = 'abalone_dataset.mat';
        cfg.epochs = 25000;
    case "toyota"
        cfg.kind = "regression";
        cfg.dataset_file = 'toyota_dataset.mat';
        cfg.epochs = 25000;
    case "yacht"
        cfg.kind = "regression";
        cfg.dataset_file = 'yacht_dataset.mat';
        cfg.epochs = 25000;
    case "lorenz"
        cfg.kind = "dynamics";
        cfg.dataset_file = '';
        cfg.system_rate = single(2);
        cfg.epochs = 100000;
    case "sprott_s"
        cfg.kind = "dynamics";
        cfg.dataset_file = '';
        cfg.system_rate = single(4);
        cfg.epochs = 100000;
    case "vanderpol"
        cfg.kind = "dynamics";
        cfg.dataset_file = '';
        cfg.system_rate = single(8);
        cfg.training_window = single(5);
        cfg.epochs = 100000;
end

cfg = merge_struct(cfg, overrides);
if ~isfield(overrides, 'initial_bias') || isempty(cfg.initial_bias)
    cfg.initial_bias = single(cfg.threshold_voltage - cfg.resting_voltage);
else
    cfg.initial_bias = single(cfg.initial_bias);
end
cfg.method = lower(string(cfg.method));
cfg.recurrent_mode = lower(string(cfg.recurrent_mode));
cfg.training_profile = lower(string(cfg.training_profile));
cfg.eligibility_mode = lower(string(cfg.eligibility_mode));
cfg.optimizer = lower(string(cfg.optimizer));
if cfg.recurrent_mode == "full_rank"
    cfg.N_hidden = value_or_override(overrides, 'N_hidden', 6000);
    if cfg.task == "breast_cancer"
        cfg.epochs = value_or_override(overrides, 'epochs', 2000);
    elseif cfg.task == "yacht"
        cfg.epochs = value_or_override(overrides, 'epochs', 10000);
    end
end
if cfg.method == "spsa" && cfg.task == "yacht"
    cfg.epochs = value_or_override(overrides, 'epochs', 50000);
end
if cfg.method == "spsa" && cfg.task == "breast_cancer"
    cfg.epochs = value_or_override(overrides, 'epochs', 50000);
end
if cfg.method == "spsa" && cfg.task == "vanderpol"
    cfg.epochs = value_or_override(overrides, 'epochs', 200000);
end
if cfg.method == "spsa" && cfg.kind == "dynamics"
    cfg.validation_warmup_time = value_or_override( ...
        overrides, 'validation_warmup_time', cfg.test_warmup_time);
end
if cfg.training_profile == "neuron_sweep" && cfg.task == "yacht"
    cfg.epochs = value_or_override(overrides, 'epochs', 25000);
end
if isempty(cfg.spsa_schedule_epochs)
    cfg.spsa_schedule_epochs = cfg.epochs;
end
if isempty(cfg.learning_rate_schedule_epochs)
    cfg.learning_rate_schedule_epochs = cfg.epochs;
end
cfg.presentation_steps = round(cfg.presentation_time / cfg.dt);
cfg.average_steps = round(cfg.average_fraction * cfg.presentation_steps);
cfg.average_start_step = cfg.presentation_steps - cfg.average_steps + 1;
validate_config(cfg);
% Stable fingerprints make result/checkpoint identity explicit and prevent
% accidental mixing of different scientific configurations.
cfg.scientific_config_sha256 = config_hash(cfg, false);
cfg.checkpoint_config_sha256 = config_hash(cfg, true);
cfg.model_file = result_file(cfg);
end

function validate_config(cfg)
% Fail early on malformed scientific settings. Keeping validation here makes
% every training, testing and cluster entry point use the same rules.
if ~any(cfg.method == ["eprop" "spsa"])
    error('banff:method', 'method must be "eprop" or "spsa".');
end
if ~any(cfg.recurrent_mode == ["low_rank" "full_rank"])
    error('banff:architecture', 'recurrent_mode must be "low_rank" or "full_rank".');
end
if ~any(cfg.training_profile == ["main" "neuron_sweep"])
    error('banff:trainingProfile', 'training_profile must be "main" or "neuron_sweep".');
end
if ~any(cfg.eligibility_mode == ["surrogate" "hard_spike"])
    error('banff:eligibilityMode', 'eligibility_mode must be "surrogate" or "hard_spike".');
end
if cfg.optimizer ~= "amsgrad"
    error('banff:optimizer', 'optimizer must be "amsgrad".');
end

positiveIntegers = {'N_hidden','N_recurrent','batch_size','epochs', ...
    'validate_every','validate_dynamics_every','teacher_steps','closed_loop_steps', ...
    'validation_initial_conditions','test_initial_conditions'};
for i = 1:numel(positiveIntegers)
    value = double(cfg.(positiveIntegers{i}));
    if ~isscalar(value) || ~isfinite(value) || value < 1 || value ~= round(value)
        error('banff:positiveInteger', '%s must be a positive integer.', positiveIntegers{i});
    end
end
nonnegativeIntegers = {'seed','split_seed','training_seed','validation_initial_condition_seed', ...
    'test_initial_condition_seed'};
for i = 1:numel(nonnegativeIntegers)
    value = double(cfg.(nonnegativeIntegers{i}));
    if ~isscalar(value) || ~isfinite(value) || value < 0 || value ~= round(value)
        error('banff:nonnegativeInteger', '%s must be a non-negative integer.', nonnegativeIntegers{i});
    end
end

positiveFinite = {'dt','tau_membrane','tau_adaptation','tau_synapse_rise', ...
    'tau_synapse_decay','decoder_gain','learning_rate_start', ...
    'learning_rate_end','adam_epsilon','presentation_time','long_simulation_time', ...
    'training_window','validation_time','test_time'};
for i = 1:numel(positiveFinite)
    value = double(cfg.(positiveFinite{i}));
    if ~isscalar(value) || ~isfinite(value) || value <= 0
        error('banff:positiveFinite', '%s must be positive and finite.', positiveFinite{i});
    end
end
if cfg.tau_synapse_rise >= cfg.tau_synapse_decay
    error('banff:synapseTimeConstants', 'tau_synapse_rise must be smaller than tau_synapse_decay.');
end

voltages = [cfg.resting_voltage cfg.threshold_voltage cfg.reset_voltage cfg.initial_bias];
if any(~isfinite(double(voltages)))
    error('banff:voltageFinite', 'Voltage parameters and initial_bias must be finite.');
end
if cfg.threshold_voltage <= cfg.resting_voltage || cfg.reset_voltage >= cfg.threshold_voltage
    error('banff:voltageScale', ...
        'Require resting_voltage < threshold_voltage and reset_voltage < threshold_voltage.');
end
if ~isfinite(double(cfg.adaptation_jump)) || cfg.adaptation_jump < 0
    error('banff:adaptationJump', 'adaptation_jump must be finite and non-negative.');
end
if ~isfinite(double(cfg.recurrent_gain)) || cfg.recurrent_gain < 0
    error('banff:recurrentGain', 'recurrent_gain must be finite and non-negative.');
end
if ~isfinite(double(cfg.encoder_gain)) || cfg.encoder_gain < 0
    error('banff:encoderGain', 'encoder_gain must be finite and non-negative.');
end
if ~isfinite(double(cfg.excitatory_fraction)) || cfg.excitatory_fraction < 0 || cfg.excitatory_fraction > 1
    error('banff:excitatoryFraction', 'excitatory_fraction must lie in [0,1].');
end
if ~isfinite(double(cfg.full_rank_probability)) || cfg.full_rank_probability <= 0 || cfg.full_rank_probability > 1
    error('banff:fullRankProbability', 'full_rank_probability must lie in (0,1].');
end
if ~isfinite(double(cfg.hard_event_gain)) || cfg.hard_event_gain <= 0
    error('banff:hardEventGain', 'hard_event_gain must be positive and finite.');
end
if cfg.eligibility_mode == "surrogate"
    if ~isfinite(double(cfg.surrogate_peak)) || cfg.surrogate_peak <= 0
        error('banff:surrogatePeak', 'surrogate_peak must be positive and finite.');
    end
    if ~isfinite(double(cfg.surrogate_half_width)) || cfg.surrogate_half_width <= 0
        error('banff:surrogateHalfWidth', 'surrogate_half_width must be positive and finite.');
    end
end
if cfg.adam_beta1 < 0 || cfg.adam_beta1 >= 1 || cfg.adam_beta2 < 0 || cfg.adam_beta2 >= 1
    error('banff:adamBeta', 'adam_beta1 and adam_beta2 must lie in [0,1).');
end
if cfg.average_fraction <= 0 || cfg.average_fraction > 1
    error('banff:averageFraction', 'average_fraction must lie in (0,1].');
end
if cfg.presentation_steps < 1 || cfg.average_steps < 1 || cfg.average_steps > cfg.presentation_steps
    error('banff:presentation', 'The presentation/readout window is invalid.');
end
nonnegativeDurations = {'burn_in_time','validation_warmup_time','test_warmup_time', ...
    'initial_condition_jitter'};
for i = 1:numel(nonnegativeDurations)
    value = double(cfg.(nonnegativeDurations{i}));
    if ~isscalar(value) || ~isfinite(value) || value < 0
        error('banff:nonnegativeValue', '%s must be non-negative and finite.', nonnegativeDurations{i});
    end
end
if cfg.kind == "dynamics" && (~isfield(cfg,'system_rate') || ~isfinite(double(cfg.system_rate)) || cfg.system_rate <= 0)
    error('banff:systemRate', 'Dynamical-system tasks require a positive finite system_rate.');
end
if cfg.recurrent_mode == "low_rank" && cfg.N_recurrent > cfg.N_hidden
    error('banff:recurrentRank', 'N_recurrent cannot exceed N_hidden in low-rank mode.');
end
for name = {'spsa_schedule_epochs','learning_rate_schedule_epochs','verbose_every'}
    value = double(cfg.(name{1}));
    if ~isscalar(value) || ~isfinite(value) || value < 1 || value ~= round(value)
        error('banff:positiveInteger', '%s must be a positive integer.', name{1});
    end
end
for name = {'spsa_c_start','spsa_c_end'}
    value = double(cfg.(name{1}));
    if ~isscalar(value) || ~isfinite(value) || value <= 0
        error('banff:spsaScale', '%s must be positive and finite.', name{1});
    end
end
if ~(isscalar(cfg.checkpoint_hours) && ...
        ((isfinite(double(cfg.checkpoint_hours)) && cfg.checkpoint_hours >= 0) || isinf(double(cfg.checkpoint_hours))))
    error('banff:checkpointHours', 'checkpoint_hours must be non-negative or Inf.');
end
validate_phase_metric(cfg.phase_metric);
end

function validate_phase_metric(metric)
required = {'projections','trim_fraction','subsample','transient_fraction','max_points'};
if ~isstruct(metric) || ~all(isfield(metric, required))
    error('banff:phaseMetric', 'phase_metric is missing required fields.');
end
if metric.projections < 2 || metric.projections ~= round(metric.projections) || ...
        metric.subsample < 1 || metric.subsample ~= round(metric.subsample) || ...
        metric.max_points < 2 || metric.max_points ~= round(metric.max_points)
    error('banff:phaseMetric', 'Invalid integer phase_metric settings.');
end
if metric.trim_fraction < 0 || metric.trim_fraction >= .5 || ...
        2*floor(metric.trim_fraction*metric.projections) >= metric.projections || ...
        metric.transient_fraction < 0 || metric.transient_fraction >= 1
    error('banff:phaseMetric', 'Invalid phase_metric trimming/transient settings.');
end
end

function task = canonical_task(task)
task = lower(string(task));
task = replace(task, "-", "_");
task = replace(task, " ", "_");
switch task
    case {"bc", "breastcancer"}
        task = "breast_cancer";
    case {"afromnist", "vai", "afro_mnist"}
        task = "afro_mnist_vai";
    case {"car", "car_price"}
        task = "toyota";
    case {"sprott", "sprotts"}
        task = "sprott_s";
end
valid = ["breast_cancer","mnist","afro_mnist_vai","abalone", ...
    "toyota","yacht","lorenz","sprott_s","vanderpol"];
if ~any(task == valid)
    error('banff:task', 'Unknown task "%s".', task);
end
end


%% Small configuration helpers
function result = merge_struct(base, changes)
result = base;
names = fieldnames(changes);
for index = 1:numel(names)
    result.(names{index}) = changes.(names{index});
end
end

function value = value_or_override(overrides, field, defaultValue)
if isfield(overrides, field)
    value = overrides.(field);
else
    value = defaultValue;
end
end

function file = result_file(cfg)
% A custom model_file override is respected. Otherwise derive a deterministic
% filename from task/method/architecture/configuration/seed.
if isfield(cfg, 'model_file') && ~isempty(cfg.model_file)
    file = char(cfg.model_file);
    return;
end
shortHash = cfg.scientific_config_sha256(1:12);
if cfg.method == "eprop"
    name = sprintf('%s_%s_%s_%s_N%d_cfg%s_seed%03d.mat', ...
        cfg.task, cfg.method, cfg.recurrent_mode, cfg.eligibility_mode, ...
        cfg.N_hidden, shortHash, cfg.seed);
else
    name = sprintf('%s_%s_%s_N%d_cfg%s_seed%03d.mat', ...
        cfg.task, cfg.method, cfg.recurrent_mode, cfg.N_hidden, shortHash, cfg.seed);
end
file = fullfile(cfg.output_directory, 'models', name);
end

function hash = config_hash(cfg, includeSeed)
clean = cfg;
% Runtime/location fields do not define the scientific experiment.
runtimeFields = {'checkpoint_hours','output_directory','model_file','verbose_every', ...
    'scientific_config_sha256','checkpoint_config_sha256'};
for index = 1:numel(runtimeFields)
    if isfield(clean, runtimeFields{index})
        clean = rmfield(clean, runtimeFields{index});
    end
end
if ~includeSeed && isfield(clean, 'seed')
    clean = rmfield(clean, 'seed');
end
% Preserve hashes of existing one-phase experiments created before the obsolete
% continuation setting was removed.  The former zero value had no effect.
if ~isfield(clean, 'spsa_continuation_boundary')
    clean.spsa_continuation_boundary = 0;
end
% Do not let parameters that are inactive in the selected method change the
% experiment identity.
if isfield(clean, 'method') && string(clean.method) == "spsa"
    inactive = {'eligibility_mode','hard_event_gain','surrogate_peak','surrogate_half_width'};
elseif isfield(clean, 'eligibility_mode') && string(clean.eligibility_mode) == "hard_spike"
    inactive = {'surrogate_peak','surrogate_half_width'};
else
    inactive = {'hard_event_gain'};
end
for index = 1:numel(inactive)
    if isfield(clean, inactive{index})
        clean = rmfield(clean, inactive{index});
    end
end
clean = canonicalize_struct(clean);
hash = sha256_text(jsonencode(clean));
end

function value = canonicalize_struct(value)
if isstruct(value)
    names = sort(fieldnames(value));
    ordered = struct();
    for index = 1:numel(names)
        ordered.(names{index}) = canonicalize_struct(value.(names{index}));
    end
    value = ordered;
elseif iscell(value)
    for index = 1:numel(value)
        value{index} = canonicalize_struct(value{index});
    end
end
end

function hash = sha256_text(text)
engine = javaMethod('getInstance', 'java.security.MessageDigest', 'SHA-256');
engine.update(uint8(unicode2native(char(text), 'UTF-8')));
digest = typecast(engine.digest(), 'uint8');
hash = lower(reshape(dec2hex(digest).', 1, []));
end
