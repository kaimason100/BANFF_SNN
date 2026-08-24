function varargout = banff_plot(action, varargin)
%BANFF_PLOT Plot/replay adapter for publication figures.
%   Training and testing use BANFF_MODEL. Plotting reuses the readable CPU
%   reference step in BANFF_MODEL instead of maintaining a third copy of the
%   neuron equations.

switch lower(string(action))
    case "model"
        varargout{1} = validate_model(varargin{:});
    case "static_data"
        varargout{1} = static_data(varargin{:});
    case "closed_loop"
        varargout{1} = closed_loop_input(varargin{:});
    case "step"
        [varargout{1:nargout}] = replay_step(varargin{:});
    case "cascade"
        [varargout{1:nargout}] = replay_cascade(varargin{:});
    case "static_traces"
        [varargout{1:nargout}] = static_traces(varargin{:});
    case "phase_distance"
        varargout{1} = phase_distance(varargin{:});
    otherwise
        error('banff:plot', 'Unknown plotting action "%s".', action);
end
end

function P = validate_model(P, options)
required = {'N_hidden','N_output','W_in','W_out','B','recurrent_mode'};
missing = required(~isfield(P, required));
if ~isempty(missing)
    error('banff:publicationModel', 'Model lacks required field(s): %s.', ...
        strjoin(missing, ', '));
end
if nargin > 1 && isstruct(options) && isfield(options, 'recurrent_mode') && ...
        string(P.recurrent_mode) ~= string(options.recurrent_mode)
    error('banff:publicationArchitecture', ...
        'Regenerated model and publication options disagree about the architecture.');
end
end

function data = static_data(domain, options)
if nargin < 2 || ~isstruct(options)
    error('banff:plotData', 'Publication options are required.');
end
saved = struct('train_index', options.idx_train, ...
    'validation_index', options.idx_val, 'test_index', options.idx_test, ...
    'feature_mean', field_or(options, 'feature_mean', []), ...
    'feature_std', field_or(options, 'feature_std', []), ...
    'target_mean', field_or(options, 'target_mean', []), ...
    'target_std', field_or(options, 'target_std', []), ...
    'dataset_sha256', field_or(options, 'dataset_sha256', ''));
[raw, information] = banff_data('static', options, saved);
if string(domain) ~= string(options.kind)
    error('banff:plotDataDomain', 'Requested and saved task domains disagree.');
end
data = struct('X_train', raw.X_train, 'Y_train', raw.Y_train, ...
    'X_val', raw.X_validation, 'Y_val', raw.Y_validation, ...
    'X_test', raw.X_test, 'Y_test', raw.Y_test, ...
    'idx_train', information.train_index, 'idx_val', information.validation_index, ...
    'idx_test', information.test_index, 'mu_X', information.feature_mean, ...
    'sigma_X', information.feature_std, 'mu_y', information.target_mean, ...
    'sigma_y', information.target_std);
end

function evaluation = closed_loop_input(options)
cfg = options;
cfg.test_time = single(field_or(options, 'T_sim', options.test_time));
cfg.test_warmup_time = single(field_or(options, 'closed_loop_warmup_time', ...
    field_or(options, 'closed_loop_test_warmup_time', options.test_warmup_time)));
if isfield(options, 'closed_loop_test_ics')
    cfg.test_initial_conditions = options.closed_loop_test_ics;
end
if isfield(options, 'closed_loop_test_ic_seed')
    cfg.test_initial_condition_seed = options.closed_loop_test_ic_seed;
end
system = banff_data('system', cfg.task);
initial = banff_data('initial_conditions', cfg, "test");
duration = cfg.test_time + cfg.test_warmup_time;
normalised = single((banff_data('trajectory', system, initial(:, 1), duration, cfg) ...
    - options.dynamics_mu) ./ options.dynamics_sigma);
evaluation = struct('x_true', {{normalised}}, ...
    'lambda', {{[true false(1, size(normalised, 2)-1)]}}, ...
    'warmup_steps', round(cfg.test_warmup_time / cfg.dt));
end

function [u, w, rho, spike, localGate, xSyn, r] = ...
        replay_step(P, inputCurrent, u, w, xSyn, r)
state = reference_state(P, u, w, xSyn, r);
[state, spike, rho, ~, localGate] = ...
    banff_model('reference_step', P, state, inputCurrent, false);
u = state.u;
w = state.w;
xSyn = state.x;
r = state.r;
end

function [spikes, voltage] = static_traces(P, X, options)
sampleCount = size(X, 2);
steps = double(options.steps_present);
spikes = false(P.N_hidden, steps, sampleCount);
voltage = zeros(P.N_hidden, steps, 'single');
for sample = 1:sampleCount
    u = repmat(single(P.restingVoltage), P.N_hidden, 1);
    w = zeros(P.N_hidden, 1, 'single');
    x = zeros(P.N_hidden, 1, 'single');
    r = zeros(P.N_hidden, 1, 'single');
    inputCurrent = P.W_in * (P.inputScale .* single(X(:, sample)));
    for step = 1:steps
        [u, w, ~, fired, ~, x, r] = ...
            replay_step(P, inputCurrent, u, w, x, r);
        spikes(:, step, sample) = fired;
        if sample == 1
            voltage(:, step) = u;
        end
    end
end
end

function state = reference_state(P, u, w, x, r)
state = struct('u', single(u), 'w', single(w), 'x', single(x), 'r', single(r), ...
    'epsilonVoltage', zeros(size(u), 'single'), ...
    'epsilonAdaptation', zeros(size(u), 'single'), ...
    'eligibilityRise', zeros(size(u), 'single'), ...
    'eligibilityDecay', zeros(size(u), 'single'));
if size(state.u, 1) ~= P.N_hidden
    error('banff:plotState', 'Replay state does not match the model size.');
end
end

function distance = phase_distance(prediction, truth, options)
if nargin < 3 || isempty(options), options = struct(); end
metric = struct('projections', field_or(options, 'NumProjections', 128), ...
    'trim_fraction', field_or(options, 'TrimFraction', .10), ...
    'subsample', field_or(options, 'Subsample', 5), ...
    'transient_fraction', field_or(options, 'TransientFraction', .10), ...
    'max_points', field_or(options, 'MaxPoints', 1250));
distance = banff_metrics('phase_distance', prediction, truth, metric);
end

function [xSyn, r] = replay_cascade(P, fraction, xSyn, r)
rise = exp(single(fraction) .* log(max(P.gammaRise, realmin('single'))));
decay = exp(single(fraction) .* log(max(P.gammaDecay, realmin('single'))));
xSyn = rise .* xSyn;
r = decay .* r + (single(1)-decay) .* xSyn;
end

function value = field_or(S, name, defaultValue)
if isfield(S, name) && ~isempty(S.(name))
    value = S.(name);
else
    value = defaultValue;
end
end
