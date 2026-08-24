function varargout = banff_model(action, varargin)
%BANFF_MODEL Fixed-weight spiking network and MATLAB GPU simulator.
%   This is the complete mathematical core of the publication code. Structural
%   weights are deterministic and fixed; only the hidden bias B is trainable.
%   Element-wise neuron, synapse and eligibility updates are fused by GPU
%   ARRAYFUN. No CUDA source or MEX file is required.

switch lower(string(action))
    case "create"
        varargout{1} = create_model(varargin{:});
    case "gpu"
        varargout{1} = move_to_gpu(varargin{:});
    case "static"
        [varargout{1:nargout}] = simulate_static(varargin{:});
    case "dynamics"
        [varargout{1:nargout}] = simulate_dynamics(varargin{:});
    case "adam"
        varargout{1} = adam_update(varargin{:});
    case "gather"
        varargout{1} = gather_trainable_state(varargin{:});
    case "reference_step"
        [varargout{1:nargout}] = reference_step(varargin{:});
    case "gpu_step"
        [varargout{1:nargout}] = gpu_step_for_test(varargin{:});
    otherwise
        error('banff:modelAction', 'Unknown model action "%s".', action);
end
end

%% Model construction
function P = create_model(nInput, nOutput, cfg)
nHidden = double(cfg.N_hidden);
nRecurrent = double(cfg.N_recurrent);

seed = uint64(cfg.seed);
daleSign = -ones(1, nHidden, 'single');
daleSign(index_uniform(5, 1:nHidden, 0, seed) <= cfg.excitatory_fraction) = 1;

% Fixed input encoder, Equation 10 and Table 2.
P.W_in = zeros(nHidden, nInput, 'single');
for neuron = 1:nHidden
    values = index_uniform(1, neuron, 1:nInput, seed);
    P.W_in(neuron, :) = sqrt(single(3)) * cfg.encoder_gain ...
        .* (single(2) .* single(values) - single(1));
end

% Fixed recurrent scaffold.  The principal architecture is factorized; the
% supplementary full-rank architecture uses one sparse matrix.  Both obey the
% same presynaptic Dale-like sign convention.
P.recurrent_mode = string(cfg.recurrent_mode);
if P.recurrent_mode == "low_rank"
    P.W_feedback = zeros(nRecurrent, nHidden, 'single');
    feedbackScale = sqrt(single(3) ./ single(nHidden));
    for row = 1:nRecurrent
        values = index_uniform(20, row, 1:nHidden, seed);
        weights = feedbackScale .* (single(2) .* single(values) - single(1));
        P.W_feedback(row, :) = abs(weights) .* daleSign;
    end

    P.recurrent_expansion = zeros(nHidden, nRecurrent, 'single');
    expansionScale = single(1) ./ sqrt(single(nRecurrent));
    for column = 1:nRecurrent
        weights = single(index_normal(30, (1:nHidden).', column, seed));
        P.recurrent_expansion(:, column) = expansionScale .* abs(weights);
    end
    P.self_coupling = cfg.recurrent_gain .* sum( ...
        P.recurrent_expansion .* P.W_feedback.', 2);
    P.W_recurrent = sparse([], [], [], nHidden, nHidden);
    P.recurrent_storage = "factorized";
else
    P.W_feedback = zeros(0, 0, 'single');
    P.recurrent_expansion = zeros(0, 0, 'single');
    P.self_coupling = zeros(0, 1, 'single');
    P.W_recurrent = create_full_rank_recurrence(nHidden, daleSign, cfg, seed);
    P.recurrent_storage = "sparse_double";
end

% Independent fixed signed-uniform task decoder.
decoderSeed = seed + uint64(1009);
values = index_uniform(61, 1:nOutput, 1:nHidden, decoderSeed);
P.W_out = cfg.decoder_gain .* sqrt(single(3)) ...
    .* single(single(2) .* single(values) - single(1)) ./ sqrt(single(nHidden));

P.B = repmat(single(cfg.initial_bias), nHidden, 1);
P.m = zeros(nHidden, 1, 'single');
P.v = zeros(nHidden, 1, 'single');
P.vMax = zeros(nHidden, 1, 'single');
P.adamStep = 0;

P.N_input = nInput;
P.N_output = nOutput;
P.N_hidden = nHidden;
P.N_recurrent = nRecurrent;
P.presentationSteps = double(cfg.presentation_steps);
P.averageSteps = double(cfg.average_steps);
P.averageStartStep = double(cfg.average_start_step);
P.inputScale = single(1 / sqrt(nInput));
P.recurrentGain = single(cfg.recurrent_gain);
P.dt = single(cfg.dt);
P.alpha = single(exp(-cfg.dt / cfg.tau_membrane));
P.beta = single(exp(-cfg.dt / cfg.tau_adaptation));
P.gammaRise = single(exp(-cfg.dt / cfg.tau_synapse_rise));
P.gammaDecay = single(exp(-cfg.dt / cfg.tau_synapse_decay));
P.logAlpha = single(log(P.alpha));
P.logBeta = single(log(P.beta));
P.logGammaRise = single(log(P.gammaRise));
P.logGammaDecay = single(log(P.gammaDecay));
P.restingVoltage = single(cfg.resting_voltage);
P.thresholdVoltage = single(cfg.threshold_voltage);
P.resetVoltage = single(cfg.reset_voltage);
P.adaptationJump = single(cfg.adaptation_jump);
% Eligibility-learning mode.  Keep the GPU kernel flag numeric so it is
% compatible with ARRAYFUN in MATLAB R2023a.
%   0 = continuous triangular surrogate
%   1 = hard-spike-gated eligibility
P.hardSpikeEligibility = single(cfg.eligibility_mode == "hard_spike");
P.hardEventGain = single(cfg.hard_event_gain); % 1/mV
P.eligibility_mode = string(cfg.eligibility_mode);

% Triangular surrogate pseudo-derivative parameters.  These are used only
% when eligibility_mode == "surrogate".
P.surrogatePeak = single(cfg.surrogate_peak);
P.surrogateHalfWidth = single(cfg.surrogate_half_width);
P.synapticJump = single(-log(P.gammaRise) / P.dt);
P.seed = double(cfg.seed);

% Keep only one descriptive field name per mathematical quantity.
P.dale_sign = int8(daleSign(:));
end

function matrix = create_full_rank_recurrence(n, daleSign, cfg, seed)
% Sparse fixed full-rank proof architecture from Supplementary Equations.
probability = double(cfg.full_rank_probability);
blockSize = min(n, 512);
rowBlocks = cell(ceil(n / blockSize), 1);
columnBlocks = cell(size(rowBlocks));
valueBlocks = cell(size(rowBlocks));
block = 0;
for firstColumn = 1:blockSize:n
    block = block + 1;
    selectedColumns = firstColumn:min(n, firstColumn + blockSize - 1);
    mask = index_uniform(70, 1:n, selectedColumns, seed + uint64(2003)) ...
        < probability;
    mask(sub2ind(size(mask), selectedColumns, 1:numel(selectedColumns))) = false;
    [rowIndex, localColumn] = find(mask);
    columnIndex = reshape(selectedColumns(localColumn), [], 1);
    raw = index_normal_pairs(71, rowIndex, columnIndex, seed + uint64(3001));
    rowBlocks{block} = int32(rowIndex);
    columnBlocks{block} = int32(columnIndex);
    signs = reshape(daleSign(double(columnIndex)), [], 1);
    valueBlocks{block} = abs(single(raw(:))) .* signs;
end
rows = vertcat(rowBlocks{1:block});
columns = vertcat(columnBlocks{1:block});
values = vertcat(valueBlocks{1:block});
meanFanIn = max(single(1), single(numel(values)) / single(n));
values = single(cfg.recurrent_gain) .* values ./ sqrt(meanFanIn);
matrix = sparse(double(rows), double(columns), double(values), n, n);
end

%% GPU preparation
function P = move_to_gpu(P)
denseFields = {'W_in','W_feedback','recurrent_expansion','self_coupling', ...
    'W_out','B','m','v','vMax'};
for index = 1:numel(denseFields)
    field = denseFields{index};
    P.(field) = gpuArray(single(P.(field)));
end

scalarFields = {'inputScale','recurrentGain','alpha','beta','gammaRise', ...
    'gammaDecay','logAlpha','logBeta','logGammaRise','logGammaDecay', ...
    'restingVoltage','thresholdVoltage','resetVoltage','adaptationJump', ...
    'hardSpikeEligibility','hardEventGain','surrogatePeak','surrogateHalfWidth','synapticJump'};
for index = 1:numel(scalarFields)
    field = scalarFields{index};
    P.(field) = gpuArray(single(P.(field)));
end

if P.recurrent_mode == "full_rank"
    % MATLAB R2023a sparse gpuArray matrices are double precision.
    P.W_recurrent = gpuArray(P.W_recurrent);
else
    P.W_recurrent = gpuArray.zeros(0, 0, 'single');
end
end

%% Static classification and regression simulation
function [averageOutput, averageEligibility, spikeCount] = ...
        simulate_static(P, inputBatch, trackEligibility, recordSpikes)
if nargin < 4
    recordSpikes = false;
end
inputBatch = gpuArray(single(inputBatch));
trackEligibilityHost = logical(trackEligibility);
trackEligibilityGpu = gpuArray(trackEligibilityHost);
recordSpikes = logical(recordSpikes);
batchSize = size(inputBatch, 2);
state = initial_state(P, batchSize);
inputCurrent = P.W_in * (P.inputScale .* inputBatch);
outputSum = gpuArray.zeros(P.N_output, batchSize, 'single');
eligibilitySum = gpuArray.zeros(P.N_hidden, batchSize, 'single');
spikeCount = gpuArray.zeros(P.N_hidden, batchSize, 'single');

for step = 1:P.presentationSteps
    totalCurrent = inputCurrent + recurrent_current(P, state.r) + P.B;
    if recordSpikes
        [state, spike] = fused_step( ...
            P, state, totalCurrent, trackEligibilityGpu, true);
        spikeCount = spikeCount + single(spike);
    else
        state = fused_step(P, state, totalCurrent, trackEligibilityGpu, false);
    end
    if step >= P.averageStartStep
        outputSum = outputSum + P.W_out * state.r;
        if trackEligibilityHost
            eligibilitySum = eligibilitySum + state.eligibilityDecay;
        end
    end
end
averageOutput = outputSum ./ single(P.averageSteps);
averageEligibility = eligibilitySum ./ single(P.averageSteps);
end

%% Scheduled-sampling or closed-loop dynamical simulation
function [loss, biasGradient, output, events] = ...
        simulate_dynamics(P, targetSequence, teacherForcing, trackEligibility, recordEvents)
targetSequence = gpuArray(single(targetSequence));
teacherForcing = logical(teacherForcing);
trackEligibilityHost = logical(trackEligibility);
trackEligibilityGpu = gpuArray(trackEligibilityHost);
steps = size(targetSequence, 2) - 1;
state = initial_state(P, 1);
previousOutput = gpuArray.zeros(P.N_output, 1, 'single');
output = gpuArray.zeros(P.N_output, steps, 'single');
biasGradient = gpuArray.zeros(P.N_hidden, 1, 'single');
loss = gpuArray.zeros(1, 1, 'single');
events = struct('neuron', int32([]), 'step', int32([]), 'rho', single([]));

eventBlockSize = 1000;
if recordEvents
    spikeBlock = gpuArray.false(P.N_hidden, min(eventBlockSize, steps));
    rhoBlock = gpuArray.zeros(P.N_hidden, min(eventBlockSize, steps), 'single');
end

for step = 1:steps
    if step == 1 || teacherForcing(step)
        networkInput = targetSequence(:, step);
    else
        networkInput = previousOutput;
    end
    inputCurrent = P.W_in * (P.inputScale .* networkInput);
    totalCurrent = inputCurrent + recurrent_current(P, state.r) + P.B;
    [state, spike, rho] = fused_step( ...
        P, state, totalCurrent, trackEligibilityGpu, recordEvents);

    prediction = P.W_out * state.r;
    output(:, step) = prediction;
    previousOutput = prediction;
    errorSignal = prediction - targetSequence(:, step + 1);
    loss = loss + sum(errorSignal .* errorSignal, 'all');
    if trackEligibilityHost
        learningSignal = P.W_out.' * (single(2) .* errorSignal);
        biasGradient = biasGradient + learningSignal .* state.eligibilityDecay;
    end

    if recordEvents
        localStep = mod(step - 1, eventBlockSize) + 1;
        spikeBlock(:, localStep) = spike;
        rhoBlock(:, localStep) = rho;
        if localStep == size(spikeBlock, 2) || step == steps
            events = append_event_block(events, spikeBlock(:, 1:localStep), ...
                rhoBlock(:, 1:localStep), step - localStep);
            remaining = steps - step;
            if remaining > 0
                nextWidth = min(eventBlockSize, remaining);
                spikeBlock = gpuArray.false(P.N_hidden, nextWidth);
                rhoBlock = gpuArray.zeros(P.N_hidden, nextWidth, 'single');
            end
        end
    end
end
end

function state = initial_state(P, batchSize)
state = struct();
state.u = repmat(P.restingVoltage, P.N_hidden, batchSize);
state.w = gpuArray.zeros(P.N_hidden, batchSize, 'single');
state.x = gpuArray.zeros(P.N_hidden, batchSize, 'single');
state.r = gpuArray.zeros(P.N_hidden, batchSize, 'single');
state.epsilonVoltage = gpuArray.zeros(P.N_hidden, batchSize, 'single');
state.epsilonAdaptation = gpuArray.zeros(P.N_hidden, batchSize, 'single');
state.eligibilityRise = gpuArray.zeros(P.N_hidden, batchSize, 'single');
state.eligibilityDecay = gpuArray.zeros(P.N_hidden, batchSize, 'single');
end

function current = recurrent_current(P, filteredSpikes)
if P.recurrent_mode == "full_rank"
    % R2023a sparse GPU matrices are double precision.
    % Perform the sparse multiply in double, then return to the
    % single-precision state representation used by the rest of BANFF.
    current = single(P.W_recurrent * double(filteredSpikes));
else
    latentState = P.W_feedback * filteredSpikes;
    current = P.recurrentGain .* (P.recurrent_expansion * latentState) ...
        - P.self_coupling .* filteredSpikes;
end
end

function [state, spike, rho] = ...
        fused_step(P, state, totalCurrent, trackEligibility, returnEvents)
if returnEvents
    [state.u, state.w, state.x, state.r, state.epsilonVoltage, ...
        state.epsilonAdaptation, state.eligibilityRise, ...
        state.eligibilityDecay, spike, rho] = arrayfun( ...
        @neuron_synapse_eligibility_step, state.u, state.w, state.x, state.r, ...
        state.epsilonVoltage, state.epsilonAdaptation, state.eligibilityRise, ...
        state.eligibilityDecay, totalCurrent, trackEligibility, ...
        P.alpha, P.logAlpha, P.logBeta, P.logGammaRise, P.logGammaDecay, ...
        P.restingVoltage, P.thresholdVoltage, P.resetVoltage, ...
        P.adaptationJump, P.hardSpikeEligibility, P.hardEventGain, ...
        P.surrogatePeak, P.surrogateHalfWidth, P.synapticJump);
else
    [state.u, state.w, state.x, state.r, state.epsilonVoltage, ...
        state.epsilonAdaptation, state.eligibilityRise, ...
        state.eligibilityDecay] = arrayfun( ...
        @neuron_synapse_eligibility_step, state.u, state.w, state.x, state.r, ...
        state.epsilonVoltage, state.epsilonAdaptation, state.eligibilityRise, ...
        state.eligibilityDecay, totalCurrent, trackEligibility, ...
        P.alpha, P.logAlpha, P.logBeta, P.logGammaRise, P.logGammaDecay, ...
        P.restingVoltage, P.thresholdVoltage, P.resetVoltage, ...
        P.adaptationJump, P.hardSpikeEligibility, P.hardEventGain, ...
        P.surrogatePeak, P.surrogateHalfWidth, P.synapticJump);
    spike = [];
    rho = [];
end
end

function events = append_event_block(events, spikeGpu, rhoGpu, firstStep)
[neuronGpu, localStepGpu] = find(spikeGpu);
linearIndexGpu = sub2ind(size(rhoGpu), neuronGpu, localStepGpu);
neuron = gather(neuronGpu);
localStep = gather(localStepGpu);
rho = gather(rhoGpu(linearIndexGpu));
events.neuron = [events.neuron; int32(neuron)];
events.step = [events.step; int32(firstStep + localStep)];
events.rho = [events.rho; single(rho)];
end

%% Adam/AMSGrad bias update
function P = adam_update(P, gradient, learningRate, averageCount, cfg)
P.adamStep = P.adamStep + 1;
gradient = single(gradient) ./ single(max(1, averageCount));
b1 = single(cfg.adam_beta1);
b2 = single(cfg.adam_beta2);
P.m = b1 .* P.m + (single(1) - b1) .* gradient;
P.v = b2 .* P.v + (single(1) - b2) .* gradient .* gradient;
mHat = P.m ./ (single(1) - b1 .^ single(P.adamStep));
vHat = P.v ./ (single(1) - b2 .^ single(P.adamStep));
P.vMax = max(P.vMax, vHat);
P.B = P.B - single(learningRate) .* mHat ...
    ./ (sqrt(P.vMax) + single(cfg.adam_epsilon));
end

function state = gather_trainable_state(P)
state = struct('B', gather(P.B), 'm', gather(P.m), 'v', gather(P.v), ...
    'vMax', gather(P.vMax), 'adamStep', P.adamStep);
end

%% Fused scalar GPU kernel: forward ALIF/LSTI plus local-state e-prop
function [u, w, x, r, epsilonVoltage, epsilonAdaptation, eligibilityRise, ...
        eligibilityDecay, spikeOutput, rhoOutput, rawEligibilityOutput, ...
        localGateOutput] = ...
        neuron_synapse_eligibility_step(u0, w0, x0, r0, epsilonVoltage0, ...
        epsilonAdaptation0, eligibilityRise0, eligibilityDecay0, current, ...
        trackEligibility, alpha, logAlpha, logBeta, logGammaRise, ...
        logGammaDecay, restingVoltage, thresholdVoltage, resetVoltage, ...
        adaptationJump, hardSpikeEligibility, hardEventGain, ...
        surrogatePeak, surrogateHalfWidth, synapticJump)
one = single(1);
zero = single(0);
tiny = single(1.1754944e-38);

% Forward candidate state and hard spike decision.  As in the original
% e-prop implementation, the pseudo-derivative below is evaluated at the
% same pre-reset voltage variable used by the spike nonlinearity.
candidateVoltage = restingVoltage + alpha * (u0 - restingVoltage) ...
    + (one - alpha) * (current - w0);
spike = candidateVoltage >= thresholdVoltage;
voltageChange = candidateVoltage - u0;
rho = (thresholdVoltage - u0) / max(voltageChange, tiny);
rho = min(max(rho, zero), one);
if ~spike || voltageChange <= zero
    rho = zero;
end

% Forward LSTI split.  rho is a forward timing variable and is deliberately
% treated as stop-gradient in the local e-prop calculation.
alphaBefore = exp(rho * logAlpha);
betaBefore = exp(rho * logBeta);
alphaAfter = exp((one - rho) * logAlpha);
betaAfter = exp((one - rho) * logBeta);

eventVoltage = restingVoltage + alphaBefore * (u0 - restingVoltage) ...
    + (one - alphaBefore) * (current - w0);
eventAdaptation = betaBefore * w0;

if trackEligibility
    % Propagate the complete local bias-sensitivity state to the point at
    % which the eligibility source is defined.  For a real spike this is the
    % LSTI event time rho.  On a non-spiking step it is the end of the step.
    % This makes event magnitude and event timing refer to the same state.
    eligibilityRho = one;
    if spike
        eligibilityRho = rho;
    end
    alphaEligibilityBefore = exp(eligibilityRho * logAlpha);
    betaEligibilityBefore = exp(eligibilityRho * logBeta);
    alphaEligibilityAfter = exp((one - eligibilityRho) * logAlpha);
    betaEligibilityAfter = exp((one - eligibilityRho) * logBeta);

    epsilonVoltage = alphaEligibilityBefore * epsilonVoltage0 ...
        + (one - alphaEligibilityBefore) * (one - epsilonAdaptation0);
    epsilonAdaptation = betaEligibilityBefore * epsilonAdaptation0;

    if hardSpikeEligibility > zero
        % Principal event-gated rule.  hardEventGain has units 1/mV, so the
        % raw spike eligibility has the correct derivative units.
        if spike
            localGate = hardEventGain;
            rawEligibility = localGate * epsilonVoltage;
        else
            localGate = zero;
            rawEligibility = zero;
        end
    else
        % Optional continuous-surrogate ablation.  The pseudo-derivative is
        % evaluated at the same full-step candidate voltage used by the hard
        % spike decision.  On a spike, its state sensitivity is the LSTI
        % pre-event sensitivity above; otherwise it is the end-step sensitivity.
        scaledVoltage = (candidateVoltage - thresholdVoltage) ...
            / max(surrogateHalfWidth, tiny);
        pseudoDerivative = surrogatePeak ...
            * max(zero, one - abs(scaledVoltage));
        localGate = pseudoDerivative;
        rawEligibility = localGate * epsilonVoltage;
    end
else
    epsilonVoltage = epsilonVoltage0;
    epsilonAdaptation = epsilonAdaptation0;
    rawEligibility = zero;
    localGate = zero;
    eligibilityRho = zero;
    alphaEligibilityAfter = one;
    betaEligibilityAfter = one;
end

% Forward hard reset and spike-triggered adaptation.
if spike
    eventVoltage = resetVoltage;
    eventAdaptation = eventAdaptation + adaptationJump;
end

if trackEligibility
    % Match the original e-prop stop-gradient treatment of reset transmission:
    % the selected hard-reset branch is held fixed.  For this immediate-reset
    % neuron, that removes inherited membrane sensitivity on a hard spike.
    if spike
        epsilonVoltage = zero;
    end

    % Differentiate the spike-triggered adaptation jump using the selected
    % local spike derivative.  In the principal hard mode this occurs only at
    % an actual spike; the optional surrogate ablation may inject it on
    % near-threshold non-spiking steps.
    epsilonAdaptation = epsilonAdaptation ...
        + adaptationJump * rawEligibility;

    epsilonVoltage = alphaEligibilityAfter * epsilonVoltage ...
        + (one - alphaEligibilityAfter) * (one - epsilonAdaptation);
    epsilonAdaptation = betaEligibilityAfter * epsilonAdaptation;
end

u = restingVoltage + alphaAfter * (eventVoltage - restingVoltage) ...
    + (one - alphaAfter) * (current - eventAdaptation);
w = betaAfter * eventAdaptation;

% Forward rise-decay synaptic cascade: only hard spikes drive the actual
% recurrent/readout state.
riseBefore = exp(rho * logGammaRise);
decayBefore = exp(rho * logGammaDecay);
xAtEvent = riseBefore * x0;
rAtEvent = decayBefore * r0 + (one - decayBefore) * xAtEvent;
if spike
    xAtEvent = xAtEvent + synapticJump;
end
riseAfter = exp((one - rho) * logGammaRise);
decayAfter = exp((one - rho) * logGammaDecay);
x = riseAfter * xAtEvent;
r = decayAfter * rAtEvent + (one - decayAfter) * x;

if trackEligibility
    % Filter the instantaneous local spike eligibility through the same
    % rise-decay operator as the decoder state.  This is the readout-matched
    % eligibility filtering used by e-prop.
    riseEligibilityBefore = exp(eligibilityRho * logGammaRise);
    decayEligibilityBefore = exp(eligibilityRho * logGammaDecay);
    eligibilityAtEvent = riseEligibilityBefore * eligibilityRise0;
    eligibilityDecayAtEvent = decayEligibilityBefore * eligibilityDecay0 ...
        + (one - decayEligibilityBefore) * eligibilityAtEvent;

    % Inject the selected instantaneous eligibility source.  The source is
    % continuous/triangular in surrogate mode and hard-spike-gated otherwise.
    eligibilityAtEvent = eligibilityAtEvent + synapticJump * rawEligibility;

    riseEligibilityAfter = exp((one - eligibilityRho) * logGammaRise);
    decayEligibilityAfter = exp((one - eligibilityRho) * logGammaDecay);
    eligibilityRise = riseEligibilityAfter * eligibilityAtEvent;
    eligibilityDecay = decayEligibilityAfter * eligibilityDecayAtEvent ...
        + (one - decayEligibilityAfter) * eligibilityRise;
else
    eligibilityRise = eligibilityRise0;
    eligibilityDecay = eligibilityDecay0;
end

spikeOutput = spike;
rhoOutput = rho;
rawEligibilityOutput = rawEligibility;
localGateOutput = localGate;
end

%% CPU reference step used by diagnostics and mathematical tests
function [state, spike, rho, rawEligibility, localGate] = ...
        reference_step(P, state, inputCurrent, trackEligibility)
%REFERENCE_STEP CPU wrapper around the exact scalar timestep kernel.
%   The neuron/synapse/eligibility equations live only in
%   NEURON_SYNAPSE_ELIGIBILITY_STEP above. A small nested wrapper captures the
%   scalar model constants so CPU ARRAYFUN receives only equal-sized state
%   arrays (MATLAB CPU ARRAYFUN does not scalar-expand separate inputs).
if nargin < 4
    trackEligibility = false;
end
inputCurrent = single(inputCurrent);
current = inputCurrent + recurrent_current(P, state.r) + single(P.B);
[state.u, state.w, state.x, state.r, state.epsilonVoltage, ...
    state.epsilonAdaptation, state.eligibilityRise, state.eligibilityDecay, ...
    spike, rho, rawEligibility, localGate] = arrayfun(@cpu_kernel, ...
    single(state.u), single(state.w), single(state.x), single(state.r), ...
    single(state.epsilonVoltage), single(state.epsilonAdaptation), ...
    single(state.eligibilityRise), single(state.eligibilityDecay), current);

    function [u1, w1, x1, r1, epsU1, epsW1, eRise1, eDecay1, ...
            spike1, rho1, raw1, gate1] = ...
            cpu_kernel(u0, w0, x0, r0, epsU0, epsW0, eRise0, eDecay0, I)
        [u1, w1, x1, r1, epsU1, epsW1, eRise1, eDecay1, ...
            spike1, rho1, raw1, gate1] = neuron_synapse_eligibility_step( ...
            u0, w0, x0, r0, epsU0, epsW0, eRise0, eDecay0, I, ...
            logical(trackEligibility), single(P.alpha), single(P.logAlpha), ...
            single(P.logBeta), single(P.logGammaRise), ...
            single(P.logGammaDecay), single(P.restingVoltage), ...
            single(P.thresholdVoltage), single(P.resetVoltage), ...
            single(P.adaptationJump), single(P.hardSpikeEligibility), ...
            single(P.hardEventGain), single(P.surrogatePeak), ...
            single(P.surrogateHalfWidth), single(P.synapticJump));
    end
end

function [state, spike, rho] = gpu_step_for_test(P, state, inputCurrent, trackEligibility)
%GPU_STEP_FOR_TEST Expose one fused GPU step only for regression testing.
if ~isa(P.B, 'gpuArray')
    P = move_to_gpu(P);
end
fields = fieldnames(state);
for index = 1:numel(fields)
    if ~isa(state.(fields{index}), 'gpuArray')
        state.(fields{index}) = gpuArray(single(state.(fields{index})));
    end
end
inputCurrent = gpuArray(single(inputCurrent));
totalCurrent = inputCurrent + recurrent_current(P, state.r) + P.B;
[state, spike, rho] = fused_step(P, state, totalCurrent, ...
    gpuArray(logical(trackEligibility)), true);
end

%% Index-stable random numbers (unchanged from the publication implementation)
function values = index_uniform(matrixId, rows, columns, seed)
row64 = uint64(rows(:));
column64 = uint64(columns(:)).';
[rowGrid, columnGrid] = ndgrid(row64, column64);
a = uint64_constant('9E3779B97F4A7C15');
b = uint64_constant('BF58476D1CE4E5B9');
c = uint64_constant('94D049BB133111EB');
x = add_uint64(seed, multiply_uint64(a, uint64(matrixId) + 1));
x = add_uint64(x, multiply_uint64(b, rowGrid));
x = add_uint64(x, multiply_uint64(c, columnGrid));
mixed = splitmix64(x);
values = (double(bitshift(mixed, -11)) + 0.5) / 2^53;
end

function values = index_normal(matrixId, rows, columns, seed)
uniform = index_uniform(matrixId, rows, columns, seed);
uniform = min(max(uniform, realmin), 1 - eps);
values = sqrt(2) .* erfinv(2 .* uniform - 1);
end

function values = index_normal_pairs(matrixId, rows, columns, seed)
row64 = uint64(rows(:));
column64 = uint64(columns(:));
a = uint64_constant('9E3779B97F4A7C15');
b = uint64_constant('BF58476D1CE4E5B9');
c = uint64_constant('94D049BB133111EB');
x = add_uint64(seed, multiply_uint64(a, uint64(matrixId) + 1));
x = add_uint64(x, multiply_uint64(b, row64));
x = add_uint64(x, multiply_uint64(c, column64));
mixed = splitmix64(x);
uniform = (double(bitshift(mixed, -11)) + 0.5) / 2^53;
uniform = min(max(uniform, realmin), 1 - eps);
values = sqrt(2) .* erfinv(2 .* uniform - 1);
end

function mixed = splitmix64(x)
x = add_uint64(uint64(x), uint64_constant('9E3779B97F4A7C15'));
mixed = bitxor(x, bitshift(x, -30));
mixed = multiply_uint64(mixed, uint64_constant('BF58476D1CE4E5B9'));
mixed = bitxor(mixed, bitshift(mixed, -27));
mixed = multiply_uint64(mixed, uint64_constant('94D049BB133111EB'));
mixed = bitxor(mixed, bitshift(mixed, -31));
end

function result = add_uint64(a, b)
a = uint64(a);
b = uint64(b);
mask = uint64(4294967295);
low = bitand(a, mask) + bitand(b, mask);
carry = bitshift(low, -32);
low = bitand(low, mask);
high = bitand(bitshift(a, -32) + bitshift(b, -32) + carry, mask);
result = bitshift(high, 32) + low;
end

function result = multiply_uint64(a, b)
a = uint64(a);
b = uint64(b);
mask = uint64(4294967295);
aLow = bitand(a, mask);
aHigh = bitshift(a, -32);
bLow = bitand(b, mask);
bHigh = bitshift(b, -32);
middle = add_uint64(aLow .* bHigh, aHigh .* bLow);
result = add_uint64(bitshift(middle, 32), aLow .* bLow);
end

function value = uint64_constant(hexadecimal)
hexadecimal = upper(strtrim(hexadecimal));
high = uint64(base2dec(hexadecimal(1:8), 16));
low = uint64(base2dec(hexadecimal(9:16), 16));
value = bitshift(high, 32) + low;
end
