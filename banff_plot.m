function varargout = banff_plot(action, varargin)
%BANFF_PLOT Plot/replay adapter for publication figures.
%   Training and testing use BANFF_MODEL. Plotting reuses the readable CPU
%   reference step in BANFF_MODEL instead of maintaining a third copy of the
%   neuron equations.
%   This adapter contains presentation-oriented data extraction only. It does
%   not define scientific dynamics or learning rules, and any replayed state is
%   explicitly reconstructed from a validated model and supplied initial data.

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
    case "static_current_magnitudes"
        varargout{1} = static_current_magnitudes(varargin{:});
    case "static_inverse_isi_rates"
        varargout{1} = static_inverse_isi_rates(varargin{:});
    case "dynamics_current_magnitudes"
        varargout{1} = dynamics_current_magnitudes(varargin{:});
    case "phase_distance"
        varargout{1} = phase_distance(varargin{:});
    otherwise
        error('banff:plot', 'Unknown plotting action "%s".', action);
end
end

function inverseRatesHz = static_inverse_isi_rates(P,X,options)
%STATIC_INVERSE_ISI_RATES Pool all within-sample ISIs across an assessment set.
% Static samples are independent trials and reset the complete network state.
% Intervals therefore join consecutive spikes from the same neuron within the
% same sample, never spikes separated by a sample boundary. Every neuron and
% sample is included; a neuron with fewer than two spikes in one sample has
% no mathematically defined ISI for that sample and contributes no interval.
if ~isa(P.B,'gpuArray'), P=banff_model('gpu',P); end
sampleCount=size(X,2);
batchSize=max(1,round(double(field_or(options,'batch_size',256))));
maximumChunks=ceil(sampleCount/batchSize)*double(P.presentationSteps);
intervalChunks=cell(maximumChunks,1);
chunkCount=0;
for first=1:batchSize:sampleCount
    indices=first:min(sampleCount,first+batchSize-1);
    localBatch=numel(indices);
    inputBatch=gpuArray(single(X(:,indices)));
    inputCurrent=P.W_in*(P.inputScale.*inputBatch);
    state=diagnostic_gpu_state(P,localBatch);
    lastSpikeTimeInSteps=gpuArray.zeros(P.N_hidden,localBatch,'single');
    hasSpiked=gpuArray.false(P.N_hidden,localBatch);
    for step=1:P.presentationSteps
        [state,spike,rho]=banff_model('gpu_step',P,state,inputCurrent,false);
        eventTimeInSteps=single(step-1)+rho;
        repeatedSpike=spike & hasSpiked;
        intervalsInSteps=gather(eventTimeInSteps(repeatedSpike) ...
            -lastSpikeTimeInSteps(repeatedSpike));
        if ~isempty(intervalsInSteps)
            chunkCount=chunkCount+1;
            intervalChunks{chunkCount}=double(intervalsInSteps(:));
        end
        lastSpikeTimeInSteps(spike)=eventTimeInSteps(spike);
        hasSpiked=hasSpiked|spike;
    end
end
if chunkCount==0
    inverseRatesHz=zeros(0,1);
else
    intervalsInSteps=vertcat(intervalChunks{1:chunkCount});
    intervalsInSteps=intervalsInSteps( ...
        isfinite(intervalsInSteps) & intervalsInSteps>0);
    inverseRatesHz=1./(intervalsInSteps.*double(P.dt));
end
end

function P = validate_model(P, options)
%VALIDATE_MODEL Reject figure inputs that do not match requested provenance.
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
%STATIC_DATA Return a deterministic subset suitable for explanatory panels.
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
%CLOSED_LOOP_INPUT Reconstruct the tested autonomous trajectory for plotting.
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

function [spikes, voltage, diagnostics] = static_traces(P, X, options)
%STATIC_TRACES Replay selected samples with CPU-readable per-step state output.
sampleCount = size(X, 2);
steps = double(field_or(options, 'steps_present', options.presentation_steps));
spikes = false(P.N_hidden, steps, sampleCount);
voltage = zeros(P.N_hidden, steps, 'single');
diagnostics = struct('time_seconds', single((0:steps-1) .* double(P.dt)), ...
    'mean_encoder_current', zeros(1, steps, 'single'), ...
    'mean_recurrent_current', zeros(1, steps, 'single'), ...
    'mean_bias_current', zeros(1, steps, 'single'), ...
    'mean_adaptation_current', zeros(1, steps, 'single'));
for sample = 1:sampleCount
    u = repmat(single(P.restingVoltage), P.N_hidden, 1);
    w = zeros(P.N_hidden, 1, 'single');
    x = zeros(P.N_hidden, 1, 'single');
    r = zeros(P.N_hidden, 1, 'single');
    inputCurrent = P.W_in * (P.inputScale .* single(X(:, sample)));
    for step = 1:steps
        if sample == 1
            recurrentCurrent = replay_recurrent_current(P, r);
            diagnostics.mean_encoder_current(step) = mean(inputCurrent);
            diagnostics.mean_recurrent_current(step) = mean(recurrentCurrent);
            diagnostics.mean_bias_current(step) = mean(single(P.B));
            diagnostics.mean_adaptation_current(step) = mean(w);
        end
        [u, w, ~, fired, ~, x, r] = ...
            replay_step(P, inputCurrent, u, w, x, r);
        spikes(:, step, sample) = fired;
        if sample == 1
            voltage(:, step) = u;
        end
    end
end
end

function summary = static_current_magnitudes(P, X, options)
%STATIC_CURRENT_MAGNITUDES Aggregate exact per-neuron RMS magnitudes over all
% held-out samples and presentation timesteps without retaining the full
% neuron-by-time-by-sample tensor. This diagnostic reuses BANFF_MODEL's GPU
% step and therefore does not define a separate neuron update.
if ~isa(P.B,'gpuArray'), P=banff_model('gpu',P); end
sampleCount=size(X,2);
steps=double(P.presentationSteps);
batchSize=double(field_or(options,'batch_size',256));
if sampleCount<1 || steps<1 || batchSize<1
    error('banff:plotCurrentInput', ...
        'Current aggregation requires samples, presentation steps and a positive batch size.');
end
batchSize=max(1,round(batchSize));
encoderNetSquare=gpuArray.zeros(P.N_hidden,1,'single');
encoderGrossSquare=gpuArray.zeros(P.N_hidden,1,'single');
recurrentNetSquare=gpuArray.zeros(P.N_hidden,1,'single');
recurrentGrossSquare=gpuArray.zeros(P.N_hidden,1,'single');
adaptationSquare=gpuArray.zeros(P.N_hidden,1,'single');
decoderStateSquare=gpuArray.zeros(P.N_hidden,1,'single');

for first=1:batchSize:sampleCount
    indices=first:min(sampleCount,first+batchSize-1);
    localBatch=numel(indices);
    inputBatch=gpuArray(single(X(:,indices)));
    inputCurrent=P.W_in*(P.inputScale.*inputBatch);
    grossInputCurrent=abs(P.W_in)*(P.inputScale.*abs(inputBatch));
    encoderNetSquare=encoderNetSquare+single(steps).*sum(inputCurrent.^2,2);
    encoderGrossSquare=encoderGrossSquare+single(steps).*sum(grossInputCurrent.^2,2);
    state=diagnostic_gpu_state(P,localBatch);
    for step=1:steps
        recurrentCurrent=replay_recurrent_current(P,state.r);
        grossRecurrentCurrent=replay_recurrent_gross(P,state.r);
        recurrentNetSquare=recurrentNetSquare+sum(recurrentCurrent.^2,2);
        recurrentGrossSquare=recurrentGrossSquare+sum(grossRecurrentCurrent.^2,2);
        adaptationSquare=adaptationSquare+sum(state.w.^2,2);
        state=banff_model('gpu_step',P,state,inputCurrent,false);
        if step>=P.averageStartStep
            decoderStateSquare=decoderStateSquare+sum(state.r.^2,2);
        end
    end
end

observationCount=double(sampleCount).*steps;
decoderObservationCount=double(sampleCount).*double(P.averageSteps);
biasReference=field_or(options,'initial_bias', ...
    P.thresholdVoltage-P.restingVoltage);
if isa(biasReference,'gpuArray'), biasReference=gather(biasReference); end
biasReference=single(biasReference);
decoderWeightMeanSquare=mean(P.W_out.^2,1).';
summary=struct( ...
    'encoder_net_rms',single(sqrt(gather(encoderNetSquare)./single(observationCount))), ...
    'encoder_gross_afferent_rms',single(sqrt(gather(encoderGrossSquare) ...
        ./single(observationCount))), ...
    'recurrent_net_rms',single(sqrt(gather(recurrentNetSquare)./single(observationCount))), ...
    'recurrent_gross_afferent_rms',single(sqrt(gather(recurrentGrossSquare) ...
        ./single(observationCount))), ...
    'decoder_presynaptic_rms',single(sqrt(gather( ...
        decoderWeightMeanSquare.*decoderStateSquare) ...
        ./single(decoderObservationCount))), ...
    'bias_deviation',single(abs(gather(P.B)-biasReference)), ...
    'adaptation_rms',single(sqrt(gather(adaptationSquare) ...
        ./single(observationCount))), ...
    'bias_reference_mV',double(biasReference), ...
    'test_samples',sampleCount,'timesteps_per_sample',steps, ...
    'observations_per_neuron',observationCount, ...
    'decoder_window_observations_per_neuron',decoderObservationCount, ...
    'definition',struct( ...
        'primary_magnitude',['per-neuron root mean square over held-out samples ', ...
            'and presentation timesteps'], ...
        'net_current','signed afferents are summed before the RMS is taken', ...
        'gross_afferent',['absolute presynaptic contributions are summed before ', ...
            'the RMS is taken; this is a cancellation diagnostic'], ...
        'decoder_presynaptic',['per-neuron RMS contribution to the output, ', ...
            'averaged over output dimensions and the scored decoder window']));
% Preserve the original diagnostic field names for scripts written against
% the initial implementation. Their values now use the standardized RMS
% definition recorded above.
summary.encoder_net=summary.encoder_net_rms;
summary.encoder_gross_afferent=summary.encoder_gross_afferent_rms;
summary.recurrent_net=summary.recurrent_net_rms;
summary.recurrent_gross_afferent=summary.recurrent_gross_afferent_rms;
summary.decoder_presynaptic=summary.decoder_presynaptic_rms;
summary.adaptation=summary.adaptation_rms;
summary.encoder=summary.encoder_net_rms;
summary.recurrent=summary.recurrent_net_rms;
summary.aggregate=struct( ...
    'encoder_rms_mV',population_rms(summary.encoder_net_rms), ...
    'net_recurrent_rms_mV',population_rms(summary.recurrent_net_rms), ...
    'gross_encoder_rms_mV',population_rms(summary.encoder_gross_afferent_rms), ...
    'gross_recurrent_rms_mV',population_rms(summary.recurrent_gross_afferent_rms), ...
    'adaptation_rms_mV',population_rms(summary.adaptation_rms), ...
    'bias_deviation_rms_mV',population_rms(summary.bias_deviation), ...
    'decoder_contribution_rms',population_rms(summary.decoder_presynaptic_rms));
summary.aggregate.recurrent_to_encoder_rms= ...
    summary.aggregate.net_recurrent_rms_mV/max( ...
    summary.aggregate.encoder_rms_mV,realmin);
summary.aggregate.net_to_gross_encoder_rms= ...
    summary.aggregate.encoder_rms_mV/max( ...
    summary.aggregate.gross_encoder_rms_mV,realmin);
summary.aggregate.net_to_gross_recurrent_rms= ...
    summary.aggregate.net_recurrent_rms_mV/max( ...
    summary.aggregate.gross_recurrent_rms_mV,realmin);
end

function summary = dynamics_current_magnitudes(P,cfg,dataInformation,role)
%DYNAMICS_CURRENT_MAGNITUDES Aggregate per-neuron RMS contributions during
% the complete scored closed-loop assessment. The replay uses BANFF_MODEL's
% exposed GPU timestep and exactly mirrors BANFF_EVAL's first-step teacher
% forcing followed by autonomous decoder feedback. Warmup steps establish the
% state but are excluded from all reported magnitudes.
role=lower(string(role));
if ~any(role==["validation","test"])
    error('banff:plotDynamicsCurrentRole','Role must be validation or test.');
end
if ~isa(P.B,'gpuArray'), P=banff_model('gpu',P); end
initialConditions=banff_data('initial_conditions',cfg,role);
system=banff_data('system',cfg.task);
if role=="validation"
    duration=cfg.validation_time;
    warmup=cfg.validation_warmup_time;
else
    duration=cfg.test_time;
    warmup=cfg.test_warmup_time;
end
totalDuration=duration+warmup;
warmupSteps=round(double(warmup)/double(cfg.dt));
transitionCount=round(double(totalDuration)/double(cfg.dt));
recordingSteps=transitionCount-warmupSteps;
if size(initialConditions,2)<1 || recordingSteps<1
    error('banff:plotDynamicsCurrentInput', ...
        'Dynamics current aggregation requires initial conditions and scored steps.');
end

encoderNetSquare=gpuArray.zeros(P.N_hidden,1,'single');
encoderGrossSquare=gpuArray.zeros(P.N_hidden,1,'single');
recurrentNetSquare=gpuArray.zeros(P.N_hidden,1,'single');
recurrentGrossSquare=gpuArray.zeros(P.N_hidden,1,'single');
adaptationSquare=gpuArray.zeros(P.N_hidden,1,'single');
decoderStateSquare=gpuArray.zeros(P.N_hidden,1,'single');

for condition=1:size(initialConditions,2)
    referenceRaw=banff_data('trajectory',system, ...
        initialConditions(:,condition),totalDuration,cfg);
    targetSequence=gpuArray(single((referenceRaw-dataInformation.mean) ...
        ./dataInformation.std));
    state=diagnostic_gpu_state(P,1);
    previousOutput=gpuArray.zeros(P.N_output,1,'single');
    for step=1:transitionCount
        if step==1
            inputSignal=targetSequence(:,1);
        else
            inputSignal=previousOutput;
        end
        inputCurrent=P.W_in*(P.inputScale.*inputSignal);
        recurrentCurrent=replay_recurrent_current(P,state.r);
        if step>warmupSteps
            grossInputCurrent=abs(P.W_in)*(P.inputScale.*abs(inputSignal));
            grossRecurrentCurrent=replay_recurrent_gross(P,state.r);
            encoderNetSquare=encoderNetSquare+inputCurrent.^2;
            encoderGrossSquare=encoderGrossSquare+grossInputCurrent.^2;
            recurrentNetSquare=recurrentNetSquare+recurrentCurrent.^2;
            recurrentGrossSquare=recurrentGrossSquare+grossRecurrentCurrent.^2;
            adaptationSquare=adaptationSquare+state.w.^2;
        end
        state=banff_model('gpu_step',P,state,inputCurrent,false);
        previousOutput=P.W_out*state.r;
        if step>warmupSteps
            decoderStateSquare=decoderStateSquare+state.r.^2;
        end
    end
end

observationCount=double(size(initialConditions,2))*recordingSteps;
biasReference=single(cfg.initial_bias);
decoderWeightMeanSquare=mean(P.W_out.^2,1).';
summary=struct( ...
    'encoder_net_rms',single(sqrt(gather(encoderNetSquare)./single(observationCount))), ...
    'encoder_gross_afferent_rms',single(sqrt(gather(encoderGrossSquare) ...
        ./single(observationCount))), ...
    'recurrent_net_rms',single(sqrt(gather(recurrentNetSquare)./single(observationCount))), ...
    'recurrent_gross_afferent_rms',single(sqrt(gather(recurrentGrossSquare) ...
        ./single(observationCount))), ...
    'decoder_presynaptic_rms',single(sqrt(gather( ...
        decoderWeightMeanSquare.*decoderStateSquare)./single(observationCount))), ...
    'bias_deviation',single(abs(gather(P.B)-biasReference)), ...
    'adaptation_rms',single(sqrt(gather(adaptationSquare)./single(observationCount))), ...
    'bias_reference_mV',double(biasReference), ...
    'test_samples',size(initialConditions,2), ...
    'timesteps_per_sample',recordingSteps, ...
    'observations_per_neuron',observationCount, ...
    'decoder_window_observations_per_neuron',observationCount, ...
    'definition',struct( ...
        'primary_magnitude',['per-neuron root mean square over assessment ', ...
            'initial conditions and scored closed-loop timesteps'], ...
        'net_current','signed afferents are summed before the RMS is taken', ...
        'gross_afferent',['absolute presynaptic contributions are summed before ', ...
            'the RMS is taken; this is a cancellation diagnostic'], ...
        'decoder_presynaptic',['per-neuron RMS contribution to the output, ', ...
            'averaged over output dimensions and scored timesteps']));
summary.encoder_net=summary.encoder_net_rms;
summary.encoder_gross_afferent=summary.encoder_gross_afferent_rms;
summary.recurrent_net=summary.recurrent_net_rms;
summary.recurrent_gross_afferent=summary.recurrent_gross_afferent_rms;
summary.decoder_presynaptic=summary.decoder_presynaptic_rms;
summary.adaptation=summary.adaptation_rms;
summary.encoder=summary.encoder_net_rms;
summary.recurrent=summary.recurrent_net_rms;
summary.aggregate=struct( ...
    'encoder_rms_mV',population_rms(summary.encoder_net_rms), ...
    'net_recurrent_rms_mV',population_rms(summary.recurrent_net_rms), ...
    'gross_encoder_rms_mV',population_rms(summary.encoder_gross_afferent_rms), ...
    'gross_recurrent_rms_mV',population_rms(summary.recurrent_gross_afferent_rms), ...
    'adaptation_rms_mV',population_rms(summary.adaptation_rms), ...
    'bias_deviation_rms_mV',population_rms(summary.bias_deviation), ...
    'decoder_contribution_rms',population_rms(summary.decoder_presynaptic_rms));
summary.aggregate.recurrent_to_encoder_rms= ...
    summary.aggregate.net_recurrent_rms_mV/max( ...
    summary.aggregate.encoder_rms_mV,realmin);
summary.aggregate.net_to_gross_encoder_rms= ...
    summary.aggregate.encoder_rms_mV/max( ...
    summary.aggregate.gross_encoder_rms_mV,realmin);
summary.aggregate.net_to_gross_recurrent_rms= ...
    summary.aggregate.net_recurrent_rms_mV/max( ...
    summary.aggregate.gross_recurrent_rms_mV,realmin);
end

function value = population_rms(perNeuronRms)
% Combining the per-neuron RMS values quadratically recovers the exact RMS
% over the complete neuron-by-observation population.
values=double(perNeuronRms(:));
value=sqrt(mean(values.^2));
end

function state = diagnostic_gpu_state(P,batchSize)
state=struct( ...
    'u',repmat(P.restingVoltage,P.N_hidden,batchSize), ...
    'w',gpuArray.zeros(P.N_hidden,batchSize,'single'), ...
    'x',gpuArray.zeros(P.N_hidden,batchSize,'single'), ...
    'r',gpuArray.zeros(P.N_hidden,batchSize,'single'), ...
    'epsilonVoltage',gpuArray.zeros(P.N_hidden,batchSize,'single'), ...
    'epsilonAdaptation',gpuArray.zeros(P.N_hidden,batchSize,'single'), ...
    'eligibilityRise',gpuArray.zeros(P.N_hidden,batchSize,'single'), ...
    'eligibilityDecay',gpuArray.zeros(P.N_hidden,batchSize,'single'));
end

function current = replay_recurrent_current(P, filteredSpikes)
% Mirror BANFF_MODEL's fixed recurrent-current calculation for diagnostics.
if string(P.recurrent_mode) == "full_rank"
    current = single(P.W_recurrent * double(filteredSpikes));
else
    latent = P.W_feedback * filteredSpikes;
    current = P.recurrentGain .* (P.recurrent_expansion * latent) ...
        - P.self_coupling .* filteredSpikes;
end
end

function gross = replay_recurrent_gross(P,filteredSpikes)
% Sum absolute presynaptic contributions without materialising the N-by-N
% low-rank operator. Dale-signed columns make this factorised expression
% exact; the diagonal term removes the omitted direct self-connection.
if string(P.recurrent_mode)=="full_rank"
    gross=single(abs(P.W_recurrent)*double(filteredSpikes));
else
    gross=P.recurrentGain.*(P.recurrent_expansion* ...
        (abs(P.W_feedback)*filteredSpikes)) ...
        -abs(P.self_coupling).*filteredSpikes;
    gross=max(gross,single(0));
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
%PHASE_DISTANCE Delegate to the publication metric rather than reimplement it.
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
