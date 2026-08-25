%BANFF_TRAIN Training implementation for BANFF.
%   All forward network dynamics and eligibility updates are delegated to
%   BANFF_MODEL. Static and closed-loop validation are shared with testing
%   through BANFF_EVAL.

function result = banff_train(cfg)
fprintf('BANFF: %s, %s, %s, eligibility=%s, N=%d, seed=%d\n', ...
    cfg.task, cfg.method, cfg.recurrent_mode, cfg.eligibility_mode, ...
    cfg.N_hidden, cfg.seed);
if cfg.kind == "dynamics"
    result = train_dynamics(cfg);
else
    result = train_static(cfg);
end
if result.complete
    ensure_directory(fileparts(cfg.model_file));
    save(cfg.model_file, 'result', '-v7.3');
    delete_checkpoint_if_present(checkpoint_file(cfg));
    fprintf('Saved %s\n', cfg.model_file);
end
end

function result = train_static(cfg)
[data, dataInformation] = banff_data('static', cfg);

fprintf('Data: train=%d | validation=%d | test=%d | inputs=%d | outputs=%d\n', ...
    size(data.X_train, 2), ...
    size(data.X_validation, 2), ...
    size(data.X_test, 2), ...
    size(data.X_train, 1), ...
    size(data.Y_train, 1));

P = banff_model('create', ...
    size(data.X_train, 1), size(data.Y_train, 1), cfg);
P = banff_model('gpu', P);
% Static datasets are reused every epoch. Transfer them once rather than
% reconstructing GPU batches from host memory inside each epoch/validation.
data.X_train = gpuArray(data.X_train);
data.Y_train = gpuArray(data.Y_train);
data.X_validation = gpuArray(data.X_validation);
data.Y_validation = gpuArray(data.Y_validation);
% Keep sample ordering/SPSA perturbations controlled across network seeds.
rng(cfg.training_seed, 'twister');

history = initialise_static_history(cfg.epochs);
best = struct('epoch', 0, 'loss', inf, 'metric', -inf, 'B', []);
[P, history, best, startEpoch] = resume_if_available( ...
    P, history, best, checkpoint_file(cfg), cfg);
runTimer = tic;

for epoch = startEpoch:cfg.epochs
    learningRate = schedule_value(epoch, cfg.learning_rate_schedule_epochs, ...
        cfg.learning_rate_start, cfg.learning_rate_end);
    if cfg.method == "spsa"
        [P, trainLoss, trainMetric, spsa] = ...
            static_spsa_epoch(P, data, cfg, epoch, learningRate);
        history.loss_plus(epoch) = spsa.loss_plus;
        history.loss_minus(epoch) = spsa.loss_minus;
        history.perturbation(epoch) = spsa.perturbation;
    else
        [P, trainLoss, trainMetric] = ...
            static_eprop_epoch(P, data, cfg, learningRate);
    end
    history.train_loss(epoch) = trainLoss;
    history.train_metric(epoch) = trainMetric;

    if mod(epoch, cfg.validate_every) == 0
        validation = banff_eval('static', P, data.X_validation, ...
            data.Y_validation, cfg, false);
        history.validation_loss(epoch) = validation.loss;
        history.validation_metric(epoch) = validation.metric;
        if better_static(validation, best, cfg.kind)
            best = struct('epoch', epoch, 'loss', validation.loss, ...
                'metric', validation.metric, 'B', gather(P.B));
        end
    end
    print_progress(epoch, cfg, history, best, ...
    runTimer, startEpoch, learningRate);
    if toc(runTimer) >= cfg.checkpoint_hours * 3600
        save_checkpoint(checkpoint_file(cfg), P, history, best, epoch, cfg);
        result = package_result(cfg, dataInformation, history, best, P, false);
        return;
    end
end
if best.epoch == 0
    best = struct('epoch', cfg.epochs, 'loss', history.train_loss(end), ...
        'metric', history.train_metric(end), 'B', gather(P.B));
end
result = package_result(cfg, dataInformation, history, best, P, true);
end

function [P, lossMean, metric] = static_eprop_epoch(P, data, cfg, learningRate)
sampleCount = size(data.X_train, 2);
order = randperm(sampleCount);
gradient = gpuArray.zeros(P.N_hidden, 1, 'single');
lossSum = gpuArray.zeros(1, 1, 'single');
correct = gpuArray.zeros(1, 1, 'single');
for first = 1:cfg.batch_size:sampleCount
    indices = order(first:min(sampleCount, first + cfg.batch_size - 1));
    X = data.X_train(:, indices);
    Y = data.Y_train(:, indices);
    [output, eligibility] = banff_model('static', P, X, true);
    [batchLoss, outputGradient, batchCorrect] = banff_eval('loss', output, Y, cfg.kind);
    lossSum = lossSum + batchLoss;
    correct = correct + batchCorrect;
    learningSignal = P.W_out.' * outputGradient;
    gradient = gradient + sum(learningSignal .* eligibility, 2);
end
P = banff_model('adam', P, gradient, learningRate, sampleCount, cfg);
lossMean = single(gather(lossSum) / sampleCount);
if cfg.kind == "classification"
    metric = single(100 * gather(correct) / sampleCount);
else
    % Training correlation is not used for optimisation or model selection.
    % Avoid a second complete 300-step pass merely to print this diagnostic.
    metric = single(NaN);
end
end

function [P, lossMean, metric, information] = ...
        static_spsa_epoch(P, data, cfg, epoch, learningRate)
perturbationSize = schedule_value(epoch, cfg.spsa_schedule_epochs, ...
    cfg.spsa_c_start, cfg.spsa_c_end);
direction = single(2 .* (rand(P.N_hidden, 1, 'single') > 0.5) - 1);
direction = gpuArray(direction);
originalBias = P.B;
P.B = originalBias + perturbationSize .* direction;
plus = banff_eval('static', P, data.X_train, data.Y_train, cfg, false);
P.B = originalBias - perturbationSize .* direction;
minus = banff_eval('static', P, data.X_train, data.Y_train, cfg, false);
P.B = originalBias;
gradient = ((plus.loss - minus.loss) / (single(2) * perturbationSize)) ...
    .* direction;
P = banff_model('adam', P, gradient, learningRate, 1, cfg);
training = banff_eval('static', P, data.X_train, data.Y_train, cfg, false);
lossMean = training.loss;
metric = training.metric;
information = struct('loss_plus', plus.loss, 'loss_minus', minus.loss, ...
    'perturbation', perturbationSize);
end

function result = train_dynamics(cfg)
[pool, dataInformation] = banff_data('dynamics', cfg);
dimension = size(pool.states, 1);
P = banff_model('gpu', banff_model('create', dimension, dimension, cfg));
% Keep sampled trajectory windows/SPSA perturbations controlled across seeds.
rng(cfg.training_seed, 'twister');
history = initialise_dynamics_history(cfg.epochs);
best = struct('epoch', 0, 'loss', inf, 'metric', inf, 'B', []);
[P, history, best, startEpoch] = resume_if_available( ...
    P, history, best, checkpoint_file(cfg), cfg);
teacherForcing = scheduled_sampling(pool.window_samples, cfg);
runTimer = tic;

for epoch = startEpoch:cfg.epochs
    start = randi(pool.max_start);
    sequence = pool.states(:, start:(start + pool.window_samples - 1));
    learningRate = schedule_value(epoch, cfg.learning_rate_schedule_epochs, ...
        cfg.learning_rate_start, cfg.learning_rate_end);
    if cfg.method == "spsa"
        [P, loss, spsa] = dynamics_spsa_epoch( ...
            P, sequence, teacherForcing, cfg, epoch, learningRate);
        history.loss_plus(epoch) = spsa.loss_plus;
        history.loss_minus(epoch) = spsa.loss_minus;
        history.perturbation(epoch) = spsa.perturbation;
    else
        [lossGpu, gradient] = banff_model('dynamics', ...
            P, sequence, teacherForcing, true, false);
        transitionCount = size(sequence, 2) - 1;
        P = banff_model('adam', P, gradient, learningRate, transitionCount, cfg);
        loss = single(gather(lossGpu) / transitionCount);
    end
    history.train_loss(epoch) = loss;

    if mod(epoch, cfg.validate_dynamics_every) == 0
        validation = banff_eval('closed_loop', P, cfg, dataInformation, "validation", false);
        history.validation_distance(epoch) = validation.phase_distance;
        if validation.phase_distance < best.metric
            best = struct('epoch', epoch, 'loss', loss, ...
                'metric', validation.phase_distance, 'B', gather(P.B));
        end
    end
    print_progress(epoch, cfg, history, best, ...
    runTimer, startEpoch, learningRate);
    if toc(runTimer) >= cfg.checkpoint_hours * 3600
        save_checkpoint(checkpoint_file(cfg), P, history, best, epoch, cfg);
        result = package_result(cfg, dataInformation, history, best, P, false);
        return;
    end
end
if best.epoch == 0
    best = struct('epoch', cfg.epochs, 'loss', history.train_loss(end), ...
        'metric', inf, 'B', gather(P.B));
end
result = package_result(cfg, dataInformation, history, best, P, true);
end

function [P, lossMean, information] = dynamics_spsa_epoch( ...
        P, sequence, teacherForcing, cfg, epoch, learningRate)
perturbationSize = schedule_value(epoch, cfg.spsa_schedule_epochs, ...
    cfg.spsa_c_start, cfg.spsa_c_end);
direction = gpuArray(single(2 .* (rand(P.N_hidden, 1, 'single') > 0.5) - 1));
originalBias = P.B;
P.B = originalBias + perturbationSize .* direction;
plus = banff_model('dynamics', P, sequence, teacherForcing, false, false);
P.B = originalBias - perturbationSize .* direction;
minus = banff_model('dynamics', P, sequence, teacherForcing, false, false);
P.B = originalBias;
count = size(sequence, 2) - 1;
plus = single(gather(plus) / count);
minus = single(gather(minus) / count);
gradient = ((plus - minus) / (single(2) * perturbationSize)) .* direction;
P = banff_model('adam', P, gradient, learningRate, 1, cfg);
updatedLoss = banff_model('dynamics', P, sequence, teacherForcing, false, false);
lossMean = single(gather(updatedLoss) / count);
information = struct('loss_plus', plus, 'loss_minus', minus, ...
    'perturbation', perturbationSize);
end

function result = package_result(cfg, dataInformation, history, best, P, complete)
result = struct();
result.version = 'BANFF-SNN publication-ready hard-event v3';
result.complete = complete;
result.config = cfg;
result.data_information = dataInformation;
result.history = history;
result.best = best;
result.final_trainable_state = banff_model('gather', P);
result.training = eligibility_metadata(cfg);
result.provenance = runtime_provenance(cfg);
end

function history = initialise_static_history(epochs)
history = struct();
history.train_loss = nan(epochs, 1, 'single');
history.train_metric = nan(epochs, 1, 'single');
history.validation_loss = nan(epochs, 1, 'single');
history.validation_metric = nan(epochs, 1, 'single');
history.loss_plus = nan(epochs, 1, 'single');
history.loss_minus = nan(epochs, 1, 'single');
history.perturbation = nan(epochs, 1, 'single');
end

function history = initialise_dynamics_history(epochs)
history = struct();
history.train_loss = nan(epochs, 1, 'single');
history.validation_distance = nan(epochs, 1, 'single');
history.loss_plus = nan(epochs, 1, 'single');
history.loss_minus = nan(epochs, 1, 'single');
history.perturbation = nan(epochs, 1, 'single');
end

function [P, history, best, startEpoch] = ...
        resume_if_available(P, history, best, file, cfg)
startEpoch = 1;
if exist(file, 'file') ~= 2
    return;
end
loaded = load(file, 'checkpoint');
checkpoint = loaded.checkpoint;
if ~isfield(checkpoint.config,'checkpoint_config_sha256') || ...
        ~strcmp(checkpoint.config.checkpoint_config_sha256, cfg.checkpoint_config_sha256)
    error('banff:checkpointMismatch', ...
        'Checkpoint scientific settings do not match the requested experiment.');
end
currentSource = core_source_hashes();
if ~isfield(checkpoint, 'core_source_sha256') || ...
        ~isequal(checkpoint.core_source_sha256, currentSource)
    error('banff:checkpointSourceMismatch', ...
        ['Checkpoint was created by different core source code. Delete the checkpoint ', ...
         'or resume with the exact source version that created it.']);
end
P.B = gpuArray(single(checkpoint.state.B));
P.m = gpuArray(single(checkpoint.state.m));
P.v = gpuArray(single(checkpoint.state.v));
P.vMax = gpuArray(single(checkpoint.state.vMax));
P.adamStep = checkpoint.state.adamStep;
history = checkpoint.history;
best = checkpoint.best;
rng(checkpoint.random_state);
startEpoch = checkpoint.epoch + 1;
fprintf('Resuming at epoch %d.\n', startEpoch);
end

function save_checkpoint(file, P, history, best, epoch, cfg)
ensure_directory(fileparts(file));
checkpoint = struct('epoch', epoch, 'state', banff_model('gather', P), ...
    'history', history, 'best', best, 'config', cfg, 'random_state', rng, ...
    'core_source_sha256', core_source_hashes());
save(file, 'checkpoint', '-v7.3');
end

function file = checkpoint_file(cfg)
[folder, name] = fileparts(cfg.model_file);
file = fullfile(folder, [name '_checkpoint.mat']);
end

function delete_checkpoint_if_present(file)
if exist(file, 'file') == 2
    delete(file);
end
end

function sequence = scheduled_sampling(samples, cfg)
sequence = true(1, samples);
period = cfg.teacher_steps + cfg.closed_loop_steps;
indices = 2:samples;
sequence(indices(mod(indices - 1, period) >= cfg.teacher_steps)) = false;
end

function value = schedule_value(epoch, horizon, first, last)
fraction = (double(epoch) - 1) / max(1, double(horizon) - 1);
fraction = min(max(fraction, 0), 1);
value = single(first * (last / first)^fraction);
end

function yes = better_static(candidate, best, kind)
if kind == "classification"
    yes = candidate.metric > best.metric || ...
        (candidate.metric == best.metric && candidate.loss < best.loss);
else
    yes = candidate.loss < best.loss;
end
end

function print_progress(epoch, cfg, history, best, ...
        runTimer, startEpoch, learningRate)

shouldPrint = epoch == startEpoch || ...
    mod(epoch, cfg.verbose_every) == 0 || ...
    epoch == cfg.epochs;

if ~shouldPrint
    return;
end

% Timing information for the current training run.
elapsedSeconds = toc(runTimer);
epochsCompleted = max(1, epoch - startEpoch + 1);
secondsPerEpoch = elapsedSeconds / epochsCompleted;
epochsRemaining = max(0, cfg.epochs - epoch);
etaSeconds = secondsPerEpoch * epochsRemaining;
percentComplete = 100 * double(epoch) / double(cfg.epochs);

elapsedText = format_duration(elapsedSeconds);
etaText = format_duration(etaSeconds);

% -------------------------------------------------------------------------
% Dynamical-system tasks
% -------------------------------------------------------------------------
if cfg.kind == "dynamics"

    validationEpoch = find( ...
        isfinite(history.validation_distance(1:epoch)), ...
        1, 'last');

    if isempty(validationEpoch)
        fprintf([ ...
            'epoch %d/%d (%5.1f%%) | ' ...
            'train loss %.6g | ' ...
            'val n/a | best n/a | ' ...
            'lr %.3e | %.2f s/epoch | ' ...
            'elapsed %s | ETA %s\n'], ...
            epoch, cfg.epochs, percentComplete, ...
            history.train_loss(epoch), ...
            learningRate, secondsPerEpoch, ...
            elapsedText, etaText);
    else
        fprintf([ ...
            'epoch %d/%d (%5.1f%%) | ' ...
            'train loss %.6g | ' ...
            'val@%d phase-dist %.6g | ' ...
            'best phase-dist %.6g @%d | ' ...
            'lr %.3e | %.2f s/epoch | ' ...
            'elapsed %s | ETA %s\n'], ...
            epoch, cfg.epochs, percentComplete, ...
            history.train_loss(epoch), ...
            validationEpoch, ...
            history.validation_distance(validationEpoch), ...
            best.metric, best.epoch, ...
            learningRate, secondsPerEpoch, ...
            elapsedText, etaText);
    end

    return;
end

% -------------------------------------------------------------------------
% Static classification/regression tasks
% -------------------------------------------------------------------------
validationEpoch = find( ...
    isfinite(history.validation_loss(1:epoch)), ...
    1, 'last');

if isempty(validationEpoch)

    if cfg.kind == "classification" && ...
            isfinite(history.train_metric(epoch))

        fprintf([ ...
            'epoch %d/%d (%5.1f%%) | ' ...
            'train loss %.6g acc %.2f%% | ' ...
            'val n/a | best n/a | ' ...
            'lr %.3e | %.2f s/epoch | ' ...
            'elapsed %s | ETA %s\n'], ...
            epoch, cfg.epochs, percentComplete, ...
            history.train_loss(epoch), ...
            history.train_metric(epoch), ...
            learningRate, secondsPerEpoch, ...
            elapsedText, etaText);

    else

        fprintf([ ...
            'epoch %d/%d (%5.1f%%) | ' ...
            'train loss %.6g | ' ...
            'val n/a | best n/a | ' ...
            'lr %.3e | %.2f s/epoch | ' ...
            'elapsed %s | ETA %s\n'], ...
            epoch, cfg.epochs, percentComplete, ...
            history.train_loss(epoch), ...
            learningRate, secondsPerEpoch, ...
            elapsedText, etaText);
    end

    return;
end

% Classification: metric is percentage accuracy.
if cfg.kind == "classification"

    fprintf([ ...
        'epoch %d/%d (%5.1f%%) | ' ...
        'train loss %.6g acc %.2f%% | ' ...
        'val@%d loss %.6g acc %.2f%% | ' ...
        'best acc %.2f%% loss %.6g @%d | ' ...
        'lr %.3e | %.2f s/epoch | ' ...
        'elapsed %s | ETA %s\n'], ...
        epoch, cfg.epochs, percentComplete, ...
        history.train_loss(epoch), ...
        history.train_metric(epoch), ...
        validationEpoch, ...
        history.validation_loss(validationEpoch), ...
        history.validation_metric(validationEpoch), ...
        best.metric, best.loss, best.epoch, ...
        learningRate, secondsPerEpoch, ...
        elapsedText, etaText);

% Regression: metric is Pearson correlation.
else

    fprintf([ ...
        'epoch %d/%d (%5.1f%%) | ' ...
        'train loss %.6g | ' ...
        'val@%d loss %.6g corr %.4f | ' ...
        'best loss %.6g corr %.4f @%d | ' ...
        'lr %.3e | %.2f s/epoch | ' ...
        'elapsed %s | ETA %s\n'], ...
        epoch, cfg.epochs, percentComplete, ...
        history.train_loss(epoch), ...
        validationEpoch, ...
        history.validation_loss(validationEpoch), ...
        history.validation_metric(validationEpoch), ...
        best.loss, best.metric, best.epoch, ...
        learningRate, secondsPerEpoch, ...
        elapsedText, etaText);
end
end

function text = format_duration(secondsValue)
if ~isfinite(secondsValue) || secondsValue < 0
    text = '--:--:--';
    return;
end

secondsValue = round(double(secondsValue));

hours = floor(secondsValue / 3600);
minutes = floor(mod(secondsValue, 3600) / 60);
seconds = mod(secondsValue, 60);

text = sprintf('%02d:%02d:%02d', hours, minutes, seconds);
end

function ensure_directory(folder)
if exist(folder, 'dir') ~= 7
    mkdir(folder);
end
end

function metadata = eligibility_metadata(cfg)
metadata = struct();
metadata.trainable_parameter = 'hidden bias B only';
metadata.fixed_parameters = 'input, recurrent and decoder weights';
metadata.backend = 'MATLAB gpuArray with fused arrayfun kernel';
metadata.method = char(cfg.method);
metadata.optimizer = char(cfg.optimizer);
metadata.optimizer_definition = [ ...
    'bias-corrected AMSGrad; running maximum of raw second moment, ' ...
    'followed by current-step second-moment bias correction'];
metadata.eligibility_mode = char(cfg.eligibility_mode);
metadata.reset_derivative = 'stop_gradient';
metadata.rho_derivative = 'stop_gradient';
metadata.readout_eligibility_filter = 'two-stage rise-decay matched to decoder state';
if cfg.method ~= "eprop"
    metadata.eligibility_rule = 'not used by SPSA';
    return;
end
if cfg.eligibility_mode == "hard_spike"
    metadata.eligibility_rule = ['event-gated full local-state bias sensitivity; ' ...
        'event magnitude evaluated at the LSTI event time'];
    metadata.hard_event_gain_per_mV = double(cfg.hard_event_gain);
else
    metadata.eligibility_rule = ['continuous triangular pseudo-derivative with ' ...
        'full local-state bias sensitivity'];
    metadata.surrogate_peak_per_mV = double(cfg.surrogate_peak);
    metadata.surrogate_half_width_mV = double(cfg.surrogate_half_width);
end
end

function provenance = runtime_provenance(cfg)
provenance = struct();
provenance.release = 'BANFF-SNN publication-ready hard-event v3';
provenance.scientific_config_sha256 = cfg.scientific_config_sha256;
provenance.checkpoint_config_sha256 = cfg.checkpoint_config_sha256;
provenance.matlab_version = version;
products = ver;
provenance.matlab_products = struct('name', {products.Name}, ...
    'version', {products.Version}, 'release', {products.Release});
provenance.gpu = struct();
try
    device = gpuDevice;
    provenance.gpu.name = device.Name;
    provenance.gpu.compute_capability = device.ComputeCapability;
    provenance.gpu.total_memory_bytes = double(device.TotalMemory);
    provenance.gpu.available_memory_bytes = double(device.AvailableMemory);
    if isprop(device, 'DriverVersion')
        provenance.gpu.driver_version = device.DriverVersion;
    end
catch ME
    provenance.gpu.error = ME.message;
end
provenance.core_source_sha256 = core_source_hashes();
end

function sourceHashes = core_source_hashes()
% Hash only the small scientific core. Figure scripts and documentation may
% change without invalidating a trained model or checkpoint.
root = fileparts(mfilename('fullpath'));
files = {'banff.m','banff_train.m','banff_test.m','banff_eval.m', ...
    'banff_model.m','banff_data.m','banff_metrics.m'};
sourceHashes = struct();
for index = 1:numel(files)
    field = matlab.lang.makeValidName(files{index});
    sourceHashes.(field) = file_sha256_local(fullfile(root, files{index}));
end
end

function hash = file_sha256_local(file)
hash = '';
if exist(file, 'file') ~= 2
    return;
end
engine = javaMethod('getInstance', 'java.security.MessageDigest', 'SHA-256');
fileId = fopen(file, 'r');
if fileId < 0
    return;
end
cleanup = onCleanup(@() fclose(fileId)); %#ok<NASGU>
while true
    bytes = fread(fileId, 1024 * 1024, '*uint8');
    if isempty(bytes), break; end
    engine.update(bytes);
end
digest = typecast(engine.digest(), 'uint8');
hash = lower(reshape(dec2hex(digest).', 1, []));
end
