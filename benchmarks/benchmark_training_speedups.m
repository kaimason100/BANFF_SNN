%% BANFF MATLAB-only training performance benchmark
% This script DOES NOT train, resume, save, or alter a scientific model.
% It measures proposed optimisations on the current GPU and reports both speed
% and numerical agreement. Rows are labelled as one of:
%   current path       - calls the current BANFF_MODEL implementation;
%   exact-kernel proxy - uses BANFF_MODEL's exact neuron kernel in a proxy task;
%   isolated exact     - benchmarks an algebraically exact sub-operation;
%   paged prototype    - measures a multi-seed layout with the same low-rank
%                        matrix structure but a simplified state equation.
%
% Run from the repository root or from this file in the MATLAB editor.
% Timings are synchronised GPU wall times and reported as medians. Component
% speedups are not additive and must not be interpreted as epoch-level gains.
% Numerical comparisons always use identical inputs and parameter states.
% Because hard thresholds can amplify round-off into different event histories,
% the detailed report separates primitive arithmetic differences from output,
% eligibility and cumulative-spike divergence over increasing horizons.

clearvars;
close all;
clc;

%% User settings
repository_root = fileparts(fileparts(mfilename('fullpath')));
addpath(repository_root);

settings = struct();
settings.N_hidden = 32000;
settings.N_recurrent = 10;
settings.static_inputs = 30;
settings.static_outputs = 2;
settings.static_samples = 384;
settings.static_batch_sizes = [32 64 128 256 512 1024 2048 4096];
settings.static_large_batch_sizes = [32 64 128 256 512 1024 2048 4096];
settings.maximum_large_batch_memory_fraction = 0.45;
settings.static_steps = 600;
settings.static_average_fraction = 0.5;
settings.dynamics_dimension = 3;
settings.dynamics_benchmark_steps = 600;
settings.dynamics_epochs_per_timing = 2;
settings.validation_initial_conditions = 3;
settings.seed_count = 3;
settings.prototype_steps = 600;
settings.teacher_block_steps = 30;
settings.timing_repetitions = 3;
settings.agreement_probe_batch_size = 256;
settings.agreement_horizons = [1 2 5 10 20 50 100 200 400 600];
settings.agreement_material_tolerance = 1e-5;
settings.save_results = false;
settings.output_directory = fullfile(repository_root, 'outputs', 'benchmarks');

assert(canUseGPU, 'A supported NVIDIA GPU is required.');
device = gpuDevice;
rng(1701, 'twister');

fprintf('\nBANFF MATLAB-only performance benchmark\n');
fprintf('GPU: %s | compute capability %s | %.1f GiB total memory\n', ...
    device.Name, device.ComputeCapability, double(device.TotalMemory) / 2^30);
fprintf('MATLAB: %s\n', version);
fprintf('N_hidden=%d | timing repetitions=%d\n\n', ...
    settings.N_hidden, settings.timing_repetitions);

rows = struct('Effect', {}, 'Scope', {}, 'BaselineSeconds', {}, ...
    'CandidateSeconds', {}, 'Speedup', {}, 'MaxRelativeDifference', {}, ...
    'EstimatedMemoryMiB', {}, 'Notes', {});
agreementRows = struct('Case', {}, 'Quantity', {}, 'BatchSize', {}, ...
    'Horizon', {}, 'MaxAbsoluteDifference', {}, 'RMSEDifference', {}, ...
    'RelativeL2Difference', {}, 'ScaleNormalizedMaxDifference', {}, ...
    'ExactMismatchCount', {}, 'ElementCount', {}, ...
    'ExactMismatchFraction', {}, 'ReferenceNorm2', {}, ...
    'CandidateNorm2', {}, 'NonfinitePairCount', {});

%% Construct one current low-rank static model and synthetic task
staticOverrides = struct('N_hidden', settings.N_hidden, ...
    'N_recurrent', settings.N_recurrent, 'batch_size', 32, ...
    'presentation_time', single(settings.static_steps * 1e-3), ...
    'average_fraction', single(settings.static_average_fraction), ...
    'epochs', 1, 'validate_every', 1, 'seed', 701);
staticCfg = banff("config", "breast_cancer", staticOverrides);
Pstatic = banff_model('gpu', banff_model('create', ...
    settings.static_inputs, settings.static_outputs, staticCfg));

Xhost = randn(settings.static_inputs, settings.static_samples, 'single');
labels = randi(settings.static_outputs, 1, settings.static_samples);
Yhost = zeros(settings.static_outputs, settings.static_samples, 'single');
Yhost(sub2ind(size(Yhost), labels, 1:settings.static_samples)) = 1;
Xgpu = gpuArray(Xhost);
Ygpu = gpuArray(Yhost);

% Warm up GPU ARRAYFUN/JIT before collecting timings.
static_epoch_core(Pstatic, Xgpu(:, 1:32), Ygpu(:, 1:32), staticCfg, 32);
wait(device);

%% 1. Static batch-size scaling on GPU-resident data
[baselineGradient, baselineLoss] = static_epoch_core( ...
    Pstatic, Xgpu, Ygpu, staticCfg, 32);
baselineStaticTime = gpu_wall_time(@() static_epoch_core( ...
    Pstatic, Xgpu, Ygpu, staticCfg, 32), settings.timing_repetitions, device);

for batchSize = settings.static_batch_sizes
    if batchSize > settings.static_samples
        continue;
    end
    [candidateGradient, candidateLoss] = static_epoch_core( ...
        Pstatic, Xgpu, Ygpu, staticCfg, batchSize);
    candidateTime = gpu_wall_time(@() static_epoch_core( ...
        Pstatic, Xgpu, Ygpu, staticCfg, batchSize), ...
        settings.timing_repetitions, device);
    difference = max(relative_error(candidateGradient, baselineGradient), ...
        relative_error(candidateLoss, baselineLoss));
    agreementRows(end+1) = agreement_row( ... %#ok<SAGROW>
        "Static epoch gradient", "bias gradient", batchSize, ...
        settings.static_steps, baselineGradient, candidateGradient);
    agreementRows(end+1) = agreement_row( ... %#ok<SAGROW>
        "Static epoch loss", "mean loss", batchSize, ...
        settings.static_steps, baselineLoss, candidateLoss);
    memoryMiB = estimate_static_state_mib(settings.N_hidden, batchSize);
    rows(end+1) = benchmark_row( ... %#ok<SAGROW>
        "Static batch size " + batchSize, "current path", ...
        baselineStaticTime, candidateTime, difference, memoryMiB, ...
        "Same full-dataset gradient; only reduction order changes.");
end

%% 1b. Large-batch current-path throughput sweep (up to thousands)
% A full 8,192-sample baseline split into batches of 32 would make this
% diagnostic unnecessarily long. Measure one real current-path call at each
% size and compare it with the measured time for enough batch-32 calls to
% process the same number of samples.
smallX = Xgpu(:, 1:32);
smallY = Ygpu(:, 1:32);
smallBatchTime = gpu_wall_time(@() static_epoch_core( ...
    Pstatic, smallX, smallY, staticCfg, 32), ...
    settings.timing_repetitions, device);
[smallOutput, smallEligibility] = banff_model( ...
    'static', Pstatic, smallX, true, false);
largeBatchSizeResult = 32;
largeBatchSeconds = smallBatchTime;
largeBatchSamplesPerSecond = 32 / smallBatchTime;
largeBatchProjectedSpeedup = 1;
largeBatchEstimatedMiB = estimate_static_state_mib(settings.N_hidden, 32);

for batchSize = settings.static_large_batch_sizes
    estimatedMiB = estimate_static_state_mib(settings.N_hidden, batchSize);
    availableMiB = double(device.AvailableMemory) / 2^20;
    if estimatedMiB > settings.maximum_large_batch_memory_fraction * availableMiB
        fprintf(['Skipping batch %d: conservative state estimate %.0f MiB ', ...
            'exceeds %.0f%% of currently available GPU memory (%.0f MiB).\n'], ...
            batchSize, estimatedMiB, ...
            100 * settings.maximum_large_batch_memory_fraction, availableMiB);
        continue;
    end
    try
        largeX = gpuArray.randn(settings.static_inputs, batchSize, 'single');
        largeLabels = randi(settings.static_outputs, 1, batchSize);
        largeYHost = zeros(settings.static_outputs, batchSize, 'single');
        largeYHost(sub2ind(size(largeYHost), largeLabels, 1:batchSize)) = 1;
        largeY = gpuArray(largeYHost);
        candidateTime = gpu_wall_time(@() static_epoch_core( ...
            Pstatic, largeX, largeY, staticCfg, batchSize), ...
            settings.timing_repetitions, device);

        % Confirm that samples embedded in the large batch remain independent.
        [largeOutput, largeEligibility, largeSpikeCount] = banff_model( ...
            'static', Pstatic, largeX, true, true);
        [prefixOutput, prefixEligibility, prefixSpikeCount] = banff_model( ...
            'static', Pstatic, largeX(:, 1:32), true, true);
        independenceDifference = max( ...
            relative_error(largeOutput(:, 1:32), prefixOutput), ...
            relative_error(largeEligibility(:, 1:32), prefixEligibility));
        independenceDifference = max(independenceDifference, ...
            relative_error(largeSpikeCount(:, 1:32), prefixSpikeCount));

        agreementRows(end+1) = agreement_row( ... %#ok<SAGROW>
            "Static prefix independence", "average output", batchSize, ...
            settings.static_steps, prefixOutput, largeOutput(:, 1:32));
        agreementRows(end+1) = agreement_row( ... %#ok<SAGROW>
            "Static prefix independence", "average eligibility", batchSize, ...
            settings.static_steps, prefixEligibility, largeEligibility(:, 1:32));
        agreementRows(end+1) = agreement_row( ... %#ok<SAGROW>
            "Static prefix independence", "cumulative spike count", batchSize, ...
            settings.static_steps, prefixSpikeCount, largeSpikeCount(:, 1:32));

        projectedBaseline = smallBatchTime * ceil(batchSize / 32);
        largeBatchSizeResult(end+1,1) = batchSize; %#ok<SAGROW>
        largeBatchSeconds(end+1,1) = candidateTime; %#ok<SAGROW>
        largeBatchSamplesPerSecond(end+1,1) = batchSize / candidateTime; %#ok<SAGROW>
        largeBatchProjectedSpeedup(end+1,1) = projectedBaseline / candidateTime; %#ok<SAGROW>
        largeBatchEstimatedMiB(end+1,1) = estimatedMiB; %#ok<SAGROW>
        rows(end+1) = benchmark_row( ... %#ok<SAGROW>
            "Static large batch " + batchSize, "current-path throughput", ...
            projectedBaseline, candidateTime, independenceDifference, ...
            estimatedMiB, ...
            "Baseline is measured batch-32 time multiplied by the required call count; candidate is one real BANFF call.");
        clear largeX largeY largeYHost largeOutput largeEligibility ...
            largeSpikeCount prefixOutput prefixEligibility prefixSpikeCount;
        wait(device);
    catch exception
        fprintf(2, 'Skipping batch %d after GPU allocation/execution failure: %s\n', ...
            batchSize, exception.message);
        clear largeX largeY largeYHost largeOutput largeEligibility ...
            largeSpikeCount prefixOutput prefixEligibility prefixSpikeCount;
        wait(device);
    end
end
clear smallOutput smallEligibility;
large_batch_scaling = table(largeBatchSizeResult(:), largeBatchSeconds(:), ...
    largeBatchSamplesPerSecond(:), largeBatchProjectedSpeedup(:), ...
    largeBatchEstimatedMiB(:), 'VariableNames', ...
    {'BatchSize','MeasuredSeconds','SamplesPerSecond', ...
    'SpeedupVersusBatch32Projection','EstimatedStateMiB'});

%% 1c. Locate the onset of batch-dependent numerical divergence
% Re-run one fixed prefix both alone and embedded in a larger batch.  Sweeping
% the horizon distinguishes immediate GEMM/layout differences from later
% amplification caused by hard spike decisions and recurrent feedback.
probeBatchSize = settings.agreement_probe_batch_size;
while estimate_static_state_mib(settings.N_hidden, probeBatchSize) > ...
        settings.maximum_large_batch_memory_fraction * double(device.AvailableMemory) / 2^20
    probeBatchSize = floor(probeBatchSize / 2);
end
assert(probeBatchSize >= 32, ...
    'Insufficient GPU memory for the minimum agreement probe batch.');
probeX = gpuArray.randn(settings.static_inputs, probeBatchSize, 'single');
embeddedInputCurrent = Pstatic.W_in * (Pstatic.inputScale .* probeX);
aloneInputCurrent = Pstatic.W_in * (Pstatic.inputScale .* probeX(:, 1:32));
agreementRows(end+1) = agreement_row( ... %#ok<SAGROW>
    "Static input-current layout", "encoded input current", probeBatchSize, ...
    0, aloneInputCurrent, embeddedInputCurrent(:, 1:32));
probeHorizons = unique([settings.agreement_horizons settings.static_steps]);
probeHorizons = probeHorizons(probeHorizons >= 1 & ...
    probeHorizons <= settings.static_steps);
for horizon = probeHorizons
    probeModel = Pstatic;
    probeModel.presentationSteps = double(horizon);
    probeModel.averageStartStep = 1;
    probeModel.averageSteps = double(horizon);
    [embeddedOutput, embeddedEligibility, embeddedSpikeCount] = banff_model( ...
        'static', probeModel, probeX, true, true);
    [aloneOutput, aloneEligibility, aloneSpikeCount] = banff_model( ...
        'static', probeModel, probeX(:, 1:32), true, true);
    agreementRows(end+1) = agreement_row( ... %#ok<SAGROW>
        "Static divergence horizon", "average output", probeBatchSize, ...
        horizon, aloneOutput, embeddedOutput(:, 1:32));
    agreementRows(end+1) = agreement_row( ... %#ok<SAGROW>
        "Static divergence horizon", "average eligibility", probeBatchSize, ...
        horizon, aloneEligibility, embeddedEligibility(:, 1:32));
    agreementRows(end+1) = agreement_row( ... %#ok<SAGROW>
        "Static divergence horizon", "cumulative spike count", probeBatchSize, ...
        horizon, aloneSpikeCount, embeddedSpikeCount(:, 1:32));
end
clear probeX embeddedInputCurrent aloneInputCurrent probeModel ...
    embeddedOutput embeddedEligibility ...
    embeddedSpikeCount aloneOutput aloneEligibility aloneSpikeCount;

%% 2. Repeated host-to-GPU batch transfers versus resident training arrays
hostTransferTime = gpu_wall_time(@() static_epoch_core( ...
    Pstatic, Xhost, Yhost, staticCfg, 32), settings.timing_repetitions, device);
residentTime = baselineStaticTime;
[residentGradient, residentLoss] = static_epoch_core( ...
    Pstatic, Xgpu, Ygpu, staticCfg, 32);
[hostGradient, hostLoss] = static_epoch_core( ...
    Pstatic, Xhost, Yhost, staticCfg, 32);
transferDifference = max(relative_error(residentGradient, hostGradient), ...
    relative_error(residentLoss, hostLoss));
rows(end+1) = benchmark_row( ... %#ok<SAGROW>
    "Keep static data on GPU", "current path", hostTransferTime, ...
    residentTime, transferDifference, bytes_to_mib(numel(Xhost)+numel(Yhost)), ...
    "Eliminates per-batch gpuArray construction and PCIe transfers.");

%% 3. Algebraically commute the linear decoder with temporal averaging
decoderBatch = min(64, settings.static_samples);
decoderSteps = round(settings.static_steps * settings.static_average_fraction);
decoderState = gpuArray.randn(settings.N_hidden, decoderBatch, 'single');
decoderWeight = Pstatic.W_out;
decoder_repeated(decoderWeight, decoderState, decoderSteps);
decoder_once(decoderWeight, decoderState, decoderSteps);
repeatedDecoderTime = gpu_wall_time(@() decoder_repeated( ...
    decoderWeight, decoderState, decoderSteps), settings.timing_repetitions, device);
singleDecoderTime = gpu_wall_time(@() decoder_once( ...
    decoderWeight, decoderState, decoderSteps), settings.timing_repetitions, device);
repeatedDecoder = decoder_repeated(decoderWeight, decoderState, decoderSteps);
singleDecoder = decoder_once(decoderWeight, decoderState, decoderSteps);
rows(end+1) = benchmark_row( ... %#ok<SAGROW>
    "Decode after temporal averaging", "isolated exact", ...
    repeatedDecoderTime, singleDecoderTime, ...
    relative_error(repeatedDecoder, singleDecoder), ...
    bytes_to_mib(numel(decoderState)), ...
    "Measures the decoder section only; full-epoch gain is smaller.");

%% 4. Exact-kernel proxy for batching independent validation trajectories
validationInputs = Xgpu(:, 1:settings.validation_initial_conditions);
sequentialValidation = validation_proxy_sequential(Pstatic, validationInputs);
batchedValidation = validation_proxy_batched(Pstatic, validationInputs);
sequentialValidationTime = gpu_wall_time(@() validation_proxy_sequential( ...
    Pstatic, validationInputs), settings.timing_repetitions, device);
batchedValidationTime = gpu_wall_time(@() validation_proxy_batched( ...
    Pstatic, validationInputs), settings.timing_repetitions, device);
rows(end+1) = benchmark_row( ... %#ok<SAGROW>
    "Batch " + settings.validation_initial_conditions + " validation trajectories", ...
    "exact-kernel proxy", ...
    sequentialValidationTime, batchedValidationTime, ...
    relative_error(sequentialValidation, batchedValidation), ...
    estimate_static_state_mib(settings.N_hidden, settings.validation_initial_conditions), ...
    "Exact BANFF neuron/recurrent kernel with independent constant inputs; closed-loop plumbing is not included.");

%% Construct one current dynamics model and sequence
dynamicsOverrides = struct('N_hidden', settings.N_hidden, ...
    'N_recurrent', settings.N_recurrent, 'epochs', 2, 'seed', 811);
dynamicsCfg = banff("config", "lorenz", dynamicsOverrides);
Pdynamics = banff_model('gpu', banff_model('create', ...
    settings.dynamics_dimension, settings.dynamics_dimension, dynamicsCfg));
sequence = gpuArray.randn(settings.dynamics_dimension, ...
    settings.dynamics_benchmark_steps + 1, 'single');
teacherForcing = scheduled_sampling_local( ...
    settings.dynamics_benchmark_steps + 1, dynamicsCfg);
banff_model('dynamics', Pdynamics, sequence, teacherForcing, true, false);
wait(device);

%% 5. Current dynamics path and epoch-time extrapolation
currentDynamicsTime = gpu_wall_time(@() banff_model('dynamics', ...
    Pdynamics, sequence, teacherForcing, true, false), ...
    settings.timing_repetitions, device);
secondsPerStep = currentDynamicsTime / settings.dynamics_benchmark_steps;
dynamics_projection = table( ...
    ["Lorenz"; "Sprott-S"; "Van der Pol"; "One 5-IC validation"], ...
    [20000; 20000; 5000; 250000], ...
    secondsPerStep .* [20000; 20000; 5000; 250000], ...
    'VariableNames', {'Workload','TransitionCount','ProjectedSeconds'});

rows(end+1) = benchmark_row( ... %#ok<SAGROW>
    "Dynamics current-path reference", "current path", ...
    currentDynamicsTime, currentDynamicsTime, 0, ...
    estimate_dynamics_state_mib(settings.N_hidden, 1), ...
    "Reference timing used for the workload projection table.");

%% 6. Gather every epoch versus buffered scalar loss gathering
epochCount = settings.dynamics_epochs_per_timing;
[gatherBias, gatherLoss] = dynamics_epoch_group( ...
    Pdynamics, sequence, teacherForcing, dynamicsCfg, epochCount, true);
[bufferBias, bufferLoss] = dynamics_epoch_group( ...
    Pdynamics, sequence, teacherForcing, dynamicsCfg, epochCount, false);
gatherEachTime = gpu_wall_time(@() dynamics_epoch_group( ...
    Pdynamics, sequence, teacherForcing, dynamicsCfg, epochCount, true), ...
    settings.timing_repetitions, device);
bufferedTime = gpu_wall_time(@() dynamics_epoch_group( ...
    Pdynamics, sequence, teacherForcing, dynamicsCfg, epochCount, false), ...
    settings.timing_repetitions, device);
gatherDifference = max(relative_error(gatherBias, bufferBias), ...
    relative_error(gatherLoss, bufferLoss));
rows(end+1) = benchmark_row( ... %#ok<SAGROW>
    "Buffer per-epoch scalar gathers", "current path", ...
    gatherEachTime, bufferedTime, gatherDifference, 0, ...
    "Preserves every epoch loss but synchronises once per timing group.");

%% 7. Precompute the teacher-forced encoder block with one GEMM
teacherInput = gpuArray.randn(settings.dynamics_dimension, ...
    settings.teacher_block_steps, 'single');
encoderLoop = encoder_column_loop(Pdynamics.W_in, teacherInput, Pdynamics.inputScale);
encoderBlock = encoder_block(Pdynamics.W_in, teacherInput, Pdynamics.inputScale);
encoderLoopTime = gpu_wall_time(@() encoder_column_loop( ...
    Pdynamics.W_in, teacherInput, Pdynamics.inputScale), ...
    settings.timing_repetitions, device);
encoderBlockTime = gpu_wall_time(@() encoder_block( ...
    Pdynamics.W_in, teacherInput, Pdynamics.inputScale), ...
    settings.timing_repetitions, device);
rows(end+1) = benchmark_row( ... %#ok<SAGROW>
    "Precompute 30 teacher encoder currents", "isolated exact", ...
    encoderLoopTime, encoderBlockTime, relative_error(encoderLoop, encoderBlock), ...
    bytes_to_mib(numel(encoderBlock)), ...
    "Only teacher-forced steps can use this; reported speedup is for encoder work only.");

%% 8. Fuse elementwise current addition into an ARRAYFUN state kernel
fusionColumns = 3;
u = gpuArray.randn(settings.N_hidden, fusionColumns, 'single');
w = gpuArray.randn(settings.N_hidden, fusionColumns, 'single');
inputCurrent = gpuArray.randn(settings.N_hidden, fusionColumns, 'single');
recurrentCurrent = gpuArray.randn(settings.N_hidden, fusionColumns, 'single');
biasCurrent = gpuArray.randn(settings.N_hidden, fusionColumns, 'single');
[unfusedU, unfusedW] = proxy_unfused_update( ...
    u, w, inputCurrent, recurrentCurrent, biasCurrent);
[fusedU, fusedW] = proxy_fused_update( ...
    u, w, inputCurrent, recurrentCurrent, biasCurrent);
unfusedTime = gpu_wall_time(@() proxy_unfused_update( ...
    u, w, inputCurrent, recurrentCurrent, biasCurrent), ...
    settings.timing_repetitions, device);
fusedTime = gpu_wall_time(@() proxy_fused_update( ...
    u, w, inputCurrent, recurrentCurrent, biasCurrent), ...
    settings.timing_repetitions, device);
fusionDifference = max(relative_error(unfusedU, fusedU), ...
    relative_error(unfusedW, fusedW));
rows(end+1) = benchmark_row( ... %#ok<SAGROW>
    "Fuse total-current addition", "isolated exact", ...
    unfusedTime, fusedTime, fusionDifference, ...
    bytes_to_mib(5 * numel(u)), ...
    "Proxy scalar state equation; measures removal of elementwise launch/materialisation.");

%% 9. Avoid storing dynamics outputs when training requests only loss/gradient
outputWeight = Pdynamics.W_out;
filteredState = gpuArray.randn(settings.N_hidden, 1, 'single');
outputSteps = settings.dynamics_benchmark_steps;
storedOutput = output_storage_loop(outputWeight, filteredState, outputSteps, true);
lastOutput = output_storage_loop(outputWeight, filteredState, outputSteps, false);
storeTime = gpu_wall_time(@() output_storage_loop( ...
    outputWeight, filteredState, outputSteps, true), ...
    settings.timing_repetitions, device);
noStoreTime = gpu_wall_time(@() output_storage_loop( ...
    outputWeight, filteredState, outputSteps, false), ...
    settings.timing_repetitions, device);
storageDifference = relative_error(storedOutput(:, end), lastOutput);
rows(end+1) = benchmark_row( ... %#ok<SAGROW>
    "Skip unused training output storage", "isolated exact", ...
    storeTime, noStoreTime, storageDifference, ...
    bytes_to_mib(settings.dynamics_dimension * outputSteps), ...
    "Prediction GEMM remains necessary; only output-array writes are removed.");

%% 10. Allocate state arrays versus resetting reusable buffers
stateTemplate = make_state_template(settings.N_hidden, 32);
allocatedState = allocate_state_arrays(settings.N_hidden, 32);
resetState = reset_state_arrays(stateTemplate);
allocateTime = gpu_wall_time(@() allocate_state_arrays( ...
    settings.N_hidden, 32), settings.timing_repetitions, device);
resetTime = gpu_wall_time(@() reset_state_arrays( ...
    stateTemplate), settings.timing_repetitions, device);
rows(end+1) = benchmark_row( ... %#ok<SAGROW>
    "Reuse/reset state buffers", "isolated exact", ...
    allocateTime, resetTime, state_difference(allocatedState, resetState), ...
    estimate_static_state_mib(settings.N_hidden, 32), ...
    "MATLAB copy-on-write may limit reuse; benchmark determines whether it helps on this release.");

%% 11. Three independent seeds: sequential versus paged low-rank prototype
prototype = make_paged_prototype(settings.N_hidden, settings.N_recurrent, ...
    settings.dynamics_dimension, settings.seed_count);
[sequentialSeedOutput, sequentialSeedU, sequentialSeedR, sequentialSeedSpikes] = ...
    seed_prototype_sequential(prototype, settings.prototype_steps);
[pagedSeedOutput, pagedSeedU, pagedSeedR, pagedSeedSpikes] = ...
    seed_prototype_paged(prototype, settings.prototype_steps);
sequentialSeedTime = gpu_wall_time(@() seed_prototype_sequential( ...
    prototype, settings.prototype_steps), settings.timing_repetitions, device);
pagedSeedTime = gpu_wall_time(@() seed_prototype_paged( ...
    prototype, settings.prototype_steps), settings.timing_repetitions, device);
rows(end+1) = benchmark_row( ... %#ok<SAGROW>
    "Train 3 seeds as pages", "paged prototype", ...
    sequentialSeedTime, pagedSeedTime, ...
    relative_error(sequentialSeedOutput, pagedSeedOutput), ...
    prototype_memory_mib(prototype), ...
    "Same low-rank encoder/recurrent/decoder shapes; simplified state equation, so throughput evidence is provisional.");

agreementRows(end+1) = agreement_row( ... %#ok<SAGROW>
    "Paged seed full horizon", "decoder output", settings.seed_count, ...
    settings.prototype_steps, sequentialSeedOutput, pagedSeedOutput);
agreementRows(end+1) = agreement_row( ... %#ok<SAGROW>
    "Paged seed full horizon", "membrane state", settings.seed_count, ...
    settings.prototype_steps, sequentialSeedU, pagedSeedU);
agreementRows(end+1) = agreement_row( ... %#ok<SAGROW>
    "Paged seed full horizon", "filtered spike state", settings.seed_count, ...
    settings.prototype_steps, sequentialSeedR, pagedSeedR);
agreementRows(end+1) = agreement_row( ... %#ok<SAGROW>
    "Paged seed full horizon", "cumulative spike count", settings.seed_count, ...
    settings.prototype_steps, sequentialSeedSpikes, pagedSeedSpikes);

prototypeHorizons = unique([settings.agreement_horizons settings.prototype_steps]);
prototypeHorizons = prototypeHorizons(prototypeHorizons >= 1 & ...
    prototypeHorizons <= settings.prototype_steps);
for horizon = prototypeHorizons
    [sequentialOutput, sequentialU, sequentialR, sequentialSpikes] = ...
        seed_prototype_sequential(prototype, horizon);
    [pagedOutput, pagedU, pagedR, pagedSpikes] = ...
        seed_prototype_paged(prototype, horizon);
    agreementRows(end+1) = agreement_row( ... %#ok<SAGROW>
        "Paged seed divergence horizon", "decoder output", ...
        settings.seed_count, horizon, sequentialOutput, pagedOutput);
    agreementRows(end+1) = agreement_row( ... %#ok<SAGROW>
        "Paged seed divergence horizon", "membrane state", ...
        settings.seed_count, horizon, sequentialU, pagedU);
    agreementRows(end+1) = agreement_row( ... %#ok<SAGROW>
        "Paged seed divergence horizon", "filtered spike state", ...
        settings.seed_count, horizon, sequentialR, pagedR);
    agreementRows(end+1) = agreement_row( ... %#ok<SAGROW>
        "Paged seed divergence horizon", "cumulative spike count", ...
        settings.seed_count, horizon, sequentialSpikes, pagedSpikes);
end

%% Results
benchmark_results = struct2table(rows);
benchmark_results.Speedup = benchmark_results.BaselineSeconds ./ ...
    benchmark_results.CandidateSeconds;
benchmark_results.PercentTimeSaved = 100 .* ...
    (1 - benchmark_results.CandidateSeconds ./ benchmark_results.BaselineSeconds);
benchmark_results = movevars(benchmark_results, 'PercentTimeSaved', ...
    'After', 'Speedup');
numerical_agreement_details = struct2table(agreementRows);
numerical_agreement_onset = agreement_onset_table( ...
    numerical_agreement_details, settings.agreement_material_tolerance);

fprintf('\nMeasured benchmark results\n');
disp(benchmark_results(:, {'Effect','Scope','BaselineSeconds', ...
    'CandidateSeconds','Speedup','PercentTimeSaved', ...
    'MaxRelativeDifference','EstimatedMemoryMiB'}));
fprintf('\nCurrent dynamics-path workload projections\n');
disp(dynamics_projection);
fprintf('\nLarge static-batch throughput scaling\n');
disp(large_batch_scaling);
fprintf('\nDetailed numerical agreement diagnostics\n');
disp(numerical_agreement_details);
fprintf('\nFirst detected numerical divergence by quantity\n');
disp(numerical_agreement_onset);

%% Plots
figure('Color', 'w');
tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile;
bar(benchmark_results.Speedup);
yline(1, 'k--');
grid on;
ylabel('Measured speedup (baseline/candidate)');
title(sprintf('GPU: %s', device.Name));
set(gca, 'XTick', 1:height(benchmark_results), ...
    'XTickLabel', benchmark_results.Effect, 'XTickLabelRotation', 35);
nexttile;
semilogy(max(benchmark_results.MaxRelativeDifference, eps), 'o-', ...
    'LineWidth', 1.2, 'MarkerSize', 6);
grid on;
ylabel('Maximum relative difference');
xlabel('Benchmark row');
title('Numerical agreement (prototype rows may differ by operation ordering)');

figure('Color', 'w');
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile;
loglog(large_batch_scaling.BatchSize, large_batch_scaling.SamplesPerSecond, ...
    'o-', 'LineWidth', 1.4, 'MarkerSize', 7);
grid on;
xlabel('Batch size');
ylabel('Samples per second');
title('Current-path static throughput');
nexttile;
loglog(large_batch_scaling.BatchSize, large_batch_scaling.EstimatedStateMiB, ...
    'o-', 'LineWidth', 1.4, 'MarkerSize', 7);
grid on;
xlabel('Batch size');
ylabel('Estimated state/temporary memory (MiB)');
title('Conservative memory scaling');

figure('Color', 'w');
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile;
plot_agreement_horizon(numerical_agreement_details, ...
    "Static divergence horizon", ...
    ["average output" "average eligibility" "cumulative spike count"]);
title(sprintf('Static prefix: alone vs embedded in batch %d', probeBatchSize));
nexttile;
plot_agreement_horizon(numerical_agreement_details, ...
    "Paged seed divergence horizon", ...
    ["decoder output" "membrane state" "filtered spike state" ...
    "cumulative spike count"]);
title(sprintf('%d seeds: sequential vs paged', settings.seed_count));

if settings.save_results
    if exist(settings.output_directory, 'dir') ~= 7
        mkdir(settings.output_directory);
    end
    stamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
    save(fullfile(settings.output_directory, ...
        ['training_speed_benchmark_' stamp '.mat']), ...
        'benchmark_results', 'dynamics_projection', 'large_batch_scaling', ...
        'numerical_agreement_details', 'numerical_agreement_onset', 'settings');
    writetable(benchmark_results, fullfile(settings.output_directory, ...
        ['training_speed_benchmark_' stamp '.csv']));
    writetable(large_batch_scaling, fullfile(settings.output_directory, ...
        ['large_batch_scaling_' stamp '.csv']));
    writetable(numerical_agreement_details, fullfile(settings.output_directory, ...
        ['numerical_agreement_' stamp '.csv']));
    writetable(numerical_agreement_onset, fullfile(settings.output_directory, ...
        ['numerical_agreement_onset_' stamp '.csv']));
end

fprintf(['\nInterpretation rule: only "current path" rows are end-to-end/current-', ...
    'implementation measurements.\nExact-kernel proxies and isolated rows ', ...
    'quantify components; paged-prototype results require a production ', ...
    'equivalence implementation before adoption.\nThe detailed agreement ', ...
    'table reports absolute, RMS, norm-relative and exact-mismatch measures. ', ...
    'For cumulative spike count, any nonzero mismatch fraction means that ', ...
    'hard event selections have diverged by that horizon.\n']);

%% Local benchmark functions
function [gradient, lossMean] = static_epoch_core(P, X, Y, cfg, batchSize)
%STATIC_EPOCH_CORE Current full-dataset e-prop reduction without parameter update.
sampleCount = size(X, 2);
gradient = gpuArray.zeros(P.N_hidden, 1, 'single');
lossSum = gpuArray.zeros(1, 1, 'single');
for first = 1:batchSize:sampleCount
    indices = first:min(sampleCount, first + batchSize - 1);
    xBatch = gpuArray(single(X(:, indices)));
    yBatch = gpuArray(single(Y(:, indices)));
    [output, eligibility] = banff_model('static', P, xBatch, true, false);
    [batchLoss, outputGradient] = banff_eval('loss', output, yBatch, cfg.kind);
    learningSignal = P.W_out.' * outputGradient;
    gradient = gradient + sum(learningSignal .* eligibility, 2);
    lossSum = lossSum + batchLoss;
end
gradient = gradient ./ single(sampleCount);
lossMean = lossSum ./ single(sampleCount);
end

function seconds = gpu_wall_time(functionHandle, repetitions, device)
%GPU_WALL_TIME Median synchronised elapsed time after one untimed warm-up call.
functionHandle();
wait(device);
samples = zeros(repetitions, 1);
for repetition = 1:repetitions
    wait(device);
    timer = tic;
    functionHandle();
    wait(device);
    samples(repetition) = toc(timer);
end
seconds = median(samples);
end

function row = benchmark_row(effect, scope, baseline, candidate, difference, memoryMiB, notes)
row = struct('Effect', string(effect), 'Scope', string(scope), ...
    'BaselineSeconds', double(baseline), 'CandidateSeconds', double(candidate), ...
    'Speedup', double(baseline) / double(candidate), ...
    'MaxRelativeDifference', double(difference), ...
    'EstimatedMemoryMiB', double(memoryMiB), 'Notes', string(notes));
end

function row = agreement_row(caseName, quantity, batchSize, horizon, reference, candidate)
%AGREEMENT_ROW Compute complementary absolute, norm-relative and exact metrics.
% ScaleNormalizedMaxDifference uses a global scale floor of one; it therefore
% cannot become large solely because an individual reference element is zero.
reference = double(gather(reference));
candidate = double(gather(candidate));
assert(isequal(size(reference), size(candidate)), ...
    'Agreement comparison requires arrays of identical size.');

finitePair = isfinite(reference) & isfinite(candidate);
nonfinitePairCount = nnz(~finitePair);
referenceFinite = reference(finitePair);
candidateFinite = candidate(finitePair);
differenceFinite = referenceFinite - candidateFinite;
if isempty(differenceFinite)
    maxAbsolute = NaN;
    rmsDifference = NaN;
    relativeL2 = NaN;
    scaleNormalisedMax = NaN;
    referenceNorm = NaN;
    candidateNorm = NaN;
else
    absoluteDifference = abs(differenceFinite);
    maxAbsolute = max(absoluteDifference);
    rmsDifference = sqrt(mean(differenceFinite .* differenceFinite));
    referenceNorm = norm(referenceFinite(:), 2);
    candidateNorm = norm(candidateFinite(:), 2);
    relativeL2 = norm(differenceFinite(:), 2) ./ ...
        max([eps, referenceNorm, candidateNorm]);
    scale = max([1; abs(referenceFinite(:)); abs(candidateFinite(:))]);
    scaleNormalisedMax = maxAbsolute ./ scale;
end

elementCount = numel(reference);
exactMismatchCount = nnz(reference ~= candidate);
row = struct('Case', string(caseName), 'Quantity', string(quantity), ...
    'BatchSize', double(batchSize), 'Horizon', double(horizon), ...
    'MaxAbsoluteDifference', double(maxAbsolute), ...
    'RMSEDifference', double(rmsDifference), ...
    'RelativeL2Difference', double(relativeL2), ...
    'ScaleNormalizedMaxDifference', double(scaleNormalisedMax), ...
    'ExactMismatchCount', double(exactMismatchCount), ...
    'ElementCount', double(elementCount), ...
    'ExactMismatchFraction', double(exactMismatchCount) ./ double(elementCount), ...
    'ReferenceNorm2', double(referenceNorm), ...
    'CandidateNorm2', double(candidateNorm), ...
    'NonfinitePairCount', double(nonfinitePairCount));
end

function plot_agreement_horizon(details, caseName, quantities)
hold on;
for quantity = quantities
    selected = details(details.Case == caseName & ...
        details.Quantity == quantity, :);
    selected = sortrows(selected, 'Horizon');
    if ~isempty(selected)
        loglog(selected.Horizon, ...
            max(selected.MaxAbsoluteDifference, eps), ...
            'o-', 'LineWidth', 1.2, 'MarkerSize', 6, ...
            'DisplayName', char(quantity));
    end
end
hold off;
grid on;
xlabel('Simulation horizon (steps)');
ylabel('Maximum absolute difference');
legend('Location', 'best');
end

function onset = agreement_onset_table(details, materialTolerance)
%AGREEMENT_ONSET_TABLE Locate exact and practically material divergence horizons.
% For cumulative spike counts, the first exact mismatch is direct evidence that
% at least one hard event selection differed by that horizon.
isHorizonSweep = details.Case == "Static divergence horizon" | ...
    details.Case == "Paged seed divergence horizon";
sweep = details(isHorizonSweep, :);
pairs = unique(sweep(:, {'Case','Quantity'}), 'rows', 'stable');
caseValues = strings(height(pairs), 1);
quantityValues = strings(height(pairs), 1);
firstExact = nan(height(pairs), 1);
firstMaterial = nan(height(pairs), 1);
firstNonfinite = nan(height(pairs), 1);
finalMaxAbsolute = nan(height(pairs), 1);
finalRelativeL2 = nan(height(pairs), 1);
for index = 1:height(pairs)
    caseValues(index) = pairs.Case(index);
    quantityValues(index) = pairs.Quantity(index);
    selected = sweep(sweep.Case == pairs.Case(index) & ...
        sweep.Quantity == pairs.Quantity(index), :);
    selected = sortrows(selected, 'Horizon');
    exactIndex = find(selected.ExactMismatchCount > 0, 1, 'first');
    materialIndex = find(selected.MaxAbsoluteDifference > materialTolerance, ...
        1, 'first');
    nonfiniteIndex = find(selected.NonfinitePairCount > 0, 1, 'first');
    if ~isempty(exactIndex), firstExact(index) = selected.Horizon(exactIndex); end
    if ~isempty(materialIndex), firstMaterial(index) = selected.Horizon(materialIndex); end
    if ~isempty(nonfiniteIndex), firstNonfinite(index) = selected.Horizon(nonfiniteIndex); end
    finalMaxAbsolute(index) = selected.MaxAbsoluteDifference(end);
    finalRelativeL2(index) = selected.RelativeL2Difference(end);
end
onset = table(caseValues, quantityValues, firstExact, firstMaterial, ...
    firstNonfinite, finalMaxAbsolute, finalRelativeL2, ...
    'VariableNames', {'Case','Quantity','FirstExactMismatchHorizon', ...
    'FirstAboveToleranceHorizon','FirstNonfiniteHorizon', ...
    'FinalMaxAbsoluteDifference','FinalRelativeL2Difference'});
end

function errorValue = relative_error(first, second)
first = double(gather(first));
second = double(gather(second));
denominator = max([1; abs(first(:)); abs(second(:))]);
errorValue = max(abs(first(:) - second(:)), [], 'omitnan') / denominator;
if isempty(errorValue), errorValue = 0; end
end

function value = bytes_to_mib(singleValueCount)
value = 4 * double(singleValueCount) / 2^20;
end

function value = estimate_static_state_mib(nHidden, batchSize)
% Eight recurrent/eligibility states plus current, bias, state/eligibility sums
% and conservative temporaries for the optimized static path.
% This is deliberately conservative and is used only as an allocation guard.
value = bytes_to_mib(14 * double(nHidden) * double(batchSize));
end

function value = estimate_dynamics_state_mib(nHidden, trajectoryCount)
value = bytes_to_mib(12 * double(nHidden) * double(trajectoryCount));
end

function output = decoder_repeated(W, r, steps)
output = gpuArray.zeros(size(W, 1), size(r, 2), 'single');
for step = 1:steps
    output = output + W * r;
end
output = output ./ single(steps);
end

function output = decoder_once(W, r, steps)
stateSum = gpuArray.zeros(size(r), 'single');
for step = 1:steps
    stateSum = stateSum + r;
end
output = W * (stateSum ./ single(steps));
end

function output = validation_proxy_sequential(P, inputs)
output = gpuArray.zeros(P.N_output, size(inputs, 2), 'single');
for index = 1:size(inputs, 2)
    output(:, index) = banff_model('static', P, inputs(:, index), false, false);
end
end

function output = validation_proxy_batched(P, inputs)
output = banff_model('static', P, inputs, false, false);
end

function sequence = scheduled_sampling_local(samples, cfg)
sequence = true(1, samples);
period = double(cfg.teacher_steps + cfg.closed_loop_steps);
indices = 2:samples;
sequence(indices(mod(indices - 1, period) >= cfg.teacher_steps)) = false;
end

function [finalBias, losses] = dynamics_epoch_group(P, sequence, teacherForcing, cfg, epochCount, gatherEach)
lossBuffer = gpuArray.zeros(epochCount, 1, 'single');
losses = zeros(epochCount, 1, 'single');
for epoch = 1:epochCount
    [loss, gradient] = banff_model('dynamics', P, sequence, teacherForcing, true, false);
    P = banff_model('adam', P, gradient, single(1e-3), size(sequence, 2)-1, cfg);
    if gatherEach
        losses(epoch) = gather(loss);
    else
        lossBuffer(epoch) = loss;
    end
end
if ~gatherEach
    losses = gather(lossBuffer);
end
finalBias = gather(P.B);
end

function output = encoder_column_loop(W, X, scale)
output = gpuArray.zeros(size(W, 1), size(X, 2), 'single');
for column = 1:size(X, 2)
    output(:, column) = W * (scale .* X(:, column));
end
end

function output = encoder_block(W, X, scale)
output = W * (scale .* X);
end

function [u1, w1] = proxy_unfused_update(u, w, inputCurrent, recurrentCurrent, bias)
totalCurrent = inputCurrent + recurrentCurrent + bias;
[u1, w1] = arrayfun(@proxy_state_scalar, u, w, totalCurrent);
end

function [u1, w1] = proxy_fused_update(u, w, inputCurrent, recurrentCurrent, bias)
[u1, w1] = arrayfun(@proxy_state_components_scalar, ...
    u, w, inputCurrent, recurrentCurrent, bias);
end

function [u1, w1] = proxy_state_scalar(u, w, current)
u1 = single(.97) .* u + single(.03) .* (current - w);
w1 = single(.995) .* w + single(.005) .* u1;
end

function [u1, w1] = proxy_state_components_scalar(u, w, inputCurrent, recurrentCurrent, bias)
current = inputCurrent + recurrentCurrent + bias;
u1 = single(.97) .* u + single(.03) .* (current - w);
w1 = single(.995) .* w + single(.005) .* u1;
end

function output = output_storage_loop(W, r, steps, storeAll)
if storeAll
    output = gpuArray.zeros(size(W, 1), steps, 'single');
    for step = 1:steps
        output(:, step) = W * r;
    end
else
    output = gpuArray.zeros(size(W, 1), 1, 'single');
    for step = 1:steps
        output = W * r;
    end
end
end

function state = make_state_template(nHidden, batchSize)
state = allocate_state_arrays(nHidden, batchSize);
end

function state = allocate_state_arrays(nHidden, batchSize)
state = struct();
state.u = gpuArray.zeros(nHidden, batchSize, 'single');
state.w = gpuArray.zeros(nHidden, batchSize, 'single');
state.x = gpuArray.zeros(nHidden, batchSize, 'single');
state.r = gpuArray.zeros(nHidden, batchSize, 'single');
state.epsilonVoltage = gpuArray.zeros(nHidden, batchSize, 'single');
state.epsilonAdaptation = gpuArray.zeros(nHidden, batchSize, 'single');
state.eligibilityRise = gpuArray.zeros(nHidden, batchSize, 'single');
state.eligibilityDecay = gpuArray.zeros(nHidden, batchSize, 'single');
end

function state = reset_state_arrays(state)
names = fieldnames(state);
for index = 1:numel(names)
    value = state.(names{index});
    value(:) = single(0);
    state.(names{index}) = value;
end
end

function value = state_difference(first, second)
names = fieldnames(first);
value = 0;
for index = 1:numel(names)
    value = max(value, relative_error(first.(names{index}), second.(names{index})));
end
end

function prototype = make_paged_prototype(nHidden, rankValue, dimension, seedCount)
prototype.W_in = gpuArray.randn(nHidden, dimension, seedCount, 'single') ./ sqrt(single(dimension));
prototype.W_feedback = gpuArray.randn(rankValue, nHidden, seedCount, 'single') ./ sqrt(single(nHidden));
prototype.W_expansion = gpuArray.randn(nHidden, rankValue, seedCount, 'single') ./ sqrt(single(rankValue));
prototype.W_out = gpuArray.randn(dimension, nHidden, seedCount, 'single') ./ sqrt(single(nHidden));
prototype.B = gpuArray.randn(nHidden, 1, seedCount, 'single') .* single(.01);
prototype.input = gpuArray.randn(dimension, 1, seedCount, 'single');
end

function [output, finalU, finalR, spikeCount] = seed_prototype_sequential(P, steps)
%SEED_PROTOTYPE_SEQUENTIAL Reference layout: one independent seed at a time.
seedCount = size(P.W_in, 3);
output = gpuArray.zeros(size(P.W_out, 1), 1, seedCount, 'single');
finalU = gpuArray.zeros(size(P.W_in, 1), 1, seedCount, 'single');
finalR = gpuArray.zeros(size(P.W_in, 1), 1, seedCount, 'single');
spikeCount = gpuArray.zeros(size(P.W_in, 1), 1, seedCount, 'single');
for seed = 1:seedCount
    one.W_in = P.W_in(:,:,seed);
    one.W_feedback = P.W_feedback(:,:,seed);
    one.W_expansion = P.W_expansion(:,:,seed);
    one.W_out = P.W_out(:,:,seed);
    one.B = P.B(:,:,seed);
    one.input = P.input(:,:,seed);
    [oneOutput, oneU, oneR, oneSpikeCount] = seed_prototype_one(one, steps);
    output(:,:,seed) = oneOutput;
    finalU(:,:,seed) = oneU;
    finalR(:,:,seed) = oneR;
    spikeCount(:,:,seed) = oneSpikeCount;
end
end

function [output, u, r, spikeCount] = seed_prototype_one(P, steps)
nHidden = size(P.W_in, 1);
u = gpuArray.zeros(nHidden, 1, 'single');
r = gpuArray.zeros(nHidden, 1, 'single');
spikeCount = gpuArray.zeros(nHidden, 1, 'single');
networkInput = P.input;
for step = 1:steps
    latent = P.W_feedback * r;
    current = P.W_in * networkInput + P.W_expansion * latent + P.B;
    u = single(.97) .* u + single(.03) .* current - single(.01) .* r;
    spike = single(u > 0);
    r = single(.98) .* r + spike;
    spikeCount = spikeCount + spike;
    output = P.W_out * r;
    networkInput = single(.7) .* P.input + single(.3) .* output;
end
end

function [output, u, r, spikeCount] = seed_prototype_paged(P, steps)
%SEED_PROTOTYPE_PAGED Candidate layout: seed is the third matrix-page dimension.
nHidden = size(P.W_in, 1);
seedCount = size(P.W_in, 3);
u = gpuArray.zeros(nHidden, 1, seedCount, 'single');
r = gpuArray.zeros(nHidden, 1, seedCount, 'single');
spikeCount = gpuArray.zeros(nHidden, 1, seedCount, 'single');
networkInput = P.input;
for step = 1:steps
    latent = pagemtimes(P.W_feedback, r);
    current = pagemtimes(P.W_in, networkInput) ...
        + pagemtimes(P.W_expansion, latent) + P.B;
    u = single(.97) .* u + single(.03) .* current - single(.01) .* r;
    spike = single(u > 0);
    r = single(.98) .* r + spike;
    spikeCount = spikeCount + spike;
    output = pagemtimes(P.W_out, r);
    networkInput = single(.7) .* P.input + single(.3) .* output;
end
end

function value = prototype_memory_mib(P)
value = 0;
names = fieldnames(P);
for index = 1:numel(names)
    value = value + bytes_to_mib(numel(P.(names{index})));
end
end
