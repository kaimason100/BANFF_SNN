%% simulate_random_network_activity.m
% Forward-only CPU diagnostic using the same model constructor and readable
% reference timestep as the training code. No neuron parameters are copied.
% This example is qualitative and does not optimise parameters. State arrays
% follow the production model convention, and all displayed times are seconds.

clear; clc; close all;
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);

%% User settings
N_hidden = 500;
nInput = 2;
duration = 10.0;          % s
nVoltageTraces = 8;
nRasterNeurons = N_hidden;
inputStd = 1.0;
inputHoldTime = 0.025;    % s
networkSeed = 1;
inputSeed = 1234;

%% Resolve the canonical BANFF configuration
cfg = banff("config", "lorenz", struct( ...
    'seed', networkSeed, ...
    'N_hidden', N_hidden, ...
    'N_recurrent', min(10, N_hidden), ...
    'recurrent_mode', "low_rank"));
P = banff_model('create', nInput, 1, cfg);

dt = double(cfg.dt);
nSteps = round(duration / dt);
time = (0:nSteps-1) .* dt;
inputHoldSteps = max(1, round(inputHoldTime / dt));
nBlocks = ceil(nSteps / inputHoldSteps);
rng(inputSeed, 'twister');
blockInput = single(inputStd) .* randn(nInput, nBlocks, 'single');
randomInput = repelem(blockInput, 1, inputHoldSteps);
randomInput = randomInput(:, 1:nSteps);

state = struct( ...
    'u', repmat(P.restingVoltage, P.N_hidden, 1), ...
    'w', zeros(P.N_hidden, 1, 'single'), ...
    'x', zeros(P.N_hidden, 1, 'single'), ...
    'r', zeros(P.N_hidden, 1, 'single'), ...
    'epsilonVoltage', zeros(P.N_hidden, 1, 'single'), ...
    'epsilonAdaptation', zeros(P.N_hidden, 1, 'single'), ...
    'eligibilityRise', zeros(P.N_hidden, 1, 'single'), ...
    'eligibilityDecay', zeros(P.N_hidden, 1, 'single'));

traceIds = unique(round(linspace(1, P.N_hidden, min(nVoltageTraces, P.N_hidden))));
voltage = zeros(numel(traceIds), nSteps, 'single');
spikes = false(P.N_hidden, nSteps);

fprintf('Forward diagnostic: N=%d, duration=%.3f s, dt=%.4g s\n', ...
    P.N_hidden, duration, dt);
for step = 1:nSteps
    inputCurrent = P.W_in * (P.inputScale .* randomInput(:, step));
    [state, fired] = banff_model('reference_step', P, state, inputCurrent, false);
    voltage(:, step) = state.u(traceIds);
    spikes(:, step) = fired;
end

counts = sum(spikes, 2);
totalSpikes = sum(counts);
meanRate = totalSpikes / (P.N_hidden * duration);
activeFraction = mean(counts > 0);
isi = [];
for neuron = 1:P.N_hidden
    locations = find(spikes(neuron, :));
    if numel(locations) >= 2
        isi = [isi diff(locations) .* dt .* 1e3]; %#ok<AGROW>
    end
end

fprintf('Total spikes: %d\n', totalSpikes);
fprintf('Mean population rate: %.3f Hz\n', meanRate);
fprintf('Active-neuron fraction: %.3f\n', activeFraction);
if ~isempty(isi)
    fprintf('Median ISI: %.3f ms | mean ISI: %.3f ms | CV: %.3f\n', ...
        median(isi), mean(isi), std(isi)/mean(isi));
end

figure;
plot(time, voltage.'); xlabel('Time (s)'); ylabel('Membrane voltage (mV)');

displayNeurons = min(nRasterNeurons, P.N_hidden);
[neuron, step] = find(spikes(1:displayNeurons, :));
figure;
scatter((step-1).*dt, neuron, 5, '.'); xlabel('Time (s)'); ylabel('Neuron');

figure;
if isempty(isi)
    text(.5,.5,'No inter-spike intervals','HorizontalAlignment','center'); axis off;
else
    histogram(isi, 50); xlabel('Inter-spike interval (ms)'); ylabel('Count');
end
