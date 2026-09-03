%% Eigenvalues of the effective recurrent-current operator
% This diagnostic constructs the recurrent parameters using BANFF's production
% model initialisation, then analyses the linear operator that maps filtered
% presynaptic spike rates to recurrent current.
%
% For the low-rank architecture, the production operator is
%
%   W_eff = g_rec E F - diag(d),
%   d     = diag(g_rec E F),
%
% where E is the recurrent expansion, F is the feedback matrix, and g_rec is
% the recurrent gain. Thus W_eff has an exactly zero diagonal, as used during
% simulation. The script never forms the dense N-by-N matrix unless N is below
% the explicitly selected small-network threshold.
%
% Important interpretation: W_eff maps filtered rates to current. Its
% eigenvalues are therefore not eigenvalues of the complete discrete-time
% neuron-state update. In particular, the unit circle is not by itself a
% stability boundary for this current operator. Stability also depends on the
% membrane, adaptation and synaptic-filter dynamics and on the local slope of
% the spike-generation mechanism.

clearvars;
close all;
clc;

%% User settings
task = "lorenz";
seed = 1;

% These overrides should match the network whose initial recurrent operator
% is to be examined. Fixed recurrent weights are not trained by BANFF, so a
% saved trained model is not needed when these settings and the seed match.
overrides = struct;
overrides.N_hidden = 32000;
overrides.recurrent_mode = "low_rank";  % "low_rank" or "full_rank"
overrides.N_recurrent = 10;             % used only for low rank
overrides.recurrent_gain = 0.05;

numberOfEigenvalues = 30;    % extremal Ritz values requested from eigs
eigsTolerance = 1e-8;
eigsMaximumIterations = 2000;
exactSpectrumMaximumN = 1500; % set to 0 to disable dense exact calculation
diagonalReferencePoints = 2000;

%% Construct the same recurrent operator used by the production model
projectRoot = fileparts(fileparts(mfilename("fullpath")));
addpath(projectRoot);

overrides.seed = seed;
cfg = banff("config", task, overrides);

if cfg.kind == "dynamics"
    system = banff_data("system", cfg.task);
    inputDimension = numel(system.initial_state);
    outputDimension = inputDimension;
else
    [data, ~] = banff_data("static", cfg);
    inputDimension = size(data.X_train, 1);
    outputDimension = size(data.Y_train, 1);
end

parameters = banff_model("create", inputDimension, outputDimension, cfg);
N = cfg.N_hidden;

fprintf("Effective recurrent-operator eigenvalue diagnostic\n");
fprintf("Task: %s | seed: %d | N: %d | architecture: %s\n", ...
    char(cfg.task), seed, N, char(cfg.recurrent_mode));
fprintf("Recurrent gain: %.9g\n\n", cfg.recurrent_gain);

if cfg.recurrent_mode == "low_rank"
    expansion = double(parameters.recurrent_expansion); % N by rank
    feedback = double(parameters.W_feedback);           % rank by N
    recurrentGain = double(parameters.recurrentGain);
    removedDiagonal = double(parameters.self_coupling(:));

    % AB and BA have the same non-zero eigenvalues. Consequently, all non-zero
    % eigenvalues of g_rec E F can be calculated exactly from the small
    % rank-by-rank matrix g_rec F E.
    uncorrectedNonzeroEigenvalues = eig(recurrentGain .* (feedback * expansion));

    applyOperator = @(x) recurrentGain .* (expansion * (feedback * x)) ...
        - removedDiagonal .* x;

    fprintf("Low-rank dimension: %d\n", size(feedback, 1));
    fprintf(['Before diagonal removal, the operator has at most %d non-zero ' ...
        'eigenvalues and %d structural zeros.\n'], ...
        size(feedback, 1), max(0, N - size(feedback, 1)));
    fprintf("Uncorrected low-rank spectral radius: %.9g\n", ...
        max(abs(uncorrectedNonzeroEigenvalues)));
    fprintf(['Removed self-coupling d: min %.9g, mean %.9g, ' ...
        'standard deviation %.9g, max %.9g\n\n'], ...
        min(removedDiagonal), mean(removedDiagonal), ...
        std(removedDiagonal), max(removedDiagonal));

    explicitMatrix = [];
    if N <= exactSpectrumMaximumN
        explicitMatrix = recurrentGain .* (expansion * feedback) ...
            - diag(removedDiagonal);
    end
elseif cfg.recurrent_mode == "full_rank"
    explicitSparseMatrix = double(parameters.W_recurrent);
    applyOperator = @(x) explicitSparseMatrix * x;
    uncorrectedNonzeroEigenvalues = complex(zeros(0, 1));
    removedDiagonal = full(diag(explicitSparseMatrix));

    fprintf("Full-rank matrix non-zeros: %d (density %.6g)\n\n", ...
        nnz(explicitSparseMatrix), nnz(explicitSparseMatrix) / double(N)^2);

    explicitMatrix = [];
    if N <= exactSpectrumMaximumN
        explicitMatrix = full(explicitSparseMatrix);
    end
else
    error("Unsupported recurrent mode: %s", cfg.recurrent_mode);
end

%% Calculate the exact small-network spectrum or extremal Ritz values
if ~isempty(explicitMatrix)
    [exactVectors, exactMatrix] = eig(explicitMatrix);
    exactEigenvalues = diag(exactMatrix);
    exactResiduals = eigen_residuals(applyOperator, exactVectors, ...
        exactEigenvalues);
    reportedCount = min(numberOfEigenvalues, N);
    [~, exactMagnitudeOrder] = sort(abs(exactEigenvalues), "descend");
    exactMagnitudeOrder = exactMagnitudeOrder(1:reportedCount);
    magnitudeEigenvalues = exactEigenvalues(exactMagnitudeOrder);
    magnitudeResiduals = exactResiduals(exactMagnitudeOrder);
    [~, exactRightmostOrder] = sort(real(exactEigenvalues), "descend");
    exactRightmostOrder = exactRightmostOrder(1:reportedCount);
    rightmostEigenvalues = exactEigenvalues(exactRightmostOrder);
    rightmostResiduals = exactResiduals(exactRightmostOrder);
    magnitudeFlag = 0;
    rightmostFlag = 0;
    spectrumDescription = "exact full spectrum";
else
    exactEigenvalues = complex(zeros(0, 1));
    if N <= 2
        error("The matrix-free eigensolver requires N greater than 2.");
    end

    requestedCount = min(numberOfEigenvalues, N - 2);
    eigsOptions = struct;
    eigsOptions.isreal = true;
    eigsOptions.issym = false;
    eigsOptions.tol = eigsTolerance;
    eigsOptions.maxit = eigsMaximumIterations;
    eigsOptions.p = min(N, max(2 * requestedCount + 10, 20));
    eigsOptions.v0 = deterministic_start_vector(N);
    eigsOptions.disp = 0;

    [magnitudeVectors, magnitudeMatrix, magnitudeFlag] = eigs( ...
        applyOperator, N, requestedCount, 'largestabs', eigsOptions);
    magnitudeEigenvalues = diag(magnitudeMatrix);
    magnitudeResiduals = eigen_residuals(applyOperator, magnitudeVectors, ...
        magnitudeEigenvalues);

    [rightmostVectors, rightmostMatrix, rightmostFlag] = eigs( ...
        applyOperator, N, requestedCount, 'largestreal', eigsOptions);
    rightmostEigenvalues = diag(rightmostMatrix);
    rightmostResiduals = eigen_residuals(applyOperator, rightmostVectors, ...
        rightmostEigenvalues);
    spectrumDescription = sprintf("%d extremal Ritz values in each search", ...
        requestedCount);
end

[~, magnitudeOrder] = sort(abs(magnitudeEigenvalues), "descend");
magnitudeEigenvalues = magnitudeEigenvalues(magnitudeOrder);
magnitudeResiduals = magnitudeResiduals(magnitudeOrder);

[~, rightmostOrder] = sort(real(rightmostEigenvalues), "descend");
rightmostEigenvalues = rightmostEigenvalues(rightmostOrder);
rightmostResiduals = rightmostResiduals(rightmostOrder);

spectralRadiusEstimate = max(abs(magnitudeEigenvalues));
spectralAbscissaEstimate = max(real(rightmostEigenvalues));

fprintf("Spectrum calculation: %s\n", spectrumDescription);
fprintf("Spectral radius%s: %.9g\n", estimate_suffix(exactEigenvalues), ...
    spectralRadiusEstimate);
fprintf("Spectral abscissa%s: %.9g\n", estimate_suffix(exactEigenvalues), ...
    spectralAbscissaEstimate);
fprintf("Largest-magnitude eigs flag: %d | rightmost eigs flag: %d\n", ...
    magnitudeFlag, rightmostFlag);
fprintf(['A flag of zero indicates convergence. Inspect the residual columns ' ...
    'below even when the flag is zero.\n\n']);

magnitudeTable = eigenvalue_table(magnitudeEigenvalues, magnitudeResiduals);
rightmostTable = eigenvalue_table(rightmostEigenvalues, rightmostResiduals);

fprintf("Eigenvalues ordered by decreasing magnitude\n");
disp(magnitudeTable);
fprintf("Eigenvalues ordered by decreasing real part\n");
disp(rightmostTable);

%% Plot the calculated spectrum
figure;
tiledlayout(1, 2, "TileSpacing", "compact", "Padding", "compact");

nexttile;
hold on;
if ~isempty(exactEigenvalues)
    scatter(real(exactEigenvalues), imag(exactEigenvalues), 24, ...
        "filled", "DisplayName", "Exact eigenvalues");
else
    scatter(real(magnitudeEigenvalues), imag(magnitudeEigenvalues), 42, ...
        "o", "LineWidth", 1.2, "DisplayName", "Largest magnitude");
    scatter(real(rightmostEigenvalues), imag(rightmostEigenvalues), 42, ...
        "x", "LineWidth", 1.2, "DisplayName", "Largest real part");
end
if ~isempty(uncorrectedNonzeroEigenvalues)
    scatter(real(uncorrectedNonzeroEigenvalues), ...
        imag(uncorrectedNonzeroEigenvalues), 55, "d", "LineWidth", 1.2, ...
        "DisplayName", "g_{rec}EF before diagonal removal");
end
xline(0, ":", "HandleVisibility", "off");
yline(0, ":", "HandleVisibility", "off");
axis equal;
grid on;
xlabel("Real part");
ylabel("Imaginary part");
title("Effective recurrent-current eigenvalues");
legend("Location", "best");

nexttile;
if cfg.recurrent_mode == "low_rank"
    numberToPlot = min(diagonalReferencePoints, N);
    sampleIndices = unique(round(linspace(1, N, numberToPlot)));
    scatter(-removedDiagonal(sampleIndices), zeros(numel(sampleIndices), 1), ...
        15, "filled", "MarkerFaceAlpha", 0.25);
    hold on;
    xline(0, ":");
    grid on;
    xlabel("Real-axis value");
    ylabel("Imaginary part");
    title("Sample of -d diagonal reference values");
    subtitle("Reference values are not, individually, matrix eigenvalues");
else
    histogram(real(magnitudeEigenvalues));
    grid on;
    xlabel("Real part");
    ylabel("Count");
    title("Real parts of calculated eigenvalues");
end

fprintf(['\nInterpretation warning: these eigenvalues describe the filtered-rate-' ...
    'to-current operator only. They do not, alone, establish dynamical ' ...
    'stability, chaos, or an edge-of-chaos regime.\n']);

%% Local functions
function residuals = eigen_residuals(applyOperator, vectors, eigenvalues)
%EIGEN_RESIDUALS Return relative right-eigenpair residual norms.
residuals = zeros(size(eigenvalues));
for index = 1:numel(eigenvalues)
    vector = vectors(:, index);
    operatorVector = applyOperator(vector);
    residuals(index) = norm(operatorVector - eigenvalues(index) .* vector) ...
        / max(norm(operatorVector), eps);
end
end

function output = deterministic_start_vector(N)
%DETERMINISTIC_START_VECTOR Avoid changing MATLAB's global random stream.
indices = (1:N).';
output = sin(indices .* sqrt(2)) + cos(indices .* sqrt(3));
output = output ./ norm(output);
end

function output = eigenvalue_table(eigenvalues, residuals)
%EIGENVALUE_TABLE Format complex eigenvalues without unequal table columns.
modeIndex = (1:numel(eigenvalues)).';
realPart = real(eigenvalues(:));
imaginaryPart = imag(eigenvalues(:));
magnitude = abs(eigenvalues(:));
relativeResidual = residuals(:);
output = table(modeIndex, realPart, imaginaryPart, magnitude, ...
    relativeResidual, 'VariableNames', {'Mode', 'RealPart', ...
    'ImaginaryPart', 'Magnitude', 'RelativeResidual'});
end

function output = estimate_suffix(exactEigenvalues)
%ESTIMATE_SUFFIX Label matrix-free extremal results as estimates.
if isempty(exactEigenvalues)
    output = ' estimate';
else
    output = '';
end
end
