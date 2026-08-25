%% Rate-Preserving Spike-Timing Shuffle Analysis
% This script creates a reusable analysis MAT file from saved full-test spike
% events. It does not rerun the network or true dynamical system. Each neuron
% retains its exact spike count in every shuffle window while its within-window
% timing changes. Run plot_jitter.m afterwards to make figures from the data.

clear; clc; close all;
repo_root = project_root();
addpath(fullfile(repo_root, 'figures', 'matlab'), '-begin');
add_project_paths(repo_root);

%% ---------------- Saved Test Selection ----------------
% The script loads the newest timestamped saved test for each task below.
% Each selected saved test must contain exactly three trained-network seeds.
analysis_dir = fullfile(repo_root, 'outputs', 'publication_analysis');
task_ids = {
    'dynamical_systems_lorenz'
    'dynamical_systems_sprotts'
    'dynamical_systems_vanderpol'
    };
network_seed_indices = [1 2 3];
required_metric_network_seeds = 3;

%% ---------------- Timing-Shuffle Controls ----------------
% These are the only perturbed conditions. The unperturbed network is stored
% once as the baseline; no duplicate 0 ms or one-timestep no-shuffle condition
% is generated. Values are shuffle-window widths in seconds.
rate_shuffle_window_s = [0.025 0.050 0.100 0.200 0.4 0.8];
rate_shuffle_seed = 2026;

%% ---------------- Output ----------------
% Timestamped results are kept separately from the source test analyses so
% figure styling can be changed later without repeating this expensive step.
output_dir = fullfile(analysis_dir, 'jitter_analysis');
if exist(output_dir, 'dir') ~= 7, mkdir(output_dir); end

opts = struct();
opts.analysis_dir = analysis_dir;
opts.task_ids = task_ids;
opts.network_seed_indices = network_seed_indices;
opts.required_metric_network_seeds = required_metric_network_seeds;
opts.rate_shuffle_window_s = rate_shuffle_window_s;
opts.rate_shuffle_seed = rate_shuffle_seed;

result = analyse_saved_dynamics_rate_coding(opts);
saved_jitter_analysis = struct();
saved_jitter_analysis.schema_version = 1;
saved_jitter_analysis.analysis_kind = 'rate_preserving_within_window_timing_shuffle';
saved_jitter_analysis.created_at = datestr(now, 31);
saved_jitter_analysis.shuffle_window_ms = 1e3 .* rate_shuffle_window_s;
saved_jitter_analysis.source_analysis_dir = analysis_dir;
saved_jitter_analysis.result = result;

stamp = datestr(now, 'yyyymmdd_HHMMSS');
output_file = fullfile(output_dir, sprintf('rate_preserving_jitter_analysis_%s.mat', stamp));
save(output_file, 'saved_jitter_analysis', '-v7.3');
fprintf(['Saved rate-preserving timing-shuffle analysis data to:\n%s\n', ...
    'Run plot_jitter.m to generate figures without recomputing the analysis.\n'], output_file);

