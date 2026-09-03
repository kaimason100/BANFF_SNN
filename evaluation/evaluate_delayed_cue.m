%% Delayed cue-response evaluation
% Evaluate completed delayed-cue models under the intact, zero-recurrence,
% and cue-removed conditions. Figures produced from this Live Script appear
% in the Live Editor output panel.

%% User settings
seeds = 1:3;
profile = "main";
overrides = struct();
display_options = struct( ...
    'minimum_full_accuracy_percent',80, ...
    'minimum_ablation_drop_points',20, ...
    'maximum_cue_removed_accuracy_percent',60, ...
    'save_figures',false, ...
    'figure_visibility',"on");

%% Run the held-out evaluation and recurrence controls
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root,fullfile(root,'evaluation'));
report = banff_evaluate_delayed_cue( ...
    seeds,profile,overrides,display_options);
