% Package orientation: Shared plotting entry point for saved dynamics test data.

function result = plot_saved_dynamics_test_figures(opts)
%PLOT_SAVED_DYNAMICS_TEST_FIGURES Plot trajectories, phase portraits, sections and return maps.
%   This is intentionally data-only: it loads saved full-test results and
%   delegates rendering to the project plotting helpers.

if nargin < 1 || isempty(opts)
    opts = struct();
end
result = load_saved_dynamics_test_results(opts);
min_peak_prominence = get_opt_local(opts, 'min_peak_prominence', 1);
snn_plot_dynamics_test_trajectories(result);
snn_plot_dynamics_sections_and_return_maps(result, '', min_peak_prominence);
fprintf('Plotted saved closed-loop test data from %d analysis file(s).%s', ...
    numel(result.source_analysis_files), newline);
end

function value = get_opt_local(S, name, default_value)
if isstruct(S) && isfield(S, name) && ~isempty(S.(name))
    value = S.(name);
else
    value = default_value;
end
end
