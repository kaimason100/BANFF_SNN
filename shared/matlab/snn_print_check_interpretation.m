% Package orientation: Shared MATLAB utility used across tasks. Read this with the caller open so input/output structs and saved-result fields are clear.

function snn_print_check_interpretation(summary_table, check_label)
%SNN_PRINT_CHECK_INTERPRETATION Print a plain-English CPU-vs-GPU check summary.
%   SUMMARY_TABLE is the output of snn_check_summary_table. CHECK_LABEL is
%   optional and is used only in the printed heading.

if nargin < 2 || isempty(check_label)
    check_label = 'CPU/GPU check';
end
if isempty(summary_table)
    fprintf('\n%s interpretation: no check rows were produced.\n', char(check_label));
    return;
end

fprintf('\n%s interpretation\n', char(check_label));
fprintf('------------------------------------------------------------\n');
fprintf('What matters: CPU-GPU difference columns should be tiny. Training-effect columns show learning happened; they are not errors.\n');

verdict = string(summary_table.verdict);
n_rows = height(summary_table);
n_ok = sum(verdict == "OK");
n_skipped = sum(verdict == "OUTPUT SKIPPED");
n_scale = sum(verdict == "CHECK SCALE");
n_review = sum(verdict == "REVIEW");

if n_review == 0 && n_skipped == 0
    if n_scale == 0
        fprintf('Overall: OK. All %d seed(s) agree within the configured check tolerance.\n', n_rows);
    else
        fprintf('Overall: CHECK SCALE. %d/%d seed(s) are within tolerance but large relative to the training effect.\n', n_scale, n_rows);
    end
elseif n_review == 0
    if n_scale == 0
        fprintf('Overall: OK with skipped output arrays in %d/%d seed(s). Bias, loss, and neural replay still ran.\n', n_skipped, n_rows);
    else
        fprintf('Overall: CHECK SCALE with skipped output arrays in %d/%d seed(s). Bias, loss, and neural replay still ran.\n', n_skipped, n_rows);
    end
else
    fprintf('Overall: REVIEW. %d/%d seed(s) need attention.\n', n_review, n_rows);
end

print_worst_line(summary_table, 'Before-training output/trajectory CPU-GPU diff', ...
    'initial_abs', 'initial_rel');
print_worst_line(summary_table, 'After-training output/trajectory CPU-GPU diff', ...
    'main_cpu_gpu_abs', 'main_cpu_gpu_rel');
print_worst_line(summary_table, 'Before-training bias CPU-GPU diff', 'bias_before_abs', 'bias_before_rel');
print_worst_line(summary_table, 'After-training bias CPU-GPU diff', 'bias_abs', 'bias_rel');
print_worst_line(summary_table, 'CPU output/trajectory training change', 'cpu_output_change_abs', '');
print_worst_line(summary_table, 'GPU output/trajectory training change', 'gpu_output_change_abs', '');
print_worst_line(summary_table, 'CPU bias training change', 'cpu_bias_change_abs', '');
print_worst_line(summary_table, 'GPU bias training change', 'gpu_bias_change_abs', '');
print_worst_line(summary_table, 'CPU-vs-GPU output training-change diff', 'output_training_effect_abs_diff', '');
print_worst_line(summary_table, 'CPU-vs-GPU bias training-change diff', 'bias_training_effect_abs_diff', '');
print_worst_line(summary_table, 'Post-training loss CPU-GPU diff', 'loss_after_abs', '');
print_worst_line(summary_table, 'Training loss-change CPU-GPU diff', 'loss_delta_abs_diff', '');
print_worst_line(summary_table, 'Post-training metric CPU-GPU diff', 'metric_after_abs', '');
print_worst_line(summary_table, 'Worst neural replay diff', 'worst_neural_abs', '');
print_percent_line(summary_table, 'Output/trajectory mismatch as percent of training effect', ...
    'main_diff_vs_training_percent');
print_percent_line(summary_table, 'Bias mismatch as percent of training effect', ...
    'bias_diff_vs_training_percent');
print_percent_line(summary_table, 'Loss mismatch as percent of CPU post-training loss', ...
    'loss_after_diff_percent');
print_percent_line(summary_table, 'Metric mismatch as percent of CPU post-training metric', ...
    'metric_after_diff_percent');

fprintf('\nQualitative categories:\n');
print_category_counts(summary_table, 'output_quality', 'Output/trajectory agreement');
print_category_counts(summary_table, 'performance_quality', 'Loss/metric agreement');
print_category_counts(summary_table, 'learning_quality', 'Training/bias agreement');
print_category_counts(summary_table, 'neural_quality', 'Neural replay agreement');

fprintf('\nPer-seed verdicts:\n');
for ii = 1:n_rows
    seed_value = summary_table.seed(ii);
    if isnan(seed_value)
        seed_text = sprintf('%d', ii);
    else
        seed_text = sprintf('%g', seed_value);
    end
    fprintf('  seed %s: %-14s output %.3g, bias %.3g, loss %.3g, neural %.3g. %s\n', ...
        seed_text, char(verdict(ii)), ...
        value_or_nan(summary_table.main_cpu_gpu_abs(ii)), ...
        value_or_nan(summary_table.bias_abs(ii)), ...
        value_or_nan(summary_table.loss_after_abs(ii)), ...
        value_or_nan(summary_table.worst_neural_abs(ii)), ...
        char(summary_table.read_first(ii)));
end
fprintf('------------------------------------------------------------\n\n');
end

function print_worst_line(T, label, abs_field, rel_field)
if ~any(strcmp(T.Properties.VariableNames, abs_field))
    return;
end
values = double(T.(abs_field));
[worst_value, idx] = max_omitnan_compat(values);
if isnan(worst_value)
    fprintf('%s: unavailable.\n', label);
    return;
end
seed_text = seed_label(T, idx);
if ~isempty(rel_field) && any(strcmp(T.Properties.VariableNames, rel_field))
    rel_values = double(T.(rel_field));
    fprintf('%s: %.6g at seed %s (relative %.6g).\n', ...
        label, worst_value, seed_text, value_or_nan(rel_values(idx)));
else
    fprintf('%s: %.6g at seed %s.\n', label, worst_value, seed_text);
end
end

function print_percent_line(T, label, field_name)
if ~any(strcmp(T.Properties.VariableNames, field_name))
    return;
end
values = double(T.(field_name));
[worst_value, idx] = max_omitnan_compat(values);
if isnan(worst_value)
    fprintf('%s: unavailable because the training effect was zero or missing.\n', label);
    return;
end
fprintf('%s: %.6g%% at seed %s. Smaller is better.\n', ...
    label, worst_value, seed_label(T, idx));
end

function print_category_counts(T, field_name, label)
if ~any(strcmp(T.Properties.VariableNames, field_name))
    return;
end
values = string(T.(field_name));
cats = unique(values);
pieces = strings(numel(cats), 1);
for ii = 1:numel(cats)
    pieces(ii) = sprintf('%s=%d', char(cats(ii)), sum(values == cats(ii)));
end
fprintf('%s: %s.\n', label, char(strjoin(pieces, ', ')));
end

function [value, idx] = max_omitnan_compat(values)
values = double(values(:));
valid = isfinite(values);
if ~any(valid)
    value = nan;
    idx = 1;
    return;
end
valid_idx = find(valid);
[value, local_idx] = max(values(valid));
idx = valid_idx(local_idx);
end

function s = seed_label(T, idx)
if any(strcmp(T.Properties.VariableNames, 'seed')) && isfinite(T.seed(idx))
    s = sprintf('%g', T.seed(idx));
else
    s = sprintf('%d', idx);
end
end

function v = value_or_nan(v)
if ~isfinite(v)
    v = nan;
end
end
