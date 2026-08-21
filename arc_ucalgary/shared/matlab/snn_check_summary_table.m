% Package orientation: Shared MATLAB utility used across tasks. Read this with the caller open so input/output structs and saved-result fields are clear.

function T = snn_check_summary_table(results)
%SNN_CHECK_SUMMARY_TABLE Convert GPU-vs-CPU check results to a compact table.
%   RESULTS may be one result struct or an array of result structs returned
%   by snn_primary_api('check_static',...) or snn_primary_api('check_dynamics',...).

if iscell(results)
    result_cells = results(:);
else
    results = results(:);
    result_cells = cell(numel(results),1);
    for jj = 1:numel(results)
        result_cells{jj} = results(jj);
    end
end
n = numel(result_cells);
seed = nan(n,1);
initial_abs = nan(n,1);
initial_rel = nan(n,1);
initial_rms = nan(n,1);
after_abs = nan(n,1);
after_rel = nan(n,1);
after_rms = nan(n,1);
bias_before_abs = nan(n,1);
bias_before_rel = nan(n,1);
bias_before_rms = nan(n,1);
bias_abs = nan(n,1);
bias_rel = nan(n,1);
bias_rms = nan(n,1);
cpu_output_change_abs = nan(n,1);
gpu_output_change_abs = nan(n,1);
cpu_bias_change_abs = nan(n,1);
gpu_bias_change_abs = nan(n,1);
cpu_initial_loss = nan(n,1);
gpu_initial_loss = nan(n,1);
cpu_after_loss = nan(n,1);
gpu_after_loss = nan(n,1);
loss_after_abs = nan(n,1);
loss_train_abs = nan(n,1);
cpu_loss_delta = nan(n,1);
gpu_loss_delta = nan(n,1);
cpu_initial_metric = nan(n,1);
gpu_initial_metric = nan(n,1);
cpu_after_metric = nan(n,1);
gpu_after_metric = nan(n,1);
metric_after_abs = nan(n,1);
metric_train_abs = nan(n,1);
cpu_metric_delta = nan(n,1);
gpu_metric_delta = nan(n,1);
neural_spike_abs = nan(n,1);
neural_spike_rel = nan(n,1);
neural_rate_abs = nan(n,1);
neural_rate_rel = nan(n,1);
neural_voltage_abs = nan(n,1);
neural_voltage_rel = nan(n,1);
neural_input_current_abs = nan(n,1);
neural_input_current_rel = nan(n,1);
neural_recurrent_current_abs = nan(n,1);
neural_recurrent_current_rel = nan(n,1);
neural_adaptation_abs = nan(n,1);
neural_adaptation_rel = nan(n,1);
active_percent_abs = nan(n,1);
epochs = nan(n,1);
output_compared = false(n,1);
tol_abs = nan(n,1);
tol_rel = nan(n,1);

for ii = 1:n
    r = result_cells{ii};
    if isfield(r, 'options') && isfield(r.options, 'seed')
        seed(ii) = double(r.options.seed);
    elseif isfield(r, 'seed')
        seed(ii) = double(r.seed);
    end
    if isfield(r, 'options') && isfield(r.options, 'epochs')
        epochs(ii) = double(r.options.epochs);
    end
    if isfield(r, 'options') && isfield(r.options, 'check_tolerance')
        tol_abs(ii) = get_struct_value(r.options.check_tolerance, 'abs', nan);
        tol_rel(ii) = get_struct_value(r.options.check_tolerance, 'rel', nan);
    end
    initial_abs(ii) = get_cmp_value(r, 'initial', 'max_abs');
    initial_rel(ii) = get_cmp_value(r, 'initial', 'max_rel');
    initial_rms(ii) = get_cmp_value(r, 'initial', 'rms');
    after_abs(ii) = get_cmp_value(r, 'after_training', 'max_abs');
    after_rel(ii) = get_cmp_value(r, 'after_training', 'max_rel');
    after_rms(ii) = get_cmp_value(r, 'after_training', 'rms');
    bias_before_abs(ii) = get_cmp_value(r, 'bias_before_training', 'max_abs');
    bias_before_rel(ii) = get_cmp_value(r, 'bias_before_training', 'max_rel');
    bias_before_rms(ii) = get_cmp_value(r, 'bias_before_training', 'rms');
    bias_abs(ii) = get_cmp_value(r, 'bias_after_training', 'max_abs');
    bias_rel(ii) = get_cmp_value(r, 'bias_after_training', 'max_rel');
    bias_rms(ii) = get_cmp_value(r, 'bias_after_training', 'rms');
    cpu_output_change_abs(ii) = get_cmp_value(r, 'cpu_output_training_effect', 'max_abs');
    gpu_output_change_abs(ii) = get_cmp_value(r, 'gpu_output_training_effect', 'max_abs');
    cpu_bias_change_abs(ii) = get_cmp_value(r, 'cpu_bias_training_effect', 'max_abs');
    gpu_bias_change_abs(ii) = get_cmp_value(r, 'gpu_bias_training_effect', 'max_abs');
    if isfield(r, 'cpu')
        cpu_initial_loss(ii) = get_struct_value(r.cpu, 'initial_loss', nan);
        cpu_after_loss(ii) = get_struct_value(r.cpu, 'after_loss', nan);
        cpu_initial_metric(ii) = get_struct_value(r.cpu, 'initial_metric', nan);
        cpu_after_metric(ii) = get_struct_value(r.cpu, 'after_metric', nan);
    end
    if isfield(r, 'gpu')
        gpu_initial_loss(ii) = get_struct_value(r.gpu, 'initial_loss', nan);
        gpu_after_loss(ii) = get_struct_value(r.gpu, 'after_loss', nan);
        gpu_initial_metric(ii) = get_struct_value(r.gpu, 'initial_metric', nan);
        gpu_after_metric(ii) = get_struct_value(r.gpu, 'after_metric', nan);
    end
    if isfield(r, 'scalar_differences')
        loss_after_abs(ii) = get_struct_value(r.scalar_differences, 'after_loss_abs', nan);
        loss_train_abs(ii) = get_struct_value(r.scalar_differences, 'train_loss_abs', nan);
        cpu_loss_delta(ii) = get_struct_value(r.scalar_differences, 'cpu_loss_delta', nan);
        gpu_loss_delta(ii) = get_struct_value(r.scalar_differences, 'gpu_loss_delta', nan);
        metric_after_abs(ii) = get_struct_value(r.scalar_differences, 'after_metric_abs', nan);
        metric_train_abs(ii) = get_struct_value(r.scalar_differences, 'train_metric_abs', nan);
        cpu_metric_delta(ii) = get_struct_value(r.scalar_differences, 'cpu_metric_delta', nan);
        gpu_metric_delta(ii) = get_struct_value(r.scalar_differences, 'gpu_metric_delta', nan);
    end
    if isfield(r, 'neural')
        neural_spike_abs(ii) = get_nested_cmp_value(r.neural, 'spike_raster_difference', 'max_abs');
        neural_spike_rel(ii) = get_nested_cmp_value(r.neural, 'spike_raster_difference', 'max_rel');
        neural_rate_abs(ii) = get_nested_cmp_value(r.neural, 'rate_difference', 'max_abs');
        neural_rate_rel(ii) = get_nested_cmp_value(r.neural, 'rate_difference', 'max_rel');
        neural_voltage_abs(ii) = get_nested_cmp_value(r.neural, 'voltage_difference', 'max_abs');
        neural_voltage_rel(ii) = get_nested_cmp_value(r.neural, 'voltage_difference', 'max_rel');
        neural_input_current_abs(ii) = get_nested_cmp_value(r.neural, 'input_current_difference', 'max_abs');
        neural_input_current_rel(ii) = get_nested_cmp_value(r.neural, 'input_current_difference', 'max_rel');
        neural_recurrent_current_abs(ii) = get_nested_cmp_value(r.neural, 'recurrent_current_difference', 'max_abs');
        neural_recurrent_current_rel(ii) = get_nested_cmp_value(r.neural, 'recurrent_current_difference', 'max_rel');
        neural_adaptation_abs(ii) = get_nested_cmp_value(r.neural, 'adaptation_difference', 'max_abs');
        neural_adaptation_rel(ii) = get_nested_cmp_value(r.neural, 'adaptation_difference', 'max_rel');
        active_percent_abs(ii) = get_struct_value(r.neural, 'active_percent_abs_diff', nan);
    end
    output_compared(ii) = ~isfield(r, 'output_comparison_available') || logical(r.output_comparison_available);
end

main_cpu_gpu_abs = after_abs;
main_cpu_gpu_rel = after_rel;
main_cpu_gpu_abs(~output_compared) = bias_abs(~output_compared);
main_cpu_gpu_rel(~output_compared) = bias_rel(~output_compared);
output_training_effect_abs = max(cpu_output_change_abs, gpu_output_change_abs);
bias_training_effect_abs = max(cpu_bias_change_abs, gpu_bias_change_abs);
main_diff_vs_training_percent = percent_of_reference(main_cpu_gpu_abs, output_training_effect_abs);
bias_diff_vs_training_percent = percent_of_reference(bias_abs, bias_training_effect_abs);
loss_after_diff_percent = percent_of_reference(loss_after_abs, abs(cpu_after_loss));
metric_after_diff_percent = percent_of_reference(metric_after_abs, abs(cpu_after_metric));
loss_delta_abs_diff = abs(cpu_loss_delta - gpu_loss_delta);
metric_delta_abs_diff = abs(cpu_metric_delta - gpu_metric_delta);
output_training_effect_abs_diff = abs(cpu_output_change_abs - gpu_output_change_abs);
bias_training_effect_abs_diff = abs(cpu_bias_change_abs - gpu_bias_change_abs);
worst_neural_abs = row_max_omitnan([neural_spike_abs, neural_rate_abs, neural_voltage_abs, ...
    neural_input_current_abs, neural_recurrent_current_abs, neural_adaptation_abs]);
verdict = repmat("OK", n, 1);
read_first = repmat("CPU and GPU agree within the configured check tolerance.", n, 1);
output_quality = repmat("OK", n, 1);
performance_quality = repmat("OK", n, 1);
learning_quality = repmat("OK", n, 1);
neural_quality = repmat("OK", n, 1);
for ii = 1:n
    has_nonfinite = any([initial_abs(ii), after_abs(ii), bias_abs(ii), loss_after_abs(ii), ...
        neural_spike_abs(ii), neural_rate_abs(ii), neural_voltage_abs(ii)] == inf) || ...
        any(isnan([main_cpu_gpu_abs(ii), bias_abs(ii), loss_after_abs(ii)]));
    if has_nonfinite
        verdict(ii) = "REVIEW";
        read_first(ii) = "One or more core comparisons are missing or non-finite.";
    elseif ~output_compared(ii)
        verdict(ii) = "OUTPUT SKIPPED";
        output_quality(ii) = "SKIPPED";
        read_first(ii) = "The GPU MEX did not return output arrays; bias, loss, and neural replay were checked.";
    elseif exceeds_tolerance(main_cpu_gpu_abs(ii), main_cpu_gpu_rel(ii), tol_abs(ii), tol_rel(ii))
        verdict(ii) = "REVIEW";
        output_quality(ii) = "REVIEW";
        read_first(ii) = "Post-training CPU/GPU output difference is above the configured tolerance.";
    elseif exceeds_tolerance(bias_abs(ii), bias_rel(ii), tol_abs(ii), tol_rel(ii))
        verdict(ii) = "REVIEW";
        learning_quality(ii) = "REVIEW";
        read_first(ii) = "Trained CPU/GPU bias difference is above the configured tolerance.";
    elseif exceeds_tolerance(loss_after_abs(ii), nan, tol_abs(ii), tol_rel(ii))
        verdict(ii) = "REVIEW";
        performance_quality(ii) = "REVIEW";
        read_first(ii) = "Post-training CPU/GPU loss difference is above the absolute tolerance.";
    elseif isfinite(main_diff_vs_training_percent(ii)) && main_diff_vs_training_percent(ii) > 1
        verdict(ii) = "CHECK SCALE";
        output_quality(ii) = "CHECK SCALE";
        read_first(ii) = "CPU/GPU mismatch is more than 1 percent of the training effect.";
    end
    if output_compared(ii) && output_quality(ii) == "OK" && ...
            isfinite(main_diff_vs_training_percent(ii)) && main_diff_vs_training_percent(ii) > 0.1
        output_quality(ii) = "SMALL";
    end
    if performance_quality(ii) == "OK" && isfinite(loss_after_diff_percent(ii)) && loss_after_diff_percent(ii) > 0.1
        performance_quality(ii) = "SMALL";
    end
    if learning_quality(ii) == "OK" && isfinite(bias_diff_vs_training_percent(ii)) && bias_diff_vs_training_percent(ii) > 0.1
        learning_quality(ii) = "SMALL";
    end
    if isfinite(worst_neural_abs(ii)) && worst_neural_abs(ii) > max(1, tol_abs(ii))
        neural_quality(ii) = "CHECK SCALE";
    end
end

T = table(seed, epochs, output_compared, initial_abs, initial_rms, initial_rel, ...
    after_abs, after_rms, after_rel, ...
    bias_before_abs, bias_before_rms, bias_before_rel, bias_abs, bias_rms, bias_rel, ...
    loss_after_abs, loss_train_abs, ...
    cpu_output_change_abs, gpu_output_change_abs, cpu_bias_change_abs, gpu_bias_change_abs, ...
    cpu_initial_loss, gpu_initial_loss, cpu_after_loss, gpu_after_loss, cpu_loss_delta, gpu_loss_delta, ...
    cpu_initial_metric, gpu_initial_metric, cpu_after_metric, gpu_after_metric, ...
    metric_after_abs, metric_train_abs, cpu_metric_delta, gpu_metric_delta, ...
    neural_spike_abs, neural_spike_rel, neural_rate_abs, neural_rate_rel, ...
    neural_voltage_abs, neural_voltage_rel, neural_input_current_abs, neural_input_current_rel, ...
    neural_recurrent_current_abs, neural_recurrent_current_rel, ...
    neural_adaptation_abs, neural_adaptation_rel, active_percent_abs, ...
    tol_abs, tol_rel, main_cpu_gpu_abs, main_cpu_gpu_rel, ...
    main_diff_vs_training_percent, bias_diff_vs_training_percent, ...
    loss_after_diff_percent, metric_after_diff_percent, loss_delta_abs_diff, ...
    metric_delta_abs_diff, output_training_effect_abs_diff, bias_training_effect_abs_diff, ...
    worst_neural_abs, output_quality, performance_quality, learning_quality, ...
    neural_quality, verdict, read_first);
end

function v = get_cmp_value(r, cmp_name, field_name)
if isfield(r, cmp_name) && isfield(r.(cmp_name), field_name)
    v = double(r.(cmp_name).(field_name));
else
    v = nan;
end
end

function v = get_struct_value(s, field_name, fallback)
if isfield(s, field_name)
    v = double(s.(field_name));
else
    v = fallback;
end
end

function v = get_nested_cmp_value(s, cmp_name, field_name)
if isfield(s, cmp_name) && isstruct(s.(cmp_name)) && isfield(s.(cmp_name), field_name)
    v = double(s.(cmp_name).(field_name));
else
    v = nan;
end
end

function p = percent_of_reference(value, reference)
p = nan(size(value));
mask = isfinite(value) & isfinite(reference) & abs(reference) > 0;
p(mask) = 100 .* abs(value(mask)) ./ abs(reference(mask));
zero_mask = isfinite(value) & isfinite(reference) & reference == 0 & value == 0;
p(zero_mask) = 0;
end

function out = row_max_omitnan(Y)
out = nan(size(Y,1),1);
for ii = 1:size(Y,1)
    row = Y(ii,:);
    row = row(isfinite(row));
    if ~isempty(row)
        out(ii) = max(row);
    end
end
end

function tf = exceeds_tolerance(abs_value, rel_value, abs_tol, rel_tol)
tf = false;
if isfinite(abs_tol) && isfinite(abs_value) && abs_value > abs_tol
    if ~isfinite(rel_tol) || ~isfinite(rel_value) || rel_value > rel_tol
        tf = true;
    end
end
end
