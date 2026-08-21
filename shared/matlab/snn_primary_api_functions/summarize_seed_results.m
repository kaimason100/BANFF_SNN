% summarize_seed_results.m
function summary = summarize_seed_results(results)
summary = struct();
summary.n_seeds = numel(results);
summary.seed_table = seed_results_to_table(results);
summary.metric_table = summarize_seed_metric_table(summary.seed_table);
summary.value_by_seed = primary_seed_values(summary.seed_table);
summary.mean = mean(summary.value_by_seed, 'omitnan');
summary.sd = std(summary.value_by_seed, 0, 'omitnan');
end

