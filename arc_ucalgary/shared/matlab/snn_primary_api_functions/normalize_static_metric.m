% normalize_static_metric.m
function metric = normalize_static_metric(domain, metric_raw, count)
if domain == "classification"
    metric = single(100 * double(metric_raw) / max(1,double(count)));
else
    metric = single(metric_raw);
end
end

