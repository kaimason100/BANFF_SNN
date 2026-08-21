% is_better.m
function tf = is_better(domain, loss, metric, best_loss, best_metric)
if domain == "classification"
    tf = (metric > best_metric) || (metric == best_metric && loss < best_loss);
else
    tf = loss < best_loss;
end
end

