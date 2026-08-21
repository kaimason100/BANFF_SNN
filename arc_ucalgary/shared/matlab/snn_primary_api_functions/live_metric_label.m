% live_metric_label.m
function label = live_metric_label(kind)
kind = lower(string(kind));
if kind == "classification"
    label = 'Accuracy [%]';
elseif kind == "regression"
    label = 'Pearson r';
else
    label = 'Training metric';
end
end

