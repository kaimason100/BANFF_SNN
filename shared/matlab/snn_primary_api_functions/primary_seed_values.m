% primary_seed_values.m
function vals = primary_seed_values(T)
if any(strcmp(T.Properties.VariableNames, 'PrimaryMetric'))
    vals = double(T.PrimaryMetric);
else
    vals = nan(height(T), 1);
end
end

