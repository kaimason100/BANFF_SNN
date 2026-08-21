% current_train_loss.m
function value = current_train_loss(hist, ep)
if isstruct(hist) && isfield(hist, 'train_loss')
    value = history_value(hist.train_loss, ep);
else
    value = history_value(hist, ep);
end
end

