% init_static_history.m
function hist = init_static_history(epochs)
hist = struct();
hist.train_loss = nan(epochs,1,'single');
hist.train_metric = nan(epochs,1,'single');
hist.val_loss = nan(epochs,1,'single');
hist.val_metric = nan(epochs,1,'single');
end

