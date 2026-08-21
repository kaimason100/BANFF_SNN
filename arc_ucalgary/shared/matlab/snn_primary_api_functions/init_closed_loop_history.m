% init_closed_loop_history.m
function hist = init_closed_loop_history(epochs)
hist = struct();
hist.wd = nan(epochs,1,'single');
end
