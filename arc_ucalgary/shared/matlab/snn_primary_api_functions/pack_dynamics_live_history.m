% pack_dynamics_live_history.m
function hist = pack_dynamics_live_history(train_loss, closed_hist)
hist = struct();
hist.train_loss = train_loss;
hist.closed_wd = closed_hist.wd;
end
