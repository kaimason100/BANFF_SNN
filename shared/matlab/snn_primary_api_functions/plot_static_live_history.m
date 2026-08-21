% plot_static_live_history.m
function plot_static_live_history(kind, hist, ep, opts)
epochs = 1:ep;
subplot(1,2,1);
plot(epochs, double(hist.train_loss(epochs)), 'LineWidth', 1.3); hold on;
plot_finite_history(epochs, hist.val_loss(epochs), 1.3);
xlabel('Epoch'); ylabel('Loss'); title('Training and validation loss'); grid on;
set_loss_axis_log(gca);
legend({'Train','Validation'}, 'Location', 'best');

subplot(1,2,2);
plot(epochs, double(hist.train_metric(epochs)), 'LineWidth', 1.3); hold on;
plot_finite_history(epochs, hist.val_metric(epochs), 1.3);
metric_name = live_metric_label(kind);
xlabel('Epoch'); ylabel(metric_name); title(metric_name); grid on;
legend({'Train','Validation'}, 'Location', 'best');
arch_label = get_opt(opts, 'arch_label', '');
if strlength(string(arch_label)) > 0
    live_sgtitle(sprintf('%s training: %s, epoch %d of %d', char(kind), char(arch_label), ep, opts.epochs));
else
    live_sgtitle(sprintf('%s training, epoch %d of %d', char(kind), ep, opts.epochs));
end
end
