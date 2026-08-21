% plot_dynamics_live_history.m
function plot_dynamics_live_history(hist, ep, opts)
if nargin < 3
    opts = struct();
end
arch_label = get_opt(opts, 'arch_label', '');
epochs = 1:ep;
if isstruct(hist)
    subplot(1,2,1);
    plot(epochs, double(hist.train_loss(epochs)), 'LineWidth', 1.3);
    xlabel('Epoch'); ylabel('Mean trajectory loss');
    title(sprintf('Training loss, epoch %d', ep)); grid on;
    set_loss_axis_log(gca);

    subplot(1,2,2);
    plot_finite_history(epochs, hist.closed_wd(epochs), 1.3);
    xlabel('Epoch'); ylabel('Closed-loop WD');
    title('Closed-loop validation WD'); grid on;
    set_loss_axis_log(gca);
    if strlength(string(arch_label)) > 0
        live_sgtitle(sprintf('Dynamical-system training: %s, epoch %d of %d', char(arch_label), ep, opts.epochs));
    else
        live_sgtitle(sprintf('Dynamical-system training, epoch %d of %d', ep, opts.epochs));
    end
else
    plot(epochs, double(hist(epochs)), 'LineWidth', 1.3);
    xlabel('Epoch'); ylabel('Mean trajectory loss');
    title(sprintf('Dynamical-system training loss, epoch %d of %d', ep, numel(hist)));
    grid on;
    set_loss_axis_log(gca);
    if strlength(string(arch_label)) > 0
        live_sgtitle(sprintf('Dynamical-system training: %s, epoch %d of %d', char(arch_label), ep, numel(hist)));
    end
end
end
