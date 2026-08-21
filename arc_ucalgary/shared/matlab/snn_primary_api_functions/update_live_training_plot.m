% update_live_training_plot.m
function update_live_training_plot(kind, hist, ep, opts)
%UPDATE_LIVE_TRAINING_PLOT Refreshes inline Live Editor training curves.
%   opts.live_plot.enable turns the plot on. opts.live_plot.every controls
%   the epoch interval, while epoch 1 and the final epoch are always shown.
cfg = get_opt(opts, 'live_plot', struct());
if ~logical(get_opt(cfg, 'enable', false))
    return;
end
every = max(1, round(double(get_opt(cfg, 'every', 10))));
if ~(ep == 1 || ep == opts.epochs || mod(ep, every) == 0)
    return;
end

persistent static_fig dynamics_fig
if isstruct(hist) && isfield(hist, 'closed_wd')
    if isempty(dynamics_fig) || ~ishandle(dynamics_fig)
        dynamics_fig = figure('Color', 'w');
    else
        figure(dynamics_fig);
    end
    clf(dynamics_fig);
    plot_dynamics_live_history(hist, ep, opts);
elseif isstruct(hist)
    if isempty(static_fig) || ~ishandle(static_fig)
        static_fig = figure('Color', 'w');
    else
        figure(static_fig);
    end
    clf(static_fig);
    plot_static_live_history(kind, hist, ep, opts);
else
    if isempty(dynamics_fig) || ~ishandle(dynamics_fig)
        dynamics_fig = figure('Color', 'w');
    else
        figure(dynamics_fig);
    end
    clf(dynamics_fig);
    plot_dynamics_live_history(hist, ep, opts);
end
drawnow limitrate;
end
