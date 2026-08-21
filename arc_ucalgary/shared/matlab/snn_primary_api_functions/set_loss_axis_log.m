% set_loss_axis_log.m
function set_loss_axis_log(ax)
%SET_LOSS_AXIS_LOG Use a log y-axis for strictly positive loss curves.
%   MATLAB cannot draw non-positive values on a log axis, so keep the axis
%   linear only if the currently plotted loss values are all missing or
%   non-positive. This avoids warnings for interrupted or empty histories.
ys = [];
kids = get(ax, 'Children');
for ii = 1:numel(kids)
    if isprop(kids(ii), 'YData')
        ys = [ys; double(kids(ii).YData(:))]; %#ok<AGROW>
    end
end
ys = ys(isfinite(ys) & ys > 0);
if ~isempty(ys)
    set(ax, 'YScale', 'log');
end
end

