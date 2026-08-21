% plot_finite_history.m
function plot_finite_history(epochs, values, line_width)
mask = isfinite(values);
if any(mask)
    plot(epochs(mask), double(values(mask)), 'LineWidth', line_width);
end
end

