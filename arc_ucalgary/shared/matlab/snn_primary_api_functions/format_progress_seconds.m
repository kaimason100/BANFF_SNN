% format_progress_seconds.m
function txt = format_progress_seconds(seconds)
seconds = max(0, double(seconds));
h = floor(seconds / 3600);
m = floor(mod(seconds, 3600) / 60);
s = floor(mod(seconds, 60));
txt = sprintf('%02d:%02d:%02d', h, m, s);
end

