% fmt_progress_number.m
function txt = fmt_progress_number(value)
if ~isnumeric(value) || isempty(value) || ~isfinite(double(value(1)))
    txt = 'n/a';
else
    txt = sprintf('%.6g', double(value(1)));
end
end

