% assert_check_passed.m
function assert_check_passed(cmp, tol, label)
if cmp.has_nan || cmp.has_inf
    error('snn_primary_api:checkFailed', ...
        '%s contain non-finite values: has_nan=%d, has_inf=%d.', label, cmp.has_nan, cmp.has_inf);
end
if cmp.max_abs > tol.abs && cmp.max_rel > tol.rel
    error('snn_primary_api:checkFailed', ...
        '%s differ too much: max_abs=%g, max_rel=%g.', label, cmp.max_abs, cmp.max_rel);
end
end

