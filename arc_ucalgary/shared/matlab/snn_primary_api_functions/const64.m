% const64.m
function u = const64(hex16)
hex16 = upper(strtrim(hex16));
hi = uint64(base2dec(hex16(1:8),16));
lo = uint64(base2dec(hex16(9:16),16));
u = bitshift(hi,32) + lo;
end
