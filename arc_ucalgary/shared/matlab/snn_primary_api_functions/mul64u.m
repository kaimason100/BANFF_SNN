% mul64u.m
function p = mul64u(a,b)
a = uint64(a); b = uint64(b);
mask = uint64(4294967295);
a_lo = bitand(a, mask); a_hi = bitshift(a,-32);
b_lo = bitand(b, mask); b_hi = bitshift(b,-32);
p00 = a_lo .* b_lo;
p01 = a_lo .* b_hi;
p10 = a_hi .* b_lo;
mid = add64u(p01, p10);
p = add64u(bitshift(mid,32), p00);
end

