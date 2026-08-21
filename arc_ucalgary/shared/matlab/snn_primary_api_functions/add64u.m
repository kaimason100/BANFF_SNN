% add64u.m
function r = add64u(a,b)
a = uint64(a); b = uint64(b);
mask = uint64(4294967295);
a_lo = bitand(a, mask); b_lo = bitand(b, mask);
lo = a_lo + b_lo;
carry = bitshift(lo, -32);
lo = bitand(lo, mask);
a_hi = bitshift(a,-32); b_hi = bitshift(b,-32);
hi = bitand(a_hi + b_hi + carry, mask);
r = bitshift(hi,32) + lo;
end

