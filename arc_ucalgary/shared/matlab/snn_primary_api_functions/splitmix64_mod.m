% splitmix64_mod.m
function k = splitmix64_mod(x)
x = add64u(uint64(x), const64('9E3779B97F4A7C15'));
z = bitxor(x, bitshift(x,-30));
z = mul64u(z, const64('BF58476D1CE4E5B9'));
z = bitxor(z, bitshift(z,-27));
z = mul64u(z, const64('94D049BB133111EB'));
k = bitxor(z, bitshift(z,-31));
end

