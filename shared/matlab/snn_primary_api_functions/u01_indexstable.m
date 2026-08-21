% u01_indexstable.m
function u = u01_indexstable(mat_id, i, j, seed)
i64 = uint64(i(:));
j64 = uint64(j(:)).';
[I64,J64] = ndgrid(i64, j64);
A = const64('9E3779B97F4A7C15');
B = const64('BF58476D1CE4E5B9');
C = const64('94D049BB133111EB');
x = add64u(uint64(seed), mul64u(A, uint64(mat_id)+1));
x = add64u(x, mul64u(B, I64));
x = add64u(x, mul64u(C, J64));
k = splitmix64_mod(x);
u = (double(bitshift(k,-11)) + 0.5) / 2^53;
end

