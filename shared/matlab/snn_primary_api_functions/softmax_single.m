% softmax_single.m
function p = softmax_single(z)
z = single(z);
e = exp(z - max(z));
p = e ./ max(sum(e), realmin('single'));
end

