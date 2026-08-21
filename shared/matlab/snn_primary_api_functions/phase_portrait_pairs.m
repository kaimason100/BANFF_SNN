% phase_portrait_pairs.m
function pairs = phase_portrait_pairs(n_states)
if n_states >= 2
    pairs = nchoosek(1:n_states, 2);
else
    pairs = [1 1];
end
end

