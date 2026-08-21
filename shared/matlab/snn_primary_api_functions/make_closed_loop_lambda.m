% make_closed_loop_lambda.m
function lambda = make_closed_loop_lambda(steps)
lambda = false(1, max(1, steps));
lambda(1) = true;
end

