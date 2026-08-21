% normalize_dynamics_trajectory.m
function [x, mu, sigma] = normalize_dynamics_trajectory(x) %#ok<DEFNU>
mu = mean(x,2);
sigma = std(x,0,2); sigma(sigma==0) = 1;
x = single((x - mu) ./ sigma);
end

