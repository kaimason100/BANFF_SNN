% warn_if_dynamics_norm_differs.m
function warn_if_dynamics_norm_differs(saved_mu, saved_sigma, recomputed_mu, recomputed_sigma)
tol_abs = 1e-5;
tol_rel = 1e-4;
saved_mu = single(saved_mu);
saved_sigma = single(saved_sigma);
recomputed_mu = single(recomputed_mu);
recomputed_sigma = single(recomputed_sigma);
mu_cmp = compare_arrays(saved_mu, recomputed_mu);
sigma_cmp = compare_arrays(saved_sigma, recomputed_sigma);
if (mu_cmp.max_abs > tol_abs && mu_cmp.max_rel > tol_rel) || ...
        (sigma_cmp.max_abs > tol_abs && sigma_cmp.max_rel > tol_rel)
    warning('snn_primary_api:dynamicsNormRecomputedDiffers', ...
        ['Explicit recompute_dynamics_norm=true produced normalization statistics that differ ', ...
         'from the saved training statistics. max_abs mu %.6g sigma %.6g; max_rel mu %.6g sigma %.6g.'], ...
        mu_cmp.max_abs, sigma_cmp.max_abs, mu_cmp.max_rel, sigma_cmp.max_rel);
end
end

