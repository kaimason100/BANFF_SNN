% default_static_options.m
function opts = default_static_options(domain, backend, mode)
opts = common_options(mode);
opts.PRESENT = struct('T', single(0.300), 'avg_frac', single(0.5));
opts.steps_present = max(1, round(opts.PRESENT.T/opts.dt));
opts.steps_avg = max(1, round(opts.PRESENT.avg_frac * opts.steps_present));
opts.k_avg_start = opts.steps_present - opts.steps_avg + 1;
opts.batch_size = 32;
opts.validate_every = 5;
opts.dataset = char(domain);
opts.synthetic = false;
% Regression duplicates are grouped by their complete feature-plus-target row
% before splitting, preventing identical observations from crossing partitions.
opts.group_exact_duplicate_rows = domain == "regression";
if mode == "train" && backend == "gpu"
    opts.N_hidden = 32000;
    opts.epochs = 5000;
else
    opts.N_hidden = 128;
    opts.epochs = 3;
end
if mode == "check" || mode == "bench"
    opts.N_hidden = 32;
    opts.N_rec = 4;
    opts.steps_present = 12;
    opts.steps_avg = 6;
    opts.k_avg_start = 7;
    opts.epochs = 1;
    opts.batch_size = 4;
end
end
