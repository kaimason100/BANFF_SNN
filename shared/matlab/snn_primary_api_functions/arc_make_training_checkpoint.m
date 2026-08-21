% arc_make_training_checkpoint.m
function checkpoint = arc_make_training_checkpoint(kind, domain, backend, ep, model, best, hist, opts, optimizer_state)
%ARC_MAKE_TRAINING_CHECKPOINT Capture every state needed for exact resume.
checkpoint = struct();
checkpoint.kind = char(kind);
checkpoint.domain = char(domain);
checkpoint.backend = char(backend);
checkpoint.epoch = double(ep);
checkpoint.model = model;
checkpoint.best = best;
checkpoint.history = hist;
checkpoint.final_B = model.B;
checkpoint.options = opts;
checkpoint.optimizer_state = optimizer_state;
checkpoint.rng_state = rng;
checkpoint.created_at = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
checkpoint.needs_resubmit = true;
checkpoint.complete = false;
end

