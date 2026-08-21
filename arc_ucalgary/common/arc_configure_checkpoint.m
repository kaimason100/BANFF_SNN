function opts = arc_configure_checkpoint(opts, model_file)
%ARC_CONFIGURE_CHECKPOINT Enable exact checkpoint resume for long ARC jobs.
%   The training API saves the network, histories, RNG state and GPU
%   Adam/AMSGrad buffers at epoch boundaries. ARC jobs use a 23 hour cutoff
%   so there is time to write the checkpoint and submit the resumed job
%   before the 24 hour cluster walltime limit.

[model_dir, model_base] = fileparts(model_file);
checkpoint_dir = fullfile(model_dir, '..', 'checkpoints');
if exist(checkpoint_dir, 'dir') ~= 7
    mkdir(checkpoint_dir);
end

opts.arc_checkpoint = struct();
opts.arc_checkpoint.enable = true;
opts.arc_checkpoint.max_seconds = 23 * 3600;
opts.arc_checkpoint.file = fullfile(checkpoint_dir, [model_base '_checkpoint.mat']);
opts.arc_checkpoint.model_file = model_file;
end
