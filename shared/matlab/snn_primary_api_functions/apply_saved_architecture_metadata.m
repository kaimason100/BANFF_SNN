% apply_saved_architecture_metadata.m
% Helper for snn_primary_api.

function opts = apply_saved_architecture_metadata(opts, train_result)
%APPLY_SAVED_ARCHITECTURE_METADATA Preserve architecture used for training.
if isfield(train_result, 'arch') && isstruct(train_result.arch) && isfield(train_result.arch, 'requested')
    opts.arch = train_result.arch.requested;
elseif isfield(train_result, 'arch') && isstruct(train_result.arch)
    opts.arch = train_result.arch;
elseif isfield(train_result, 'model') && isstruct(train_result.model)
    if ~isfield(opts, 'arch') || isempty(opts.arch)
        opts.arch = default_arch_options();
    end
    if isfield(train_result.model, 'recurrent_mode')
        opts.arch.recurrent_mode = string(train_result.model.recurrent_mode);
    end
    if isfield(train_result.model, 'decoder_mode')
        opts.arch.decoder_mode = string(train_result.model.decoder_mode);
    end
end
opts = normalize_arch_options(opts);
end
