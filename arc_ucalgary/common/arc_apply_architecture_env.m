function opts = arc_apply_architecture_env(opts)
%ARC_APPLY_ARCHITECTURE_ENV Configure architecture from SNN_* env variables.
%   ARC publication jobs default to the requested 32k-neuron low-rank
%   signed-uniform architecture. Environment variables remain available for
%   deliberate follow-up experiments, but the no-env ARC path is now explicit:
%       SNN_N_HIDDEN=32000
%       SNN_RECURRENT_MODE=low_rank
%       SNN_DECODER_MODE=signed
%       SNN_SIGNED_DECODER_DISTRIBUTION=uniform
if ~isfield(opts, 'arch') || isempty(opts.arch)
    opts.arch = struct();
end
opts.N_hidden = getenv_default_positive_int("SNN_N_HIDDEN", 32000);
opts.arch.recurrent_mode = getenv_default_string("SNN_RECURRENT_MODE", "low_rank");
opts.arch.decoder_mode = getenv_default_string("SNN_DECODER_MODE", "signed");
opts.arch.signed_decoder_distribution = getenv_default_string("SNN_SIGNED_DECODER_DISTRIBUTION", "uniform");
opts.arch.full_rank_p_rec = single(str2double(getenv_default_string("SNN_FULL_RANK_P_REC", "1.0")));
opts.arch.full_rank_remove_self_connections = getenv_default_logical("SNN_FULL_RANK_REMOVE_SELF_CONNECTIONS", true);
opts.arch.full_rank_storage = getenv_default_string("SNN_FULL_RANK_STORAGE", "auto");
opts.arch.full_rank_sparse_threshold = single(str2double(getenv_default_string("SNN_FULL_RANK_SPARSE_THRESHOLD", "0.10")));
opts.arch.max_dense_full_rank_N = int32(str2double(getenv_default_string("SNN_MAX_DENSE_FULL_RANK_N", "6000")));
opts.arch.max_sparse_full_rank_nnz = int64(str2double(getenv_default_string("SNN_MAX_SPARSE_FULL_RANK_NNZ", "20000000")));
opts.arch.max_full_rank_recurrent_bytes = double(str2double(getenv_default_string("SNN_MAX_FULL_RANK_RECURRENT_BYTES", num2str(2.5 * 2^30))));
end

function value = getenv_default_string(name, default_value)
value = string(getenv(char(name)));
if strlength(value) == 0
    value = string(default_value);
end
end

function value = getenv_default_logical(name, default_value)
raw = lower(getenv_default_string(name, string(default_value)));
value = any(raw == ["1", "true", "yes", "on"]);
if any(raw == ["0", "false", "no", "off"])
    value = false;
end
end

function value = getenv_default_positive_int(name, default_value)
raw = getenv_default_string(name, string(default_value));
numeric_value = str2double(raw);
if ~(isscalar(numeric_value) && isfinite(numeric_value) && numeric_value == round(numeric_value) && numeric_value > 0)
    error('arc_apply_architecture_env:badPositiveInteger', ...
        '%s must be a positive integer, got "%s".', char(name), char(raw));
end
value = numeric_value;
end
