function modes = architecture_comparison_modes(decoder_distribution_mode, base_arch)
%ARCHITECTURE_COMPARISON_MODES Architecture ablations used by scripts.
%   decoder_distribution_mode controls signed-decoder rows:
%     "both" or "all" => Gaussian and uniform signed modes
%     "gaussian"     => Gaussian signed modes only
%     "uniform"      => uniform signed modes only
%   base_arch carries caller-selected full-rank sparsity/storage controls.
if nargin >= 1 && isstruct(decoder_distribution_mode)
    if isfield(decoder_distribution_mode, 'arch') && isstruct(decoder_distribution_mode.arch)
        base_arch = decoder_distribution_mode.arch;
        decoder_distribution_mode = get_decoder_distribution_mode(decoder_distribution_mode);
    else
        base_arch = decoder_distribution_mode;
        decoder_distribution_mode = "both";
    end
end
if nargin < 1 || isempty(decoder_distribution_mode)
    decoder_distribution_mode = "both";
end
if nargin < 2 || isempty(base_arch)
    base_arch = default_arch_options();
else
    base_arch = merge_struct(default_arch_options(), base_arch);
end
decoder_distribution_mode = lower(string(decoder_distribution_mode));
if decoder_distribution_mode == "normal"
    decoder_distribution_mode = "gaussian";
end
if decoder_distribution_mode == "all"
    decoder_distribution_mode = "both";
end
if ~any(decoder_distribution_mode == ["both", "gaussian", "uniform"])
    error('architecture_comparison_modes:decoderDistributionMode', ...
        'decoder_distribution_mode must be "both", "gaussian" or "uniform", got "%s".', ...
        char(decoder_distribution_mode));
end

base_arch.recurrent_mode = "low_rank";
base_arch.decoder_mode = "shared";
base_arch.signed_decoder_distribution = "gaussian";
modes = repmat(struct('name', "", 'arch', struct()), 0, 1);

modes = append_mode(modes, "low_rank_shared", base_arch);

if any(decoder_distribution_mode == ["both", "gaussian"])
    arch = base_arch;
    arch.decoder_mode = "signed";
    arch.signed_decoder_distribution = "gaussian";
    modes = append_mode(modes, "low_rank_signed_gaussian", arch);
end
if any(decoder_distribution_mode == ["both", "uniform"])
    arch = base_arch;
    arch.decoder_mode = "signed";
    arch.signed_decoder_distribution = "uniform";
    modes = append_mode(modes, "low_rank_signed_uniform", arch);
end

arch = base_arch;
arch.recurrent_mode = "full_rank";
modes = append_mode(modes, "full_rank_shared", arch);

if any(decoder_distribution_mode == ["both", "gaussian"])
    arch = base_arch;
    arch.recurrent_mode = "full_rank";
    arch.decoder_mode = "signed";
    arch.signed_decoder_distribution = "gaussian";
    modes = append_mode(modes, "full_rank_signed_gaussian", arch);
end
if any(decoder_distribution_mode == ["both", "uniform"])
    arch = base_arch;
    arch.recurrent_mode = "full_rank";
    arch.decoder_mode = "signed";
    arch.signed_decoder_distribution = "uniform";
    modes = append_mode(modes, "full_rank_signed_uniform", arch);
end
end

function modes = append_mode(modes, name, arch)
next = numel(modes) + 1;
modes(next, 1).name = name;
modes(next, 1).arch = arch;
end

function mode = get_decoder_distribution_mode(opts)
mode = "both";
if isfield(opts, 'architecture_decoder_distributions')
    mode = string(opts.architecture_decoder_distributions);
elseif isfield(opts, 'arch') && isstruct(opts.arch) && ...
        isfield(opts.arch, 'signed_decoder_distribution_comparison')
    mode = string(opts.arch.signed_decoder_distribution_comparison);
end
end
