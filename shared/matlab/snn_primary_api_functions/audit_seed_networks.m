function [bias_by_seed, audit] = audit_seed_networks(source, seeds, model_files)
%AUDIT_SEED_NETWORKS Extract learned biases and verify distinct seed networks.
%   SOURCE may be a cell array of model file paths, a cell array of training
%   result structs, or a struct array of training results. Distinct seed values
%   must produce distinct fixed network hashes; trained biases are not included
%   in the network hash because they are the learned parameter being compared.

if nargin < 2 || isempty(seeds)
    seeds = 1:numel(source);
end
if nargin < 3
    model_files = repmat({''}, 1, numel(seeds));
end

n = numel(seeds);
bias_by_seed = repmat(empty_bias_record(), 1, n);
digests = strings(n, 1);
model_labels = strings(n, 1);
for ii = 1:n
    [train_result, model_label] = get_training_result_from_source(source, ii, model_files);
    if ~isfield(train_result, 'model') || ~isstruct(train_result.model)
        error('snn_primary_api:seedAuditMissingModel', ...
            ['Saved result for seed %g does not contain result.model. ', ...
             'Cannot verify that seed-specific fixed networks are distinct.'], double(seeds(ii)));
    end
    b = single(best_bias_from_result(train_result));
    bias_by_seed(ii) = make_bias_record(seeds(ii), b);
    digests(ii) = string(hash_fixed_network(train_result.model));
    model_labels(ii) = string(model_label);
end

audit = struct();
audit.status = "passed";
audit.n_seeds = int32(n);
audit.seed_list = seeds(:).';
audit.model_files = cellstr(model_labels);
audit.fixed_network_sha256 = cellstr(digests);
audit.compared_fields = fixed_network_fields();
audit.note = ['Hashes include fixed encoders, decoders, recurrent structure, ', ...
    'Dale signs and neuron/network dimensions. Learned biases and Adam state are excluded.'];

if n <= 1
    audit.status = "single_seed";
    fprintf('[SEED AUDIT] Single seed tested; no pairwise network-difference check required.\n');
    return;
end

if any(digests == "")
    error('snn_primary_api:seedAuditHashFailed', ...
        ['Could not compute a fixed-network hash for every seed. ', ...
         'Pairwise seed-difference checks were not run.']);
end

for ii = 1:n-1
    for jj = ii+1:n
        if isequal(seeds(ii), seeds(jj))
            continue;
        end
        if digests(ii) == digests(jj)
            error('snn_primary_api:seedNetworksIdentical', ...
                ['Fixed network hash is identical for distinct seeds %g and %g. ', ...
                 'This indicates the seed-specific networks are not actually different.'], ...
                double(seeds(ii)), double(seeds(jj)));
        end
    end
end
fprintf('[SEED AUDIT] Verified %d distinct fixed-network hash(es) across %d seed(s).\n', ...
    numel(unique(digests)), n);
end

function rec = empty_bias_record()
rec = struct('seed', NaN, 'n', int32(0), 'mean', NaN, 'sd', NaN, ...
    'min', NaN, 'max', NaN, 'values', single([]));
end

function rec = make_bias_record(seed_value, b)
b = single(b(:));
rec = empty_bias_record();
rec.seed = double(seed_value);
rec.n = int32(numel(b));
rec.values = b;
if isempty(b)
    return;
end
bd = double(b);
rec.mean = mean(bd, 'omitnan');
rec.sd = std(bd, 0, 'omitnan');
rec.min = min(bd, [], 'omitnan');
rec.max = max(bd, [], 'omitnan');
end

function [train_result, model_label] = get_training_result_from_source(source, ii, model_files)
model_label = '';
if iscell(source)
    item = source{ii};
else
    item = source(ii);
end

if ischar(item) || isstring(item)
    model_label = char(item);
    train_result = load_training_result(model_label);
    return;
end

train_result = item;
if nargin >= 3 && numel(model_files) >= ii
    model_label = char(model_files{ii});
end
if isempty(model_label) && isfield(train_result, 'model_file')
    model_label = char(train_result.model_file);
end
end

function fields = fixed_network_fields()
fields = {'N_in', 'N_hidden', 'N_out', 'N_rec', ...
    'recurrent_mode', 'decoder_mode', 'signed_decoder_distribution', ...
    'W_in', 'W_out_base_rec', 'W_out', 'Eta_rec', 'dself', 'dale_sign', ...
    'W_rec', 'W_rec_mask', 'rec_post_idx', 'rec_pre_idx', 'rec_w', 'rec_nnz', ...
    'INPUT_SCALE', 'SCALE_rec', 'dt', 'alpha', 'beta', 'gamma_sr', 'gamma_sd', ...
    'E_L', 'V_th', 'V_reset', 'a_eff', 'b_param', 'phi_u', 'delta_u', ...
    'spike_jump_sr'};
end

function hash = hash_fixed_network(model)
hash = '';
try
    if exist('javaMethod', 'builtin') ~= 5 && exist('javaMethod', 'file') ~= 2
        return;
    end
    engine = javaMethod('getInstance', 'java.security.MessageDigest', 'SHA-256');
    fields = fixed_network_fields();
    for ii = 1:numel(fields)
        name = fields{ii};
        if ~isfield(model, name)
            continue;
        end
        update_hash(engine, uint8(name));
        update_hash(engine, uint8(getByteStreamFromArray(model.(name))));
    end
    digest = typecast(engine.digest(), 'uint8');
    hash = lower(reshape(dec2hex(digest).', 1, []));
catch
    hash = '';
end
end

function update_hash(engine, bytes)
if isempty(bytes)
    bytes = uint8(0);
end
engine.update(uint8(bytes(:)));
end
