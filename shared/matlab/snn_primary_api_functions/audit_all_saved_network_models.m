function report = audit_all_saved_network_models(model_dir)
%AUDIT_ALL_SAVED_NETWORK_MODELS Audit every saved network across tasks and seeds.
%   REPORT = AUDIT_ALL_SAVED_NETWORKS() loads model MAT files from
%   outputs/models, excludes checkpoint files, and checks two invariants:
%   distinct seeds within a comparable task/model variant must have different
%   fixed-network hashes; matching seeds across tasks are compared only when
%   recurrent size, recurrent mode, decoder mode, storage, and dt agree.

if nargin < 1 || isempty(model_dir)
    model_dir = fullfile(project_root(), 'outputs', 'models');
end
files = dir(fullfile(model_dir, '*.mat'));
files = files(~contains({files.name}, '_checkpoint_'));
records = repmat(empty_record(), 0, 1);
for ii = 1:numel(files)
    path_name = fullfile(files(ii).folder, files(ii).name);
    try
        R = load_training_result(path_name);
    catch ME
        warning('audit_all_saved_networks:unreadable', 'Skipping %s: %s', files(ii).name, ME.message);
        continue;
    end
    if ~isfield(R, 'model') || ~isstruct(R.model), continue; end
    rec = empty_record(); rec.file = path_name; rec.name = files(ii).name;
    rec.task = task_from_name(files(ii).name); rec.seed = seed_from_name(files(ii).name, R);
    rec.family = model_family(R.model); rec.model = R.model;
    records(end + 1) = rec; %#ok<AGROW>
end
if isempty(records), error('audit_all_saved_networks:noModels', 'No readable saved model files were found.'); end

within = struct('task', {}, 'family', {}, 'seeds', {}, 'status', {}, 'hashes', {}, 'files', {});
groups = unique(string({records.task}) + "|" + string({records.family}));
for gg = 1:numel(groups)
    idx = find(string({records.task}) + "|" + string({records.family}) == groups(gg));
    seeds = [records(idx).seed]; unique_seeds = unique(seeds(isfinite(seeds)));
    hashes = strings(1, numel(idx));
    for kk = 1:numel(idx), hashes(kk) = string(fixed_hash(records(idx(kk)).model)); end
    status = "single_seed";
    if numel(unique_seeds) > 1
        status = "passed";
        for aa = 1:numel(unique_seeds)
            same_seed = idx(seeds == unique_seeds(aa));
            if numel(same_seed) > 1
                % Continuations may duplicate a seed; require their fixed network to agree.
                h = hashes(seeds == unique_seeds(aa));
                if numel(unique(h)) ~= 1, status = "failed_duplicate_seed_mismatch"; end
            end
        end
        for aa = 1:numel(unique_seeds)-1
            if any(hashes(seeds == unique_seeds(aa)) == hashes(seeds == unique_seeds(aa+1)))
                status = "failed_distinct_seeds_identical";
            end
        end
    end
    [task, family] = split(groups(gg), "|");
    within(end+1) = struct('task', task, 'family', family, 'seeds', unique_seeds, ...
        'status', status, 'hashes', hashes, 'files', { {records(idx).name} }); %#ok<AGROW>
end

cross = struct('seed', {}, 'task_a', {}, 'task_b', {}, 'status', {}, 'details', {});
all_seeds = unique([records.seed]); all_seeds = all_seeds(isfinite(all_seeds));
for ss = all_seeds
    idx = find([records.seed] == ss);
    for aa = 1:numel(idx)-1
        for bb = aa+1:numel(idx)
            A = records(idx(aa)); B = records(idx(bb));
            if strcmp(A.task, B.task) || ~strcmp(A.family, B.family), continue; end
            [status, details] = compare_common_fixed_fields(A.model, B.model);
            cross(end+1) = struct('seed', ss, 'task_a', string(A.name), 'task_b', string(B.name), ...
                'status', status, 'details', details); %#ok<AGROW>
        end
    end
end

report = struct('model_dir', model_dir, 'records', records, 'within_task', within, 'cross_task', cross);
fprintf('[NETWORK AUDIT] %d saved model files checked: %d within-task group(s), %d cross-task seed comparison(s).\n', ...
    numel(records), numel(within), numel(cross));
fprintf('[NETWORK AUDIT] Within-task failures: %d; cross-task failures: %d.\n', ...
    sum(startsWith(string({within.status}), "failed")), sum(startsWith(string({cross.status}), "failed")));
end

function rec = empty_record()
rec = struct('file', '', 'name', '', 'task', '', 'seed', NaN, 'family', '', 'model', struct());
end

function task = task_from_name(name)
task = regexprep(name, '(_lowrank_SPSA_GPU|_full_rank6k_gpu|_gpu).*$', '');
end

function seed = seed_from_name(name, R)
token = regexp(name, '_seed(\d+)', 'tokens', 'once');
if ~isempty(token), seed = str2double(token{1});
elseif isfield(R, 'init_seed'), seed = double(R.init_seed);
elseif isfield(R, 'options') && isfield(R.options, 'init_seed'), seed = double(R.options.init_seed);
else, seed = NaN; end
end

function family = model_family(P)
fields = {'N_hidden','N_rec','recurrent_mode','decoder_mode','recurrent_storage','dt'};
parts = strings(1, numel(fields));
for ii = 1:numel(fields)
    if isfield(P, fields{ii}), parts(ii) = string(fields{ii}) + "=" + string(P.(fields{ii})); end
end
family = strjoin(parts, ';');
end

function hash = fixed_hash(P)
fields = {'W_in','W_out_base_rec','W_out','Eta_rec','dself','dale_sign','W_rec','W_rec_mask','rec_post_idx','rec_pre_idx','rec_w'};
hash = "";
try
    engine = javaMethod('getInstance', 'java.security.MessageDigest', 'SHA-256');
    for ii = 1:numel(fields)
        if isfield(P, fields{ii}), engine.update(uint8(getByteStreamFromArray(P.(fields{ii})))); end
    end
    hash = lower(reshape(dec2hex(typecast(engine.digest(), 'uint8')).', 1, []));
catch
    hash = string(sum(double(P.W_in(:)), 'omitnan'));
end
end

function [status, details] = compare_common_fixed_fields(A, B)
fields = {'W_in','W_out_base_rec','W_out','Eta_rec','dself','dale_sign'};
details = strings(0,1); status = "passed";
for ii = 1:numel(fields)
    f = fields{ii}; if ~isfield(A,f) || ~isfield(B,f), continue; end
    a = A.(f); b = B.(f); sz = min(size(a), size(b));
    if isempty(sz) || any(sz == 0), continue; end
    subs = arrayfun(@(n) 1:n, sz, 'UniformOutput', false);
    if ~isequaln(a(subs{:}), b(subs{:}))
        status = "failed_common_fixed_parameters_differ";
        details(end+1,1) = string(f); %#ok<AGROW>
    end
end
if status == "passed", details = "common fixed parameters agree"; end
end
