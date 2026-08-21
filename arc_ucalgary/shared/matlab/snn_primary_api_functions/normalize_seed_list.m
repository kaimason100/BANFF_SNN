% normalize_seed_list.m
function seeds = normalize_seed_list(seeds, n_files)
seeds = double(seeds(:).');
if isempty(seeds)
    seeds = 1:n_files;
end
if numel(seeds) ~= n_files
    error('snn_primary_api:seedFileMismatch', 'opts.seed_list has %d entries but opts.model_files has %d files.', numel(seeds), n_files);
end
end

