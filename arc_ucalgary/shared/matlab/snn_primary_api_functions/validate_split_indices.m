% validate_split_indices.m
function validate_split_indices(data, context)
keys = {'idx_train','idx_val','idx_test'};
all_idx = [];
for ii = 1:numel(keys)
    if ~isfield(data, keys{ii}) || isempty(data.(keys{ii}))
        error('snn_primary_api:missingSplitIndices', '%s is missing %s.', context, keys{ii});
    end
    idx = double(data.(keys{ii})(:));
    if any(~isfinite(idx)) || any(idx < 1) || any(idx ~= floor(idx))
        error('snn_primary_api:badSplitIndices', '%s has invalid %s values.', context, keys{ii});
    end
    all_idx = [all_idx; idx]; %#ok<AGROW>
end
if numel(unique(all_idx)) ~= numel(all_idx)
    error('snn_primary_api:overlapSplitIndices', '%s train/validation/test indices overlap.', context);
end
end

