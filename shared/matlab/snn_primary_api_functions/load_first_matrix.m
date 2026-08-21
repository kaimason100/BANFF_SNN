% load_first_matrix.m
function M = load_first_matrix(file_name)
if exist(file_name, 'file') ~= 2
    try
        file_name = resolve_dataset_path(file_name, file_name);
    catch
        % Keep the original name so the final error reports what was requested.
    end
end
if exist(file_name, 'file') ~= 2
    error('snn_primary_api:dataMissing', 'Could not find dataset "%s" on the MATLAB path.', file_name);
end
S = load(file_name);
if isfield(S, 'data')
    M = S.data;
else
    f = fieldnames(S);
    best_field = '';
    best_elements = 0;
    for ii = 1:numel(f)
        candidate = S.(f{ii});
        if istable(candidate)
            candidate_size = size(candidate);
            candidate_elements = prod(candidate_size);
            is_data_candidate = candidate_elements > 1;
        else
            is_data_candidate = isnumeric(candidate) && ismatrix(candidate) && numel(candidate) > 1;
            candidate_elements = numel(candidate);
        end
        if is_data_candidate && candidate_elements > best_elements
            best_field = f{ii};
            best_elements = candidate_elements;
        end
    end
    if isempty(best_field)
        error('Dataset "%s" must contain variable data or at least one non-scalar numeric/table matrix.', file_name);
    end
    M = S.(best_field);
end
if istable(M), M = table2array(M); end
if ~isnumeric(M), error('Loaded dataset must be numeric or a table.'); end
M = single(M);
end

