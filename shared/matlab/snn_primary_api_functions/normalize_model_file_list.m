% normalize_model_file_list.m
function files = normalize_model_file_list(model_files)
if isstring(model_files)
    files = cellstr(model_files(:));
elseif ischar(model_files)
    files = {model_files};
elseif iscell(model_files)
    files = cell(size(model_files(:)));
    for ii = 1:numel(model_files)
        files{ii} = char(model_files{ii});
    end
else
    error('snn_primary_api:modelFilesType', 'opts.model_files must be a string array, character vector, or cell array of paths.');
end
files = files(:).';
files = files(~cellfun(@isempty, files));
if isempty(files)
    error('snn_primary_api:modelFilesEmpty', 'opts.model_files must contain at least one saved model path.');
end
end

