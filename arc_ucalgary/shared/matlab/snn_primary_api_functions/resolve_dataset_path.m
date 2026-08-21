% resolve_dataset_path.m
function file_name = resolve_dataset_path(explicit_file, default_name, alias_names)
if nargin < 3 || isempty(alias_names)
    alias_names = {};
end
if ischar(alias_names) || isstring(alias_names)
    alias_names = cellstr(string(alias_names(:)));
end
if ~isempty(explicit_file) && exist(explicit_file, 'file') == 2
    file_name = char(explicit_file);
    return;
end
root_dir = project_root();
names = [{default_name}, alias_names(:).'];
candidates = {};
candidate_names = {};
for nn = 1:numel(names)
    name = char(names{nn});
    candidates = [candidates; { ...
        fullfile(root_dir, 'data', 'raw', name)
        fullfile(root_dir, 'data', 'external', name)
        fullfile(root_dir, 'data', name)
        fullfile(pwd, name)}]; %#ok<AGROW>
    candidate_names = [candidate_names; repmat({name}, 4, 1)]; %#ok<AGROW>
end
if ~isempty(explicit_file), candidates{end+1} = char(explicit_file); end %#ok<AGROW>
if ~isempty(explicit_file), candidate_names{end+1} = char(explicit_file); end %#ok<AGROW>
for ii = 1:numel(candidates)
    if exist(candidates{ii}, 'file') == 2
        file_name = candidates{ii};
        if ~strcmp(candidate_names{ii}, default_name)
            warning('snn_primary_api:datasetAliasFallback', ...
                'Using dataset alias "%s" for canonical dataset "%s". Rename the file when possible.', ...
                candidate_names{ii}, default_name);
        end
        return;
    end
end
error('snn_primary_api:dataMissing', 'Could not find dataset "%s".', default_name);
end
