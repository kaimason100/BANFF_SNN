function repo_root = arc_resolve_repo_root(start_dir)
%ARC_RESOLVE_REPO_ROOT Resolve the standalone ARC bundle for task wrappers.
%   When the bundle is nested in a larger local repository, this function
%   intentionally selects arc_ucalgary itself. ARC must execute the copied
%   bundle code, CUDA sources, data, and MEX binaries rather than shadowing
%   them with the local development implementation.

start_dir = char(start_dir);
candidates = {fileparts(start_dir), start_dir, fileparts(fileparts(start_dir)), pwd};
for depth = 0:8
    up = ascend_dir(start_dir, depth);
    if ~isempty(up)
        candidates{end+1} = up; %#ok<AGROW>
        candidates{end+1} = fileparts(up); %#ok<AGROW>
    end
end
candidates = unique(candidates, 'stable');
for ii = 1:numel(candidates)
    if is_full_repo_root(candidates{ii})
        repo_root = candidates{ii};
        addpath(repo_root, '-begin');
        return;
    end
end
for ii = 1:numel(candidates)
    repo_root = search_down_for_repo(candidates{ii}, 5);
    if ~isempty(repo_root)
        addpath(repo_root, '-begin');
        return;
    end
end
error('arc_resolve_repo_root:notFound', ...
    ['Could not locate the standalone ARC bundle from "%s". Need ', ...
     'shared/matlab/snn_primary_api.m plus train/, common/, Classification/, ', ...
     'Regression/, and dynamical_systems/.'], start_dir);
end

function tf = is_full_repo_root(path_name)
tf = exist(fullfile(path_name, 'shared', 'matlab', 'snn_primary_api.m'), 'file') == 2 && ...
    exist(fullfile(path_name, 'train'), 'dir') == 7 && ...
    exist(fullfile(path_name, 'common'), 'dir') == 7 && ...
    exist(fullfile(path_name, 'Classification'), 'dir') == 7 && ...
    exist(fullfile(path_name, 'Regression'), 'dir') == 7 && ...
    exist(fullfile(path_name, 'dynamical_systems'), 'dir') == 7;
end

function found = search_down_for_repo(root_dir, max_depth)
found = '';
if isempty(root_dir) || exist(root_dir, 'dir') ~= 7 || max_depth < 0
    return;
end
if is_full_repo_root(root_dir)
    found = root_dir;
    return;
end
listing = dir(root_dir);
for ii = 1:numel(listing)
    if ~listing(ii).isdir || any(strcmp(listing(ii).name, {'.','..'})) || startsWith(listing(ii).name, '.')
        continue;
    end
    found = search_down_for_repo(fullfile(root_dir, listing(ii).name), max_depth - 1);
    if ~isempty(found)
        return;
    end
end
end

function out = ascend_dir(in, n)
out = in;
for ii = 1:n
    parent = fileparts(out);
    if isempty(parent) || strcmp(parent, out)
        out = '';
        return;
    end
    out = parent;
end
end
