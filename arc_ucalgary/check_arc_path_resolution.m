function report = check_arc_path_resolution()
%CHECK_ARC_PATH_RESOLUTION Dry-run path smoke test for standalone ARC bundle.
%   Run from MATLAB inside the transferred arc_ucalgary folder:
%       report = check_arc_path_resolution()

root = fileparts(mfilename('fullpath'));
addpath(root, '-begin');
addpath(fullfile(root, 'shared', 'matlab'), '-begin');
added = add_project_paths(root);

required = {'snn_primary_api', 'make_dynamical_system', 'setup_project_paths', 'add_project_paths'};
report = struct();
report.root = root;
report.added_path_count = numel(added);
for ii = 1:numel(required)
    report.(required{ii}) = which(required{ii});
    if isempty(report.(required{ii}))
        error('check_arc_path_resolution:missingFunction', ...
            'MATLAB could not resolve required function "%s" from standalone ARC root.', required{ii});
    end
end

api_file = report.snn_primary_api;
if ~contains(api_file, fullfile('shared', 'matlab'))
    error('check_arc_path_resolution:badAPIPath', ...
        'snn_primary_api resolved to an unexpected location: %s', api_file);
end
dyn_file = report.make_dynamical_system;
if ~contains(dyn_file, fullfile('dynamical_systems', 'utilities'))
    error('check_arc_path_resolution:badDynamicsPath', ...
        'make_dynamical_system resolved to an unexpected location: %s', dyn_file);
end

fprintf('[ARC path check] root: %s%s', root, newline);
fprintf('[ARC path check] snn_primary_api: %s%s', report.snn_primary_api, newline);
fprintf('[ARC path check] make_dynamical_system: %s%s', report.make_dynamical_system, newline);
fprintf('[ARC path check] setup_project_paths: %s%s', report.setup_project_paths, newline);
fprintf('[ARC path check] add_project_paths: %s%s', report.add_project_paths, newline);
fprintf('[ARC path check] passed with %d added path(s).%s', report.added_path_count, newline);
end
