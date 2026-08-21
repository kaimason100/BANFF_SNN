function utility_file = ensure_dynamics_utility_on_path(function_name)
%ENSURE_DYNAMICS_UTILITY_ON_PATH Prefer the .m dynamics utility explicitly.
%   MATLAB can be fragile when checking a bare function name if matching
%   .m and .mlx files both exist. Resolve the required .m file by path and
%   put its folder first before the caller invokes the function.

if nargin < 1 || isempty(function_name)
    error('snn_primary_api:missingUtilityName', 'A dynamics utility function name is required.');
end

function_name = char(function_name);
this_file = mfilename('fullpath');
root_dir = fileparts(fileparts(fileparts(fileparts(this_file))));
utility_file = fullfile(root_dir, 'dynamical_systems', 'utilities', [function_name '.m']);

if exist(utility_file, 'file') ~= 2
    error('snn_primary_api:dynamicsUtilityMissing', ...
        'Could not find required dynamics utility file: %s', utility_file);
end

addpath(fileparts(utility_file), '-begin');
end
