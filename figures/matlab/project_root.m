% Package orientation: Shared MATLAB utility used across tasks. Read this with the caller open so input/output structs and saved-result fields are clear.

function root = project_root()
%PROJECT_ROOT Return the absolute path to the package root.

this_file = mfilename('fullpath');
root = fileparts(fileparts(fileparts(this_file)));
end
