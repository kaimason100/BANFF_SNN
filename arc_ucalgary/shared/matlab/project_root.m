function root = project_root()
%PROJECT_ROOT Return the absolute path to the package root.

this_file = mfilename('fullpath');
root = fileparts(fileparts(fileparts(this_file)));
end
