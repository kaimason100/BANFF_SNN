% Package orientation: Shared MATLAB utility used across tasks. Read this with the caller open so input/output structs and saved-result fields are clear.

function result = check_architecture_modes(opts)
%CHECK_ARCHITECTURE_MODES Public quick sanity check for architecture modes.
if nargin < 1
    opts = struct();
end
result = snn_primary_api('check_architecture_modes', 'classification', 'cpu', opts);
end
