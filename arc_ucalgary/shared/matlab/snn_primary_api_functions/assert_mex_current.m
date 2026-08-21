% assert_mex_current.m
function assert_mex_current(mex_name, source_file)
if isempty(source_file) || exist(source_file, 'file') ~= 2
    return;
end
mex_file = which(mex_name);
if isempty(mex_file)
    return;
end
src_info = dir(source_file);
mex_info = dir(mex_file);
if isempty(src_info) || isempty(mex_info)
    return;
end
if src_info.datenum > mex_info.datenum
    error('snn_primary_api:staleMex', ...
        ['GPU MEX "%s" is older than its CUDA source. Recompile the MEX before ', ...
         'running local GPU training/testing. Source: %s | MEX: %s'], mex_name, source_file, mex_file);
end
end

