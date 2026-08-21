% mex_runtime_metadata.m
function meta = mex_runtime_metadata(backend, domain)
meta = struct();
meta.backend = char(backend);
meta.domain = char(domain);
if string(backend) ~= "gpu"
    meta.mex_name = '';
    return;
end
if any(strcmpi(char(domain), {'dynamics','dynamical_systems'}))
    mex_name = 'snn_time_loop_gpu_mex';
else
    mex_name = static_mex_name(lower(string(domain)));
end
source_file = mex_source_file(domain);
mex_file = which(mex_name);
meta.mex_name = mex_name;
meta.mex_file = mex_file;
meta.source_file = source_file;
meta.matlab_version = version;
meta.platform = computer;
if exist(source_file, 'file') == 2
    src_info = dir(source_file);
    meta.source_timestamp = src_info.date;
    meta.source_sha256 = file_sha256(source_file);
else
    meta.source_timestamp = '';
    meta.source_sha256 = '';
end
if ~isempty(mex_file) && exist(mex_file, 'file') == 3
    mex_info = dir(mex_file);
    meta.mex_timestamp = mex_info.date;
else
    meta.mex_timestamp = '';
end
meta.freshness_check = 'timestamp';
meta.hash_check_note = ['Source SHA-256 is recorded for reproducibility, but current MEX files do not expose ', ...
    'a compiled build_hash command; runtime freshness enforcement is timestamp-based.'];
try
    g = gpuDevice;
    meta.gpu_name = g.Name;
    meta.gpu_compute_capability = g.ComputeCapability;
catch
    meta.gpu_name = '';
    meta.gpu_compute_capability = '';
end
end

