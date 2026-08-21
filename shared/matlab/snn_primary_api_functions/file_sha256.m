% file_sha256.m
function hash = file_sha256(file_name)
hash = '';
try
    engine = javaMethod('getInstance', 'java.security.MessageDigest', 'SHA-256');
    fid = fopen(file_name, 'r');
    if fid < 0, return; end
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    while true
        bytes = fread(fid, 1024*1024, '*uint8');
        if isempty(bytes), break; end
        engine.update(bytes);
    end
    digest = typecast(engine.digest(), 'uint8');
    hash = lower(reshape(dec2hex(digest).', 1, []));
catch
    hash = '';
end
end

