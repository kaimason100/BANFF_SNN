% clear_static_gpu.m
function clear_static_gpu(domain)
mex_name = static_mex_name(domain);
if exist(mex_name, 'file') == 3
    feval(mex_name, 'clear');
end
end

