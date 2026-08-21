% static_mex_name.m
function name = static_mex_name(domain)
if domain == "classification"
    name = 'snn_classify_time_loop_gpu_mex';
else
    name = 'snn_regress_time_loop_gpu_mex';
end
end

