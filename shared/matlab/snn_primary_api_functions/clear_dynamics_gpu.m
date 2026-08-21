% clear_dynamics_gpu.m
function clear_dynamics_gpu()
if exist('snn_time_loop_gpu_mex', 'file') == 3
    snn_time_loop_gpu_mex('clear');
end
end

