% summarize_benchmark.m
function out = summarize_benchmark(cpu_times, gpu_times)
out = struct();
out.cpu_seconds = cpu_times;
out.gpu_seconds = gpu_times;
out.cpu_median_seconds = median(cpu_times);
out.gpu_median_seconds = median(gpu_times);
out.speedup = out.cpu_median_seconds / max(out.gpu_median_seconds, eps);
end

