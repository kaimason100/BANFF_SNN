% validate_gpu_dynamics_cell_lengths.m
function validate_gpu_dynamics_cell_lengths(x)
if ~iscell(x) || isempty(x)
    return;
end
lengths = cellfun(@(xx) size(xx,2), x);
if any(lengths ~= lengths(1))
    error('snn_primary_api:gpuVariableLengthDynamicsCells', ...
        ['GPU dynamics cell-array training currently requires equal sequence lengths because ', ...
         'the CUDA MEX is initialised for one fixed sequence length. Got lengths: %s. ', ...
         'Use CPU training or pad/split cells to a common length.'], mat2str(lengths));
end
end

