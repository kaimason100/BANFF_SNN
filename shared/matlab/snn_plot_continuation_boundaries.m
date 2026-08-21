function boundaries = snn_plot_continuation_boundaries(ax, result, total_epochs)
%SNN_PLOT_CONTINUATION_BOUNDARIES Mark saved SPSA fine-tuning boundaries.
%   BOUNDARIES = SNN_PLOT_CONTINUATION_BOUNDARIES(AX, RESULT, N) draws one
%   dashed labelled line after each prior training phase in a cumulative
%   SPSA history. Pass AX=[] to return the boundaries without drawing them.

if nargin < 3 || isempty(total_epochs)
    total_epochs = inf;
end
boundaries = zeros(1, 0);
if ~(isstruct(result) && isfield(result, 'continuation') && isstruct(result.continuation))
    return;
end
C = result.continuation;
if isfield(C, 'boundaries') && ~isempty(C.boundaries)
    boundaries = double(C.boundaries(:).');
elseif isfield(C, 'enabled') && logical(C.enabled) && isfield(C, 'source_epochs')
    boundaries = double(C.source_epochs);
end
boundaries = unique(boundaries(isfinite(boundaries) & boundaries >= 1 & boundaries < total_epochs), 'stable');
if isempty(ax)
    return;
end
for ii = 1:numel(boundaries)
    h = xline(ax, boundaries(ii), '--', sprintf('Continuation %d', ii), ...
        'Color', [0.25 0.25 0.25], 'LineWidth', 1.0);
    h.HandleVisibility = 'off';
end
end
