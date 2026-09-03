function varargout = evaluate_saved_lorenz_models(varargin)
%EVALUATE_SAVED_LORENZ_MODELS Public launcher for explicit local model files.
%   EVALUATE_SAVED_LORENZ_MODELS opens a multi-file selector and performs the
%   complete standard held-out Lorenz evaluation for the chosen trained models.
%   Paths and display options may alternatively be supplied as documented by
%   BANFF_EVALUATE_SAVED_LORENZ.

root = fileparts(mfilename('fullpath'));
evaluationDirectory = fullfile(root,'evaluation');
addpath(evaluationDirectory);
[varargout{1:nargout}] = banff_evaluate_saved_lorenz(varargin{:});
end
