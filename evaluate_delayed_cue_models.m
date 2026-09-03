function report=evaluate_delayed_cue_models(varargin)
%EVALUATE_DELAYED_CUE_MODELS Public delayed-cue evaluation launcher.
root=fileparts(mfilename('fullpath'));
addpath(fullfile(root,'evaluation'));
report=banff_evaluate_delayed_cue(varargin{:});
end
