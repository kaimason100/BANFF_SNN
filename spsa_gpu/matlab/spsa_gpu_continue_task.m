function result = spsa_gpu_continue_task(task_name, source_model_file, additional_epochs, varargin)
%SPSA_GPU_CONTINUE_TASK Continue a saved SPSA model with a new fine-tuning phase.
%   RESULT = SPSA_GPU_CONTINUE_TASK(TASK, SOURCE_FILE, ADDITIONAL_EPOCHS)
%   starts from the validation-selected model in SOURCE_FILE, resets Adam's
%   moments, and trains for ADDITIONAL_EPOCHS while retaining the terminal
%   SPSA perturbation and learning-rate values. The new result is written to
%   a sibling file ending in "_continued_<N>epochs.mat" unless ModelFile is
%   supplied in VARARGIN.
%
%   TASK is "bc", "yacht", or "vanderpol". Extra name/value options are
%   passed to SPSA_GPU_RUN_TASK, for example 'ModelFile', OUTPUT_FILE or
%   'RunChecks', false. This is a continuation phase, not an exact replay of
%   the original optimizer trajectory, because saved selected models do not
%   retain an Adam state corresponding to their selected bias vector.

if nargin < 3
    error('spsa_gpu:continuationArguments', ...
        'Task name, source model file, and additional epochs are required.');
end
result = spsa_gpu_run_task(task_name, 'ContinueFrom', source_model_file, ...
    'AdditionalEpochs', additional_epochs, varargin{:});
end
