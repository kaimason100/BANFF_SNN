% Package orientation: Shared MATLAB utility used across tasks. Read this with the caller open so input/output structs and saved-result fields are clear.

function out = snn_primary_api(action, domain, backend, opts)
%SNN_PRIMARY_API Primary-decoder training, checking, and benchmarking.
%   This release-facing API keeps the public scripts on one task-decoder path.
%   W_out_base_rec provides recurrent feedback, while W_out maps filtered
%   spikes to task outputs and supplies the supervised bias-learning signal.
%   CPU routines are deliberately small and transparent; GPU routines call
%   the gpu MEX through primary-only
%   commands added for this layout. Internal helpers live in
%   shared/matlab/snn_primary_api_functions
%   as one function per file so the implementation can be audited function-by-function.

if nargin < 2 || isempty(domain), domain = 'regression'; end
if nargin < 3 || isempty(backend), backend = 'cpu'; end
if nargin < 4 || isempty(opts), opts = struct(); end

helper_dir = fullfile(fileparts(mfilename('fullpath')), 'snn_primary_api_functions');
if exist(helper_dir, 'dir') == 7 && ~contains(path, helper_dir)
    addpath(helper_dir);
end

action = lower(string(action));
domain = lower(string(domain));
backend = lower(string(backend));

switch action
    case "train_static"
        out = train_static(domain, backend, opts);
    case "train_dynamics"
        out = train_dynamics(backend, opts);
    case "test_static"
        out = test_static(domain, backend, opts);
    case "test_dynamics"
        out = test_dynamics(backend, opts);
    case "spike_diagnostics_static"
        out = spike_diagnostics_static(domain, opts);
    case "spike_diagnostics_dynamics"
        out = spike_diagnostics_dynamics(opts);
    case "analyse_dynamics_spike_jitter"
        % Optional Gaussian-jitter diagnostic, separate from the publication
        % rate-preserving within-window timing-shuffle analysis below.
        out = analyse_dynamics_spike_jitter(opts);
    case "analyse_dynamics_rate_coding"
        % The saved-event analysis preserves fractional spike times (rho).
        % Do not recreate events from integer simulation steps here.
        out = analyse_saved_dynamics_rate_coding(opts);
    case "check_static"
        out = check_static(domain, opts);
    case "check_dynamics"
        out = check_dynamics(opts);
    case "check_dynamics_pool_training"
        out = check_dynamics_pool_training(opts);
    case "check_dynamics_gpu_validation_state"
        out = check_dynamics_gpu_validation_state(opts);
    case "check_architecture_modes"
        out = check_architecture_modes_impl(opts);
    case "bench_static"
        out = bench_static(domain, opts);
    case "bench_dynamics"
        out = bench_dynamics(opts);
    otherwise
        error('snn_primary_api:action', 'Unknown action "%s".', action);
end
end
