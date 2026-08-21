function benchmark = run_architecture_runtime_benchmark(task_kind, domain, backend, opts, output_file)
%RUN_ARCHITECTURE_RUNTIME_BENCHMARK Time short training runs for three modes.
if nargin < 5
    output_file = "";
end
task_kind = lower(string(task_kind));
domain = lower(string(domain));
backend = lower(string(backend));
decoder_distribution_mode = architecture_decoder_distribution_mode(opts);
base_arch = getfield_with_default(opts, 'arch', default_arch_options());
modes = architecture_comparison_modes(decoder_distribution_mode, base_arch);
names = strings(numel(modes), 1);
elapsed = nan(numel(modes), 1);
loss = nan(numel(modes), 1);
status = strings(numel(modes), 1);

for ii = 1:numel(modes)
    one_opts = opts;
    one_opts.arch = modes(ii).arch;
    one_opts.epochs = max(1, round(getfield_with_default(one_opts, 'epochs', 1)));
    one_opts.live_plot = struct('enable', false, 'every', 1);
    names(ii) = modes(ii).name;
    try
        t0 = tic;
        switch task_kind
            case "static"
                result = snn_primary_api('train_static', char(domain), char(backend), one_opts);
                if isfield(result, 'history') && isfield(result.history, 'train_loss')
                    loss(ii) = first_finite_local(result.history.train_loss);
                end
            case "dynamics"
                result = snn_primary_api('train_dynamics', 'dynamical_systems', char(backend), one_opts);
                if isfield(result, 'history')
                    loss(ii) = first_finite_local(result.history);
                end
            otherwise
                error('run_architecture_runtime_benchmark:taskKind', ...
                    'task_kind must be "static" or "dynamics".');
        end
        elapsed(ii) = toc(t0);
        status(ii) = "ran";
    catch ME
        elapsed(ii) = NaN;
        loss(ii) = NaN;
        status(ii) = string(ME.message);
    end
end

benchmark = struct();
benchmark.task_kind = char(task_kind);
benchmark.domain = char(domain);
benchmark.backend = char(backend);
benchmark.options = opts;
benchmark.decoder_distribution_mode = decoder_distribution_mode;
benchmark.summary = table(names, elapsed, loss, status, ...
    'VariableNames', {'Architecture','ElapsedSeconds','Loss','Status'});

if strlength(string(output_file)) > 0
    output_dir = fileparts(char(output_file));
    if ~isempty(output_dir) && exist(output_dir, 'dir') ~= 7
        mkdir(output_dir);
    end
    save(char(output_file), 'benchmark', '-v7.3');
end
end

function mode = architecture_decoder_distribution_mode(opts)
mode = "both";
if isstruct(opts) && isfield(opts, 'architecture_decoder_distributions')
    mode = string(opts.architecture_decoder_distributions);
elseif isstruct(opts) && isfield(opts, 'arch') && isstruct(opts.arch) && ...
        isfield(opts.arch, 'signed_decoder_distribution_comparison')
    mode = string(opts.arch.signed_decoder_distribution_comparison);
end
mode = lower(mode);
if mode == "normal"
    mode = "gaussian";
elseif mode == "all"
    mode = "both";
end
end

function value = getfield_with_default(s, key, default_value)
if isstruct(s) && isfield(s, key)
    value = s.(key);
else
    value = default_value;
end
end

function value = first_finite_local(x)
x = double(x(:));
idx = find(isfinite(x), 1, 'first');
if isempty(idx)
    value = NaN;
else
    value = x(idx);
end
end
