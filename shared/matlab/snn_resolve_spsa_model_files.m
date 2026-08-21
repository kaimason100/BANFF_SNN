% Package orientation: Shared MATLAB utility used by dedicated SPSA test scripts.

function [model_files, model_file] = snn_resolve_spsa_model_files(repo_root, task_stem, test_seeds, diagnostic_seed_index)
%SNN_RESOLVE_SPSA_MODEL_FILES Resolve terminal SPSA continuation chains.
%   Each requested seed starts at its original SPSA model. Saved continuation
%   files are linked through their recorded source_file field.
%   The final file of the longest cumulative branch is selected
%   automatically. This permits several alternative continuation attempts
%   from the same saved model without incorrectly concatenating their
%   histories. A warning identifies every branch selected automatically.
%   The selected file already has cumulative history, and the resolver
%   reports its total epochs and continuation boundaries for subsequent
%   training-history plots.

if nargin < 4 || isempty(diagnostic_seed_index)
    diagnostic_seed_index = 1;
end
repo_root = char(repo_root);
task_stem = char(task_stem);
test_seeds = double(test_seeds(:).');
if isempty(test_seeds) || any(~isfinite(test_seeds)) || any(test_seeds ~= round(test_seeds))
    error('snn_resolve_spsa_model_files:badSeeds', 'test_seeds must contain finite integer seed values.');
end
diagnostic_seed_index = max(1, min(numel(test_seeds), round(double(diagnostic_seed_index))));
base_files = arrayfun(@(seed) fullfile(repo_root, 'outputs', 'models', ...
    sprintf('%s_lowrank_SPSA_GPU_primary_seed%03d.mat', task_stem, seed)), ...
    test_seeds, 'UniformOutput', false);
missing = base_files(cellfun(@(path) exist(path, 'file') ~= 2, base_files));
if ~isempty(missing)
    error('snn_resolve_spsa_model_files:missingModels', ...
        'Saved SPSA model(s) were not found. Run the SPSA training for the requested seeds. First missing file: %s', ...
        missing{1});
end
model_files = cell(size(base_files));
for ii = 1:numel(base_files)
    [model_files{ii}, total_epochs, boundaries] = resolve_terminal_chain(base_files{ii});
    if ~strcmp(model_files{ii}, base_files{ii})
        fprintf('[SPSA continuation] seed %g: testing the terminal model after %d total epochs', ...
            test_seeds(ii), total_epochs);
        if ~isempty(boundaries)
            fprintf(' (phase boundaries after epoch(s): %s)', num2str(boundaries));
        end
        fprintf('.%s', newline);
    end
end
model_file = model_files{diagnostic_seed_index};
end

function [terminal_file, total_epochs, boundaries] = resolve_terminal_chain(base_file)
% Follow source-file links rather than timestamp order so only genuine
% continuations of this seed model are considered. Independent children of
% one source are alternative training runs, never sequential phases.
model_dir = fileparts(base_file);
[~, base_name, ~] = fileparts(base_file);
listing = dir(fullfile(model_dir, [base_name '_continued_*epochs*.mat']));
if isempty(listing)
    continuation_files = cell(0, 1);
else
    continuation_files = fullfile({listing.folder}, {listing.name});
end
candidate_files = [{base_file}; continuation_files(:)];
source_names = strings(numel(candidate_files), 1);
candidate_names = strings(numel(candidate_files), 1);
results = cell(numel(candidate_files), 1);
history_lengths = zeros(numel(candidate_files), 1);
for ii = 1:numel(candidate_files)
    results{ii} = load_training_result(candidate_files{ii});
    candidate_names(ii) = string(file_name_only(candidate_files{ii}));
    history_lengths(ii) = history_length(results{ii});
    if ii > 1 && isfield(results{ii}, 'continuation') && isstruct(results{ii}.continuation)
        source_names(ii) = string(file_name_only(continuation_field(results{ii}.continuation, 'source_file', '')));
    end
end

[terminal_index, total_epochs] = select_terminal_branch(1, false(numel(candidate_files), 1));
terminal_file = candidate_files{terminal_index};
terminal_result = results{terminal_index};
boundaries = continuation_boundaries(terminal_result, total_epochs);

    function [terminal_index, terminal_epochs] = select_terminal_branch(current_index, visited)
        % Select the deepest usable descendant by its cumulative history.
        % Filename ordering breaks an exact epoch-count tie reproducibly.
        if visited(current_index)
            error('snn_resolve_spsa_model_files:continuationCycle', ...
                'Continuation metadata contains a source-file cycle involving "%s".', ...
                char(candidate_names(current_index)));
        end
        visited(current_index) = true;
        children = find(source_names == candidate_names(current_index));
        if isempty(children)
            terminal_index = current_index;
            terminal_epochs = history_lengths(current_index);
            return;
        end

        child_terminals = zeros(numel(children), 1);
        child_epochs = zeros(numel(children), 1);
        for jj = 1:numel(children)
            [child_terminals(jj), child_epochs(jj)] = select_terminal_branch(children(jj), visited);
        end
        max_epochs = max(child_epochs);
        contenders = find(child_epochs == max_epochs);
        if numel(contenders) == 1
            selected = contenders;
        else
            [~, name_order] = sort(candidate_names(child_terminals(contenders)));
            selected = contenders(name_order(1));
        end
        terminal_index = child_terminals(selected);
        terminal_epochs = child_epochs(selected);

        if numel(children) > 1
            alternatives = candidate_names(child_terminals);
            alternatives(selected) = [];
            warning('snn_resolve_spsa_model_files:branchedContinuationSelected', ...
                ['Multiple SPSA continuation branches descend from "%s". Selected "%s" because it has ', ...
                 'the longest cumulative history (%d epochs); other branches are treated as alternatives: %s.'], ...
                char(candidate_names(current_index)), char(candidate_names(terminal_index)), terminal_epochs, ...
                char(strjoin(alternatives, ', ')));
        end
    end
end

function name = file_name_only(path_name)
if isempty(path_name)
    name = '';
    return;
end
[~, base, ext] = fileparts(char(path_name));
name = [base ext];
end

function n = history_length(result)
if ~isfield(result, 'history')
    n = 0;
elseif isnumeric(result.history)
    n = numel(result.history);
elseif isstruct(result.history)
    fields = fieldnames(result.history);
    if isempty(fields)
        n = 0;
    else
        n = numel(result.history.(fields{1}));
    end
else
    n = 0;
end
end

function boundaries = continuation_boundaries(result, total_epochs)
boundaries = zeros(1, 0);
if isfield(result, 'continuation') && isstruct(result.continuation)
    boundaries = double(continuation_field(result.continuation, 'boundaries', []));
    if isempty(boundaries) && logical(continuation_field(result.continuation, 'enabled', false))
        boundaries = double(continuation_field(result.continuation, 'source_epochs', []));
    end
end
boundaries = unique(boundaries(isfinite(boundaries) & boundaries >= 1 & boundaries < total_epochs), 'stable');
end

function value = continuation_field(S, name, default_value)
if isstruct(S) && isfield(S, name) && ~isempty(S.(name))
    value = S.(name);
else
    value = default_value;
end
end
