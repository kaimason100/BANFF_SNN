% save_seed_training_result.m
function seed_file = save_seed_training_result(seed_result, model_file, seed)
%SAVE_SEED_TRAINING_RESULT Save one trained network per initialization seed.
%   The aggregate script result is still returned to the caller, but each
%   seed also gets its own .mat file for later seed-wise testing/statistics.
[folder, base, ext] = fileparts(char(model_file));
if isempty(ext), ext = '.mat'; end
if isempty(folder), folder = pwd; end
if exist(folder, 'dir') ~= 7, mkdir(folder); end
seed_tag = sprintf('seed%03d', round(double(seed)));
seed_file = fullfile(folder, sprintf('%s_%s%s', base, seed_tag, ext));
result = seed_result; %#ok<NASGU>
result.model_file = seed_file;
save(seed_file, 'result', '-v7.3');
fprintf('[seed %s] saved trained network: %s%s', seed_tag, seed_file, newline);
end

