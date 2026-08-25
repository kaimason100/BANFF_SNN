function analysis = load_publication_analysis_file(file_path)
%LOAD_PUBLICATION_ANALYSIS_FILE Load a completed publication-analysis MAT file.
%   Publication exports use v7.3/HDF5 files in a OneDrive-synchronised
%   directory. A short retry window handles transient file-provider locks
%   without accepting a missing or malformed analysis variable.

max_attempts = 5;
last_error = [];
analysis = [];
loaded = false;
for attempt = 1:max_attempts
    try
        S = load(file_path, 'analysis');
        if ~isfield(S, 'analysis')
            error('load_publication_analysis_file:missingVariable', ...
                'The MAT file does not contain an analysis variable.');
        end
        analysis = S.analysis;
        loaded = true;
        break;
    catch ME
        last_error = ME;
        if attempt < max_attempts
            pause(0.25 * attempt);
        end
    end
end
if ~loaded
    rethrow(last_error);
end
validate_current_analysis(analysis, file_path);
end

function validate_current_analysis(analysis, file_path)
required = {'schema_version','scientific_config_sha256','task_id', ...
    'task_family','seeds'};
schemaVersion = [];
if isstruct(analysis) && isscalar(analysis) && isfield(analysis, 'schema_version')
    schemaVersion = double(analysis.schema_version);
end
if ~isstruct(analysis) || ~isscalar(analysis) || ...
        ~all(isfield(analysis, required)) || ...
        ~isscalar(schemaVersion) || ~isfinite(schemaVersion) || ...
        schemaVersion < 5 || ~isstruct(analysis.seeds) || isempty(analysis.seeds) || ...
        strlength(string(analysis.scientific_config_sha256)) == 0
    error('load_publication_analysis_file:obsoleteSchema', ...
        'File %s is not a current BANFF publication analysis (schema 5 or newer).', ...
        file_path);
end

fingerprint = string(analysis.scientific_config_sha256);
for index = 1:numel(analysis.seeds)
    seed = analysis.seeds(index);
    if ~isfield(seed, 'scientific_config_sha256') || ...
            string(seed.scientific_config_sha256) ~= fingerprint
        error('load_publication_analysis_file:fingerprintMismatch', ...
            'File %s contains a seed with a missing or inconsistent scientific fingerprint.', ...
            file_path);
    end
end
end
