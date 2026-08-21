function analysis = load_publication_analysis_file(file_path)
%LOAD_PUBLICATION_ANALYSIS_FILE Load a completed publication-analysis MAT file.
%   Publication exports use v7.3/HDF5 files in a OneDrive-synchronised
%   directory. A short retry window handles transient file-provider locks
%   without accepting a missing or malformed analysis variable.

max_attempts = 5;
last_error = [];
for attempt = 1:max_attempts
    try
        S = load(file_path, 'analysis');
        if ~isfield(S, 'analysis')
            error('load_publication_analysis_file:missingVariable', ...
                'The MAT file does not contain an analysis variable.');
        end
        analysis = S.analysis;
        return;
    catch ME
        last_error = ME;
        if attempt < max_attempts
            pause(0.25 * attempt);
        end
    end
end
rethrow(last_error);
end
