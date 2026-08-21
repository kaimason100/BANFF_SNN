function arc_print_result_summary(result)
%ARC_PRINT_RESULT_SUMMARY Print final result fields in a compact ARC log form.

fprintf('[ARC summary] backend: %s%s', get_text(result, 'backend', 'unknown'), newline);
if isfield(result, 'domain')
    fprintf('[ARC summary] domain: %s%s', char(result.domain), newline);
end
if isfield(result, 'system_name')
    fprintf('[ARC summary] system_name: %s%s', char(result.system_name), newline);
end
if isfield(result, 'best')
    disp('[ARC summary] best:');
    disp(result.best);
end
if isfield(result, 'test') && isfield(result.test, 'status') && ...
        strcmp(result.test.status, 'not_run_on_arc')
    fprintf('[ARC summary] held-out testing was deferred to the local workflow.%s', newline);
end
if isfield(result, 'model') && isstruct(result.model)
    P = result.model;
    fprintf('[ARC summary] model architecture: recurrent=%s | decoder=%s | signed_distribution=%s | recurrent_storage=%s%s', ...
        get_text(P, 'recurrent_mode', 'unknown'), get_text(P, 'decoder_mode', 'unknown'), ...
        get_text(P, 'signed_decoder_distribution', 'unknown'), get_text(P, 'recurrent_storage', 'unknown'), newline);
    if isfield(P, 'W_rec') && ~isempty(P.W_rec)
        nnz_rec = nnz(P.W_rec);
        density = double(nnz_rec) / max(1, double(numel(P.W_rec)));
        fprintf('[ARC summary] full-rank recurrence: nnz=%g | density=%.6g%s', double(nnz_rec), density, newline);
    end
end
if isfield(result, 'model_file')
    fprintf('[ARC summary] saved model: %s%s', result.model_file, newline);
end
end

function txt = get_text(s, name, fallback)
if isstruct(s) && isfield(s, name)
    txt = char(string(s.(name)));
else
    txt = fallback;
end
end
