% get_opt.m
function val = get_opt(s, key, default_val)
if isstruct(s) && isfield(s, key)
    val = s.(key);
else
    warn_default_fallback(s, key, default_val);
    val = default_val;
end
end

function warn_default_fallback(s, key, default_val)
%WARN_DEFAULT_FALLBACK Make option defaults visible without flooding loops.
%   The same missing option can be queried many times inside training or
%   plotting loops. Warn once per caller/key/default so fallbacks are explicit
%   but still readable.
if isstruct(s) && isfield(s, 'warn_default_fallbacks') && ...
        isequal(s.warn_default_fallbacks, false)
    return;
end
persistent warned
if isempty(warned)
    warned = containers.Map('KeyType', 'char', 'ValueType', 'logical');
end
stack = dbstack(1);
if numel(stack) >= 2
    caller = stack(2).name;
elseif ~isempty(stack)
    caller = stack(1).name;
else
    caller = 'base';
end
key_text = char(string(key));
default_text = default_to_text(default_val);
warning_key = sprintf('%s|%s|%s', caller, key_text, default_text);
if isKey(warned, warning_key)
    return;
end
warned(warning_key) = true;
warning('snn_primary_api:defaultFallback', ...
    'Using default option in %s: opts.%s = %s because the field was not supplied.', ...
    caller, key_text, default_text);
end

function txt = default_to_text(value)
if ischar(value)
    txt = ['''' value ''''];
elseif isstring(value) && isscalar(value)
    txt = ['"' char(value) '"'];
elseif isnumeric(value) || islogical(value)
    if isempty(value)
        txt = '[]';
    elseif isscalar(value)
        txt = mat2str(value);
    else
        sz = size(value);
        txt = sprintf('<%s %s>', class(value), char(strjoin(string(sz), 'x')));
    end
elseif isstruct(value)
    txt = '<struct>';
elseif iscell(value)
    txt = '<cell>';
else
    txt = ['<' class(value) '>'];
end
end
