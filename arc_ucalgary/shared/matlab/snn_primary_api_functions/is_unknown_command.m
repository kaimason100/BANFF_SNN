% is_unknown_command.m
function tf = is_unknown_command(ME)
tf = contains(string(ME.message), 'Unknown command', 'IgnoreCase', true);
end

