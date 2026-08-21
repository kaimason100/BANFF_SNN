% live_sgtitle.m
function live_sgtitle(txt)
if exist('sgtitle', 'file') == 2
    sgtitle(txt, 'Interpreter', 'none');
end
end

