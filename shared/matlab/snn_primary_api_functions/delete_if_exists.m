% delete_if_exists.m
function delete_if_exists(file_name)
if exist(file_name, 'file') == 2
    delete(file_name);
end
end

