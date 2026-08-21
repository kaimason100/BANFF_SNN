% recurrent_storage_id.m
% Helper for snn_primary_api.

function id = recurrent_storage_id(P)
storage = string(get_opt(P, 'recurrent_storage', get_opt(P, 'full_rank_storage', "dense")));
switch storage
    case "dense"
        id = int32(0);
    case "sparse"
        id = int32(1);
    otherwise
        error('snn_primary_api:unknownRecurrentStorage', ...
            'Unknown recurrent storage: %s', char(storage));
end
end
