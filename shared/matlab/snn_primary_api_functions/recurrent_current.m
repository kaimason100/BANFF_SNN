% recurrent_current.m
% Helper for snn_primary_api.

function I_rec = recurrent_current(P, r)
%RECURRENT_CURRENT Dispatch recurrent current by architecture mode.
mode = string(get_opt(P, 'recurrent_mode', "low_rank"));
switch mode
    case "low_rank"
        y_dec = P.W_out_base_rec * r;
        I_rec = P.SCALE_rec * (P.Eta_rec * y_dec) - P.dself .* r;
        I_rec = single(I_rec);
    case "full_rank"
        storage = string(get_opt(P, 'recurrent_storage', get_opt(P, 'full_rank_storage', "dense")));
        switch storage
            case {"dense", "sparse"}
                I_rec = single(P.W_rec * r);
            otherwise
                error('snn_primary_api:unknownRecurrentStorage', ...
                    'Unknown recurrent storage: %s', char(storage));
        end
    otherwise
        error('snn_primary_api:unknownRecurrentMode', ...
            'Unknown recurrent mode: %s', char(mode));
end
end
