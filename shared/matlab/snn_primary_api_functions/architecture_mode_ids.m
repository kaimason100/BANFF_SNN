% architecture_mode_ids.m
% Helper for snn_primary_api.

function [recurrent_mode_id, decoder_mode_id] = architecture_mode_ids(P)
%ARCHITECTURE_MODE_IDS Convert model architecture strings to MEX ids.
recurrent_mode = string(get_opt(P, 'recurrent_mode', "low_rank"));
decoder_mode = string(get_opt(P, 'decoder_mode', "shared"));
switch recurrent_mode
    case "low_rank"
        recurrent_mode_id = int32(0);
    case "full_rank"
        recurrent_mode_id = int32(1);
    otherwise
        error('snn_primary_api:unknownRecurrentMode', ...
            'Unknown recurrent mode: %s', char(recurrent_mode));
end
switch decoder_mode
    case "shared"
        decoder_mode_id = int32(0);
    case "signed"
        decoder_mode_id = int32(1);
    otherwise
        error('snn_primary_api:unknownDecoderMode', ...
            'Unknown decoder mode: %s', char(decoder_mode));
end
end
