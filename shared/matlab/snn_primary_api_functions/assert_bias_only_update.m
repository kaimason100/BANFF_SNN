% assert_bias_only_update.m
function assert_bias_only_update(P_before, P_after, context)
%ASSERT_BIAS_ONLY_UPDATE Guard the active release training contract.
%   Optimizer accumulators and P.B are allowed to change. Fixed network
%   structure, encoders, decoders, recurrent feedback and neuron constants
%   must remain identical.
allowed = {'B','m_b','v_b','vhat_b','t_adam'};
fields = fieldnames(P_before);
for ii = 1:numel(fields)
    key = fields{ii};
    if any(strcmp(key, allowed)) || ~isfield(P_after, key)
        continue;
    end
    if ~isequaln(P_before.(key), P_after.(key))
        error('snn_primary_api:nonBiasParameterChanged', ...
            '%s changed fixed model field "%s"; active training must update hidden bias only.', context, key);
    end
end
end

