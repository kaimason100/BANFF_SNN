% primary_bias_training_metadata.m
function meta = primary_bias_training_metadata()
%PRIMARY_BIAS_TRAINING_METADATA Describe the trainable surface of this release.
%   All scripts train the hidden-neuron bias vector only. The input encoder,
%   recurrent feedback decoder, primary readout decoder, recurrent mapping,
%   and neuron/adaptation constants are fixed after deterministic
%   initialization.
meta = struct();
meta.trainable_parameters = 'hidden_bias_only';
meta.primary_decoder_only = true;
meta.second_decoder_present = false;
meta.fixed_parameters = {'W_in','W_out_base_rec','W_out','Eta_rec','dself','W_rec','neuron_dynamics'};
meta.update_rule = 'Adam on supervised primary-decoder bias gradient';
end
