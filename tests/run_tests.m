function report = run_tests(mode)
%RUN_TESTS Mathematical and reproducibility regression tests for BANFF SNN.
%   RUN_TESTS("quick") runs CPU/data tests. RUN_TESTS("full") additionally
%   runs GPU-kernel comparisons, nine-task smoke tests and checkpoint restart.
%   Quick tests target mathematical identities, leakage prevention and indexed
%   reproducibility at small sizes. Full tests add backend equivalence and
%   end-to-end orchestration. Fixed tolerances are chosen for the precision of
%   the quantity under test rather than reused indiscriminately.

if nargin < 1, mode = "quick"; end
mode = lower(string(mode));
if ~any(mode == ["quick" "full"])
    error('banff:testMode', 'mode must be "quick" or "full".');
end
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root);
if mode == "full" && ~canUseGPU
    error('banff:testNoGPU', ...
        'run_tests("full") requires a supported GPU; use "quick" for CPU/data checks.');
end

names = {};
passed = [];
messages = {};
run(@test_default_config, 'default configuration');
run(@test_initial_bias_vector, 'heterogeneous initial-bias configuration');
run(@test_delayed_cue_data, 'delayed cue-response construction');
run(@test_dataset_hashes, 'dataset SHA-256');
run(@test_static_splits_and_preprocessing, 'static splits/preprocessing');
run(@test_low_rank_operator, 'low-rank operator and Dale signs');
run(@test_full_state_finite_difference, 'full local-state finite difference');
run(@test_lsti_event_sensitivity, 'LSTI event-time sensitivity');
run(@test_canonical_amsgrad, 'canonical AMSGrad update');
run(@test_phase_metric_identity, 'phase metric identity');
run(@test_phase_metric_shape_rejection, 'phase metric shape rejection');
run(@test_training_provenance_scope, 'training provenance compatibility scope');
run(@test_provenance_line_endings, 'cross-platform provenance line endings');
run(@test_target_integrator_refinement, 'target integrator refinement');
run(@test_dynamics_burn_in_and_jitter, 'dynamics burn-in and IC jitter conventions');
if mode == "full"
    run(@test_assessment_current_magnitudes, 'static and dynamics current magnitudes');
    run(@test_gpu_reference_equivalence, 'shared timestep CPU/GPU agreement');
    run(@test_linear_path_equivalence, 'blocked encoder/averaged decoder agreement');
    run(@test_dynamics_output_elision, 'dynamics output-elision agreement');
    run(@test_checkpoint_restart, 'checkpoint/restart equivalence');
    run(@test_ten_task_smoke, 'ten-task train/test smoke test');
    run(@test_alternative_paths, 'surrogate/full-rank/SPSA smoke test');
end

report = table(string(names(:)), logical(passed(:)), string(messages(:)), ...
    'VariableNames', {'test','passed','message'});
disp(report);
if any(~report.passed)
    error('banff:testsFailed', '%d BANFF test(s) failed.', nnz(~report.passed));
end
fprintf('All %d BANFF tests passed (%s mode).\n', height(report), mode);

    function run(testFunction, name)
        names{end+1} = name; %#ok<AGROW>
        try
            testFunction();
            passed(end+1) = true; %#ok<AGROW>
            messages{end+1} = 'ok'; %#ok<AGROW>
        catch ME
            passed(end+1) = false; %#ok<AGROW>
            if isempty(ME.stack)
                location = '';
            else
                location = sprintf(' (%s:%d)',ME.stack(1).name,ME.stack(1).line);
            end
            messages{end+1} = sprintf('%s: %s%s', ...
                ME.identifier,ME.message,location); %#ok<AGROW>
        end
    end
end

function test_default_config()
cfg = banff("config", "lorenz", struct());
assert(cfg.eligibility_mode == "hard_spike");
assert(abs(double(cfg.hard_event_gain) - 1) < 1e-12);
assert(abs(double(cfg.initial_bias) - 20) < 1e-12);
assert(cfg.N_hidden == 32000 && cfg.N_recurrent == 10);
assert(cfg.batch_size == 256);
assert(cfg.optimizer == "amsgrad");
assert(cfg.epochs == 60000);
assert(cfg.validation_warmup_time == cfg.test_warmup_time);
for task = ["lorenz" "sprott_s" "vanderpol"]
    dynamicsCfg = banff("config",task,struct());
    assert(dynamicsCfg.epochs == 60000);
    assert(dynamicsCfg.validation_warmup_time == dynamicsCfg.test_warmup_time);
end
spsaDynamics = banff("config","vanderpol",struct('method',"spsa"));
assert(spsaDynamics.epochs == 60000);
spsa = banff("config", "breast_cancer", struct('method', "spsa"));
assert(spsa.epochs == 50000);
assert(spsa.spsa_schedule_epochs == 50000);
assert(spsa.learning_rate_schedule_epochs == 50000);
assert(~isfield(spsa, 'spsa_continuation_boundary'));
end

function test_initial_bias_vector()
% Explicit heterogeneous initialization must be exact and identity-bearing.
bias = single([18; 19; 21; 22]);
cfg = banff("config", "breast_cancer", struct( ...
    'N_hidden', 4, 'N_recurrent', 2, 'initial_bias', bias));
P = banff_model('create', 3, 2, cfg);
assert(isequal(P.B, bias));
uniform = banff("config", "breast_cancer", struct( ...
    'N_hidden', 4, 'N_recurrent', 2, 'initial_bias', single(20)));
assert(~strcmp(cfg.scientific_config_sha256, ...
    uniform.scientific_config_sha256));
rejected = false;
try
    banff("config", "breast_cancer", struct( ...
        'N_hidden', 4, 'N_recurrent', 2, ...
        'initial_bias', single([19; 20; 21])));
catch exception
    rejected = strcmp(exception.identifier, 'banff:initialBiasSize');
end
assert(rejected, 'An incorrectly sized initial-bias vector was accepted.');
end

function test_delayed_cue_data()
cfg=banff('config','delayed_cue',struct( ...
    'N_hidden',16,'N_recurrent',4,'sequence_cue_steps',3, ...
    'sequence_delay_steps',5,'sequence_response_steps',2, ...
    'sequence_distractor_block_steps',2,'sequence_train_samples',8, ...
    'sequence_validation_samples',4,'sequence_test_samples',6));
[data,information]=banff_data('temporal',cfg);
assert(isequal(size(data.X_train),[3 10 8]));
assert(isequal(size(data.Y_train),[2 8]));
assert(all(sum(data.Y_train,1)==1));
[~,labels]=max(data.Y_train,[],1);
expectedCue=single(2.*(labels-1)-1);
actualCue=squeeze(data.X_train(1,1,:)).';
assert(isequal(actualCue,expectedCue));
assert(all(data.X_train(1,4:end,:)==0,'all'));
assert(all(data.X_train(3,1:8,:)==0,'all'));
assert(all(data.X_train(3,9:10,:)==1,'all'));
assert(information.delay_steps==5 && information.response_steps==2);
delayPatterns=squeeze(data.X_train(2,4:8,:)).';
assert(size(unique(delayPatterns,'rows'),1)==4);
[reloaded,reloadedInformation]=banff_data('temporal',cfg,information);
assert(isequal(data.X_test,reloaded.X_test));
assert(isequal(data.Y_test,reloaded.Y_test));
assert(isequal(information,reloadedInformation));
end

function test_dataset_hashes()
root = fileparts(fileparts(mfilename('fullpath')));
expected = struct( ...
    'abalone_dataset_mat','1f962b467f659af39bfd6f46b192acf5846c44c7ee3f62993e8223a2ec1acb7b', ...
    'afro_mnist_vai_mat','9e77781ca362d3a144a71685cc8e1d7776ddf27353f1e4c6e4cbc2cf5b362f9a', ...
    'breast_cancer_dataset_mat','4bf45f311ee0a7efc16b9548bfc68f7841483ce47b27dd09ecfe2609abd06d62', ...
    'mnist_mat','5eee8657f3c5a853f923077fab5e01215a342cae8f989a67a2d703cf9adc65fa', ...
    'toyota_dataset_mat','09a0b8c5f600d06756c7eee172ee277d17c654e3a03253fcb00ad9e6a93b1528', ...
    'yacht_dataset_mat','df92835406c002b2c1550b52d095085c86e6a8bbdbf3dc790b4d780dd9057d03');
files = {'abalone_dataset.mat','afro_mnist_vai.mat','breast_cancer_dataset.mat', ...
    'mnist.mat','toyota_dataset.mat','yacht_dataset.mat'};
for i = 1:numel(files)
    key = matlab.lang.makeValidName(files{i});
    actual = sha256_file(fullfile(root,'data','raw',files{i}));
    assert(strcmp(actual, expected.(key)), 'Dataset hash mismatch for %s.', files{i});
end
end

function test_static_splits_and_preprocessing()
tasks = ["breast_cancer","mnist","afro_mnist_vai","abalone","toyota","yacht"];
for task = tasks
    cfg = banff("config", task, struct('N_hidden', 8, 'N_recurrent', 2));
    [data1, info1] = banff_data('static', cfg);
    [data2, info2] = banff_data('static', cfg, info1);
    assert(isequal(data1.X_train, data2.X_train));
    assert(isequal(data1.Y_validation, data2.Y_validation));
    assert(isequal(data1.X_test, data2.X_test));
    assert(strcmp(info1.dataset_sha256, info2.dataset_sha256));
    allIndex = double([info1.train_index; info1.validation_index; info1.test_index]);
    assert(numel(unique(allIndex)) == numel(allIndex));
    assert(all(isfinite(info1.feature_mean), 'all') && all(info1.feature_std > 0, 'all'));
end
end

function test_low_rank_operator()
% Verify the factorised recurrent operator and explicit diagonal correction.
cfg = banff("config", "breast_cancer", struct( ...
    'N_hidden', 32, 'N_recurrent', 5, 'recurrent_gain', single(.05)));
P = banff_model('create', 3, 2, cfg);
A = double(P.recurrentGain) .* double(P.recurrent_expansion) * double(P.W_feedback) ...
    - diag(double(P.self_coupling));
assert(max(abs(diag(A))) < 2e-7);
for j = 1:P.N_hidden
    outgoing = A(:,j); outgoing(j) = 0;
    nonzero = outgoing(abs(outgoing) > 1e-12);
    if isempty(nonzero), continue; end
    if P.dale_sign(j) > 0
        assert(all(nonzero >= 0));
    else
        assert(all(nonzero <= 0));
    end
end
assert(rank(double(P.recurrent_expansion) * double(P.W_feedback), 1e-9) <= cfg.N_recurrent);
end

function test_full_state_finite_difference()
% Compare propagated local bias sensitivities with centred finite differences.
cfg = banff("config", "lorenz", struct( ...
    'N_hidden',1,'N_recurrent',1,'recurrent_gain',single(0), ...
    'encoder_gain',single(0),'initial_bias',single(4), ...
    'eligibility_mode',"hard_spike"));
P = banff_model('create',1,1,cfg);
state = make_state(P, 1);
state.u(:) = -64;
state.w(:) = 1.7;
state.epsilonVoltage(:) = .31;
state.epsilonAdaptation(:) = .44;
input = single(0);
[next, spike] = banff_model('reference_step', P, state, input, true);
assert(~spike, 'Finite-difference smooth test unexpectedly spiked.');
% The simulator state is single precision.  A 1e-3 mV perturbation makes the
% central difference comparable to a few voltage ULPs and is therefore too
% small for a reliable numerical derivative.
h = single(0.1);
plusP = P; minusP = P; plusP.B = P.B+h; minusP.B = P.B-h;
plus = state; minus = state;
plus.u = state.u + h*state.epsilonVoltage;
minus.u = state.u - h*state.epsilonVoltage;
plus.w = state.w + h*state.epsilonAdaptation;
minus.w = state.w - h*state.epsilonAdaptation;
plus = banff_model('reference_step', plusP, plus, input, false);
minus = banff_model('reference_step', minusP, minus, input, false);
fdU = (plus.u-minus.u)/(2*h);
fdW = (plus.w-minus.w)/(2*h);
voltageError = abs(double(fdU-next.epsilonVoltage));
adaptationError = abs(double(fdW-next.epsilonAdaptation));
assert(voltageError < 2e-4, ...
    'Voltage-sensitivity finite-difference error was %.3g.', voltageError);
assert(adaptationError < 2e-4, ...
    'Adaptation-sensitivity finite-difference error was %.3g.', adaptationError);
end

function test_lsti_event_sensitivity()
% Audit the selected-event, frozen-rho derivative on pre/post-event branches.
cfg = banff("config", "lorenz", struct( ...
    'N_hidden',1,'N_recurrent',1,'recurrent_gain',single(0), ...
    'encoder_gain',single(0),'initial_bias',single(80), ...
    'eligibility_mode',"hard_spike",'hard_event_gain',single(1)));
P = banff_model('create',1,1,cfg);
state = make_state(P,1);
state.u(:) = -51.2;
state.w(:) = .3;
state.epsilonVoltage(:) = .27;
state.epsilonAdaptation(:) = .36;
[next, spike, rho, raw] = banff_model('reference_step', P, state, single(0), true); %#ok<ASGLU>
assert(spike && rho > 0 && rho <= 1, 'Expected a resolved LSTI spike.');
aPre = exp(double(rho) * log(double(P.alpha)));
expectedPre = aPre*double(state.epsilonVoltage) ...
    + (1-aPre)*(1-double(state.epsilonAdaptation));
assert(abs(double(raw)/double(P.hardEventGain)-expectedPre) < 2e-6);
% Frozen-rho finite difference of the pre-event voltage map.
h = 1e-4;
baseCurrent = double(P.B);
up = double(state.u)+h*double(state.epsilonVoltage);
um = double(state.u)-h*double(state.epsilonVoltage);
wp = double(state.w)+h*double(state.epsilonAdaptation);
wm = double(state.w)-h*double(state.epsilonAdaptation);
Bp = baseCurrent+h; Bm = baseCurrent-h;
vp = double(P.restingVoltage)+aPre*(up-double(P.restingVoltage)) ...
    +(1-aPre)*(Bp-wp);
vm = double(P.restingVoltage)+aPre*(um-double(P.restingVoltage)) ...
    +(1-aPre)*(Bm-wm);
fd = (vp-vm)/(2*h);
assert(abs(fd-expectedPre) < 2e-5);
end

function test_phase_metric_identity()
options = struct('projections',32,'trim_fraction',.10,'subsample',1, ...
    'transient_fraction',0,'max_points',500);
t = linspace(0,2*pi,200).';
trajectory = [cos(t),sin(t)];
d = banff_metrics('phase_distance', trajectory, trajectory, options);
assert(abs(d) < 1e-12);
end

function test_assessment_current_magnitudes()
% Verify the streaming assessment aggregation against an analytic encoder
% magnitude and the model's explicit bias reference. This test is in full
% mode because the diagnostic deliberately reuses the production GPU step.
cfg = banff("config", "breast_cancer", struct( ...
    'N_hidden',8,'N_recurrent',2,'presentation_time',single(.005), ...
    'batch_size',2,'initial_bias',single(13)));
P = banff_model('create',3,2,cfg);
P.B = P.B + single(linspace(-1,1,P.N_hidden).');
X = single([.2 -.3 .4 .1; -.1 .5 .2 -.4; .3 .1 -.2 .6]);
summary = banff_plot('static_current_magnitudes',P,X,cfg);
fields = {'encoder_net_rms','encoder_gross_afferent_rms', ...
    'recurrent_net_rms','recurrent_gross_afferent_rms', ...
    'decoder_presynaptic_rms','bias_deviation','adaptation_rms'};
for index = 1:numel(fields)
    values = summary.(fields{index});
    assert(isequal(size(values),[P.N_hidden 1]));
    assert(all(isfinite(values),'all') && all(values>=0,'all'));
end
encoderCurrent = double(P.W_in*(P.inputScale.*X));
expectedEncoder = sqrt(mean(encoderCurrent.^2,2));
assert(max(abs(double(summary.encoder_net_rms)-expectedEncoder),[],'all') < 2e-6);
grossEncoderCurrent = double(abs(P.W_in)) * ...
    double(P.inputScale.*abs(X));
expectedGrossEncoder = sqrt(mean(grossEncoderCurrent.^2,2));
assert(max(abs(double(summary.encoder_gross_afferent_rms)-expectedGrossEncoder), ...
    [],'all') < 2e-6);
assert(all(double(summary.encoder_gross_afferent_rms)+2e-6 >= ...
    double(summary.encoder_net_rms),'all'));
assert(all(double(summary.recurrent_gross_afferent_rms)+2e-6 >= ...
    double(summary.recurrent_net_rms),'all'));
assert(isequal(summary.encoder,summary.encoder_net_rms));
assert(isequal(summary.recurrent,summary.recurrent_net_rms));
assert(isequal(summary.encoder_net,summary.encoder_net_rms));
assert(isequal(summary.recurrent_net,summary.recurrent_net_rms));
expectedPopulationEncoder=sqrt(mean(expectedEncoder.^2));
assert(abs(summary.aggregate.encoder_rms_mV-expectedPopulationEncoder)<2e-6);
assert(abs(summary.aggregate.recurrent_to_encoder_rms- ...
    summary.aggregate.net_recurrent_rms_mV/ ...
    summary.aggregate.encoder_rms_mV)<1e-12);
assert(abs(summary.aggregate.net_to_gross_encoder_rms- ...
    summary.aggregate.encoder_rms_mV/ ...
    summary.aggregate.gross_encoder_rms_mV)<1e-12);
expectedReference = single(cfg.initial_bias);
assert(abs(summary.bias_reference_mV-double(expectedReference)) < 1e-6);
assert(max(abs(double(summary.bias_deviation)- ...
    abs(double(P.B-expectedReference))),[],'all') < 1e-6);
assert(summary.test_samples==size(X,2));
assert(summary.timesteps_per_sample==double(P.presentationSteps));
assert(summary.observations_per_neuron==size(X,2)*double(P.presentationSteps));
assert(summary.decoder_window_observations_per_neuron== ...
    size(X,2)*double(P.averageSteps));

% Full-set inverse ISIs must equal the union of independently replayed
% samples. This verifies that every sample is included and that no interval
% is accidentally constructed across a static-state reset.
allInverseIsi=banff_plot('static_inverse_isi_rates',P,X,cfg);
sampleInverseIsi=cell(size(X,2),1);
for sample=1:size(X,2)
    sampleInverseIsi{sample}=banff_plot( ...
        'static_inverse_isi_rates',P,X(:,sample),cfg);
end
separateInverseIsi=vertcat(sampleInverseIsi{:});
assert(isequal(sort(allInverseIsi),sort(separateInverseIsi)));
assert(all(isfinite(allInverseIsi)) && all(allInverseIsi>0));

% Exercise the corresponding closed-loop DS aggregation at a tiny size.
dynamicsCfg=banff("config","lorenz",struct( ...
    'N_hidden',8,'N_recurrent',2,'long_simulation_time',single(.03), ...
    'burn_in_time',single(.005),'training_window',single(.005), ...
    'validation_time',single(.004),'validation_warmup_time',single(.002), ...
    'validation_initial_conditions',1));
[~,dynamicsInformation]=banff_data('dynamics',dynamicsCfg);
dynamicsP=banff_model('create',3,3,dynamicsCfg);
dynamicsSummary=banff_plot('dynamics_current_magnitudes',dynamicsP, ...
    dynamicsCfg,dynamicsInformation,"validation");
for index=1:numel(fields)
    values=dynamicsSummary.(fields{index});
    assert(isequal(size(values),[dynamicsP.N_hidden 1]));
    assert(all(isfinite(values),'all') && all(values>=0,'all'));
end
assert(all(double(dynamicsSummary.encoder_gross_afferent_rms)+2e-6>= ...
    double(dynamicsSummary.encoder_net_rms),'all'));
assert(all(double(dynamicsSummary.recurrent_gross_afferent_rms)+2e-6>= ...
    double(dynamicsSummary.recurrent_net_rms),'all'));
% The synthetic dynamics fixture has no external input, so both encoder
% magnitudes are exactly zero.  The reported diagnostic deliberately uses
% a protected denominator and should therefore be zero rather than the
% undefined direct ratio 0/0.
if dynamicsSummary.aggregate.gross_encoder_rms_mV==0
    assert(dynamicsSummary.aggregate.net_to_gross_encoder_rms==0);
else
    assert(abs(dynamicsSummary.aggregate.net_to_gross_encoder_rms- ...
        dynamicsSummary.aggregate.encoder_rms_mV/ ...
        dynamicsSummary.aggregate.gross_encoder_rms_mV)<1e-12);
end
assert(dynamicsSummary.test_samples==1);
assert(dynamicsSummary.timesteps_per_sample==4);
assert(dynamicsSummary.observations_per_neuron==4);
end

function test_gpu_reference_equivalence()
% Compare every state field and event diagnostic after one CPU/GPU transition.
if ~canUseGPU
    warning('BANFF:testNoGPU','GPU test skipped because no supported GPU is available.');
    return;
end
for mode = ["hard_spike" "surrogate"]
    cfg = banff("config", "breast_cancer", struct( ...
        'N_hidden',32,'N_recurrent',5,'eligibility_mode',mode));
    P = banff_model('create',3,2,cfg);
    rng(17,'twister');
    state = make_state(P,3);
    state.u = single(-58 + 9*rand(P.N_hidden,3,'single'));
    state.w = single(.8*rand(P.N_hidden,3,'single'));
    state.r = single(.1*rand(P.N_hidden,3,'single'));
    state.x = single(.1*rand(P.N_hidden,3,'single'));
    state.epsilonVoltage = single(rand(P.N_hidden,3,'single'));
    state.epsilonAdaptation = single(.5*rand(P.N_hidden,3,'single'));
    state.eligibilityRise = single(.01*rand(P.N_hidden,3,'single'));
    state.eligibilityDecay = single(.01*rand(P.N_hidden,3,'single'));
    input = single(randn(P.N_hidden,3,'single'));
    [reference, spikeRef, rhoRef] = banff_model('reference_step',P,state,input,true);
    [gpuState, spikeGpu, rhoGpu] = banff_model('gpu_step',P,state,input,true);
    fields = fieldnames(reference);
    for i = 1:numel(fields)
        a = double(reference.(fields{i})); b = double(gather(gpuState.(fields{i})));
        assert(max(abs(a-b),[],'all') < 5e-5, 'GPU mismatch in %s (%s).', fields{i}, mode);
    end
    assert(isequal(spikeRef, gather(spikeGpu)));
    assert(max(abs(double(rhoRef)-double(gather(rhoGpu))),[],'all') < 5e-6);
end
end

function test_phase_metric_shape_rejection()
cfg = banff("config", "lorenz", struct());
prediction = single(zeros(10,3));
truth = single(zeros(9,3));
threw = false;
try
    banff_metrics('phase_distance',prediction,truth,cfg.phase_metric);
catch ME
    threw = strcmp(ME.identifier,'banff:phaseShapeMismatch');
end
assert(threw,'A trajectory-length mismatch was not rejected.');

truth = single(zeros(10,2));
threw = false;
try
    banff_metrics('phase_distance',prediction,truth,cfg.phase_metric);
catch ME
    threw = strcmp(ME.identifier,'banff:phaseShapeMismatch');
end
assert(threw,'A trajectory-dimension mismatch was not rejected.');
end

function test_training_provenance_scope()
training = banff_provenance("training");
expected = sort({'banff_train_m','banff_eval_m','banff_model_m', ...
    'banff_data_m','banff_metrics_m'});
assert(isequal(sort(fieldnames(training)).',expected));
assert(~isfield(training,'banff_test_m'));
assert(~isfield(training,'banff_publication_m'));
assert(banff_provenance("assert_training_compatible", ...
    struct('training_source_sha256',training)));

% Exact additive transitions preserve older tasks while adding heterogeneous
% bias initialization and a separate temporal task. Unknown signatures,
% invalid scalar/vector use, and the new task itself must remain rejected.
legacy = struct( ...
    'banff_train_m','fe2c766b0cc005406d4d033f2df3e0f292cab95700f977db9d9235cc616ec627', ...
    'banff_eval_m','0faf252c04cd0536760d72424795349cfd7fbd056884d16f4a1ca5b2c43c9111', ...
    'banff_model_m', ...
    'a98b1dcc3b1fcd2b35e65329fbabdb777e4fd0b1d37e5c7e1613c1a2cf509d84', ...
    'banff_data_m','e3a742283397c5ceb2cff061abe73f01039cb9ab9b1b2f65ba87fe3deffdc44a', ...
    'banff_metrics_m','c3f34d81f3e84bd07f423896a27ee4b98c9296e110461e85b10babd5dd9b47c4');
saved = struct('training_source_sha256',legacy);
assert(banff_provenance("assert_training_compatible",saved, ...
    struct('task',"lorenz",'initial_bias',single(20))));

assert_provenance_rejected(saved, ...
    struct('task',"lorenz",'initial_bias',single([19;21])));
vectorAware=saved;
vectorAware.training_source_sha256.banff_model_m= ...
    '670af95eca96e45aff643536b0273342824373b5a41ccd72917455c8deb8a0af';
assert(banff_provenance("assert_training_compatible",vectorAware, ...
    struct('task',"lorenz",'initial_bias',single([19;21]))));
assert_provenance_rejected(vectorAware, ...
    struct('task',"delayed_cue",'initial_bias',single(20)));
unknown = saved;
unknown.training_source_sha256.banff_model_m = repmat('0',1,64);
assert_provenance_rejected(unknown, ...
    struct('task',"lorenz",'initial_bias',single(20)));
secondMismatch = saved;
secondMismatch.training_source_sha256.banff_eval_m = repmat('0',1,64);
assert_provenance_rejected(secondMismatch, ...
    struct('task',"lorenz",'initial_bias',single(20)));
end

function assert_provenance_rejected(saved, config)
rejected = false;
try
    banff_provenance("assert_training_compatible",saved,config);
catch exception
    rejected = strcmp(exception.identifier,'banff:modelSourceMismatch');
end
assert(rejected,'An incompatible training-source signature was accepted.');
end

function test_provenance_line_endings()
% An identical source file must retain its provenance across Linux (LF),
% Windows (CRLF), and a legacy/mixed text representation.
folder = tempname;
mkdir(folder);
cleanup = onCleanup(@() rmdir(folder,'s')); %#ok<NASGU>
lfFile = fullfile(folder,'lf.m');
crlfFile = fullfile(folder,'crlf.m');
mixedFile = fullfile(folder,'mixed.m');
write_bytes(lfFile,uint8(['alpha' 10 'beta' 10 'gamma' 10]));
write_bytes(crlfFile,uint8(['alpha' 13 10 'beta' 13 10 'gamma' 13 10]));
write_bytes(mixedFile,uint8(['alpha' 13 10 'beta' 10 'gamma' 13]));
expected = banff_provenance("hash_file",lfFile);
assert(strcmp(expected,banff_provenance("hash_file",crlfFile)));
assert(strcmp(expected,banff_provenance("hash_file",mixedFile)));
end

function write_bytes(file, bytes)
fileId = fopen(file,'w');
assert(fileId >= 0,'Could not create provenance test file %s.',file);
cleanup = onCleanup(@() fclose(fileId)); %#ok<NASGU>
fwrite(fileId,bytes,'uint8');
end

function test_canonical_amsgrad()
cfg = banff("config", "lorenz", struct( ...
    'N_hidden',2,'N_recurrent',1,'initial_bias',single(0), ...
    'adam_beta1',single(.9),'adam_beta2',single(.99), ...
    'adam_epsilon',single(1e-7)));
P = banff_model('create',1,1,cfg);
P.B(:) = 0;

% Independent conventional AMSGrad reference calculation.
m = zeros(2,1,'single');
v = zeros(2,1,'single');
vMax = zeros(2,1,'single');
B = zeros(2,1,'single');
gradients = single([2 .5 -1; -.25 3 .75]);
learningRates = single([.03 .02 .01]);
b1 = single(cfg.adam_beta1);
b2 = single(cfg.adam_beta2);
for step = 1:size(gradients,2)
    g = gradients(:,step);
    m = b1.*m + (single(1)-b1).*g;
    v = b2.*v + (single(1)-b2).*g.*g;
    vMax = max(vMax,v);
    mHat = m ./ (single(1)-b1.^single(step));
    vMaxHat = vMax ./ (single(1)-b2.^single(step));
    B = B-learningRates(step).*mHat ./ ...
        (sqrt(vMaxHat)+single(cfg.adam_epsilon));

    P = banff_model('adam',P,g,learningRates(step),1,cfg);
    assert(max(abs(double(P.B)-double(B))) < 2e-7);
    assert(max(abs(double(P.m)-double(m))) < 2e-7);
    assert(max(abs(double(P.v)-double(v))) < 2e-7);
    assert(max(abs(double(P.vMax)-double(vMax))) < 2e-7);
end
end

function test_checkpoint_restart()
% Require an interrupted/resumed run to reproduce uninterrupted trainable state.
if ~canUseGPU
    warning('BANFF:testNoGPU','Checkpoint test skipped because no supported GPU is available.');
    return;
end

root = tempname; mkdir(root); cleanup = onCleanup(@() rmdir(root,'s')); %#ok<NASGU>
common = struct('N_hidden',16,'N_recurrent',4,'epochs',2,'validate_every',1, ...
    'presentation_time',single(.005),'average_fraction',single(.4), ...
    'batch_size',64,'verbose_every',1000);
full = common; full.output_directory = fullfile(root,'full'); full.checkpoint_hours = inf;
A = banff("train","breast_cancer",full);
part = common; part.output_directory = fullfile(root,'resume'); part.checkpoint_hours = 0;
B1 = banff("train","breast_cancer",part);
assert(~B1.complete);
part.checkpoint_hours = inf;
B2 = banff("train","breast_cancer",part);
assert(B2.complete);
assert(max(abs(double(A.final_trainable_state.B)-double(B2.final_trainable_state.B))) < 1e-5);
end

function test_dynamics_output_elision()
% Requesting no trajectory storage must not change loss or bias gradient.
if ~canUseGPU
    warning('BANFF:testNoGPU','GPU test skipped because no supported GPU is available.');
    return;
end
cfg = banff("config", "lorenz", struct( ...
    'N_hidden',32,'N_recurrent',5,'teacher_steps',3,'closed_loop_steps',2));
P = banff_model('gpu', banff_model('create',3,3,cfg));
rng(29,'twister');
target = single(randn(3,13,'single'));
teacherForcing = true(1,13);
teacherForcing([5 6 10 11]) = false;
[lossCompact, gradientCompact] = banff_model( ...
    'dynamics', P, target, teacherForcing, true, false);
[lossStored, gradientStored, output] = banff_model( ...
    'dynamics', P, target, teacherForcing, true, false);
assert(~isempty(output) && size(output,2) == size(target,2)-1);
assert(abs(double(gather(lossCompact-lossStored))) < 1e-6);
assert(max(abs(double(gather(gradientCompact-gradientStored))),[],'all') < 1e-6);
end

function test_linear_path_equivalence()
% Audit the two linear-operation reorderings used by optimized simulation.
if ~canUseGPU
    warning('BANFF:testNoGPU','GPU test skipped because no supported GPU is available.');
    return;
end
cfg = banff("config", "lorenz", struct('N_hidden',32,'N_recurrent',5));
P = banff_model('gpu', banff_model('create',3,3,cfg));
rng(31,'twister');
inputs = gpuArray(single(randn(3,7,'single')));
encoderColumns = gpuArray.zeros(P.N_hidden,7,'single');
for column = 1:7
    encoderColumns(:,column) = P.W_in * (P.inputScale .* inputs(:,column));
end
encoderBlock = P.W_in * (P.inputScale .* inputs);
assert(max(abs(double(gather(encoderColumns-encoderBlock))),[],'all') < 5e-5);

filteredStates = gpuArray(single(randn(P.N_hidden,7,'single')));
decoderRepeated = gpuArray.zeros(P.N_output,1,'single');
for column = 1:7
    decoderRepeated = decoderRepeated + P.W_out * filteredStates(:,column);
end
decoderRepeated = decoderRepeated ./ single(7);
decoderAveraged = P.W_out * (sum(filteredStates,2) ./ single(7));
assert(max(abs(double(gather(decoderRepeated-decoderAveraged))),[],'all') < 5e-5);
end

function test_ten_task_smoke()
% Exercise train/test orchestration for every registered task at minimal size.
if ~canUseGPU
    warning('BANFF:testNoGPU','Smoke tests skipped because no supported GPU is available.');
    return;
end
root = tempname; mkdir(root); cleanup = onCleanup(@() rmdir(root,'s')); %#ok<NASGU>
tasks = ["breast_cancer","mnist","afro_mnist_vai","abalone","toyota","yacht", ...
    "lorenz","sprott_s","vanderpol","delayed_cue"];
for task = tasks
    o = struct('N_hidden',16,'N_recurrent',4,'epochs',1,'batch_size',256, ...
        'presentation_time',single(.003),'average_fraction',single(1), ...
        'validate_every',1,'validate_dynamics_every',1,'verbose_every',1000, ...
        'output_directory',fullfile(root,char(task)), ...
        'checkpoint_hours',inf);
    if any(task == ["lorenz","sprott_s","vanderpol"])
        o.long_simulation_time = single(.10);
        o.burn_in_time = single(.01);
        o.training_window = single(.02);
        o.validation_time = single(.02);
        o.test_time = single(.02);
        o.validation_warmup_time = single(.005);
        o.test_warmup_time = single(.005);
        o.validation_initial_conditions = 1;
        o.test_initial_conditions = 1;
        o.phase_metric = struct('projections',16,'trim_fraction',0, ...
            'subsample',1,'transient_fraction',0,'max_points',100);
    elseif task=="delayed_cue"
        o.sequence_cue_steps=2;
        o.sequence_delay_steps=3;
        o.sequence_response_steps=2;
        o.sequence_distractor_block_steps=1;
        o.sequence_train_samples=8;
        o.sequence_validation_samples=4;
        o.sequence_test_samples=4;
    end
    R = banff("train",task,o);
    assert(R.complete && all(isfinite(R.final_trainable_state.B)));
    T = banff("test",task,o);
    assert(isfield(T,'test'));
    if task == "lorenz" || task == "sprott_s" || task == "vanderpol"
        assert(isfinite(T.test.phase_distance));
    else
        assert(isfinite(T.test.loss));
    end
end
end

function test_target_integrator_refinement()
% Euler target trajectories should converge as dt is halved over a short,
% smooth interval. This is a numerical sanity check, not a chaotic benchmark.
base = banff("config", "lorenz", struct('system_rate',single(1)));
system = banff_data('system', "lorenz");
initial = system.initial_state;
coarse = base; coarse.dt = single(1e-3);
medium = base; medium.dt = single(5e-4);
fine = base; fine.dt = single(2.5e-4);
T = single(.02);
x1 = banff_data('trajectory',system,initial,T,coarse);
x2 = banff_data('trajectory',system,initial,T,medium);
x3 = banff_data('trajectory',system,initial,T,fine);
e12 = norm(double(x1(:,end)-x2(:,end)));
e23 = norm(double(x2(:,end)-x3(:,end)));
assert(e23 < e12, 'Euler target integration did not improve after halving dt.');
end

function test_dynamics_burn_in_and_jitter()
% The endpoint-inclusive trajectory stores t=0 in column one, so a 10-step
% burn-in must retain column 11. IC jitter denotes a symmetric half-width.
cfg = banff("config", "lorenz", struct( ...
    'N_hidden',8,'N_recurrent',2,'long_simulation_time',single(.03), ...
    'burn_in_time',single(.01),'training_window',single(.01), ...
    'initial_condition_jitter',single(.02), ...
    'validation_initial_conditions',4,'test_initial_conditions',4));
[~, information] = banff_data('dynamics', cfg);
assert(information.burn_index == round(cfg.burn_in_time/cfg.dt)+1);

system = banff_data('system', cfg.task);
base = single(system.initial_state(:));
validation = banff_data('initial_conditions', cfg, "validation");
test = banff_data('initial_conditions', cfg, "test");
expectedFirst = base;
expectedFirst(1) = expectedFirst(1)+cfg.initial_condition_jitter;
assert(isequal(validation(:,1),expectedFirst));

previous = rng;
restore = onCleanup(@() rng(previous)); %#ok<NASGU>
rng(cfg.validation_initial_condition_seed,'twister');
expectedValidationRandom = base + cfg.initial_condition_jitter .* ...
    (single(2).*rand(numel(base),cfg.validation_initial_conditions-1,'single')-single(1));
rng(cfg.test_initial_condition_seed,'twister');
expectedTest = base + cfg.initial_condition_jitter .* ...
    (single(2).*rand(numel(base),cfg.test_initial_conditions,'single')-single(1));
assert(isequal(validation(:,2:end),expectedValidationRandom));
assert(isequal(test,expectedTest));
assert(all(abs(test-base) <= cfg.initial_condition_jitter,'all'));
end

function test_alternative_paths()
if ~canUseGPU, return; end
root = tempname; mkdir(root); cleanup = onCleanup(@() rmdir(root,'s')); %#ok<NASGU>
% Continuous-surrogate ablation: same simulator, different local gate.
o = struct('N_hidden',12,'N_recurrent',3,'epochs',1,'batch_size',128, ...
    'presentation_time',single(.003),'average_fraction',single(1), ...
    'validate_every',1,'verbose_every',1000,'checkpoint_hours',inf, ...
    'eligibility_mode',"surrogate",'output_directory',fullfile(root,'surrogate'));
R = banff("train","breast_cancer",o); assert(R.complete);
T = banff("test","breast_cancer",o); assert(isfinite(T.test.loss));
% Tiny sparse full-rank path.
o.eligibility_mode = "hard_spike";
o.recurrent_mode = "full_rank";
o.N_hidden = 12;
o.full_rank_probability = single(.25);
o.output_directory = fullfile(root,'fullrank');
R = banff("train","breast_cancer",o); assert(R.complete);
T = banff("test","breast_cancer",o); assert(isfinite(T.test.loss));
% Tiny SPSA path.
o.recurrent_mode = "low_rank";
o.method = "spsa";
o.output_directory = fullfile(root,'spsa');
R = banff("train","breast_cancer",o); assert(R.complete);
T = banff("test","breast_cancer",o); assert(isfinite(T.test.loss));
end

function state = make_state(P, batch)
state = struct( ...
    'u', repmat(single(P.restingVoltage), P.N_hidden, batch), ...
    'w', zeros(P.N_hidden,batch,'single'), ...
    'x', zeros(P.N_hidden,batch,'single'), ...
    'r', zeros(P.N_hidden,batch,'single'), ...
    'epsilonVoltage', zeros(P.N_hidden,batch,'single'), ...
    'epsilonAdaptation', zeros(P.N_hidden,batch,'single'), ...
    'eligibilityRise', zeros(P.N_hidden,batch,'single'), ...
    'eligibilityDecay', zeros(P.N_hidden,batch,'single'));
end

function hash = sha256_file(file)
engine = javaMethod('getInstance','java.security.MessageDigest','SHA-256');
fid = fopen(file,'r'); assert(fid>=0,'Could not open %s.',file);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
while true
    bytes = fread(fid,1024*1024,'*uint8');
    if isempty(bytes), break; end
    engine.update(bytes);
end
digest = typecast(engine.digest(),'uint8');
hash = lower(reshape(dec2hex(digest).',1,[]));
end
