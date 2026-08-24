function report = run_tests(mode)
%RUN_TESTS Mathematical and reproducibility regression tests for BANFF SNN.
%   RUN_TESTS("quick") runs CPU/data tests. RUN_TESTS("full") additionally
%   runs GPU-kernel comparisons, nine-task smoke tests and checkpoint restart.

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
run(@test_dataset_hashes, 'dataset SHA-256');
run(@test_static_splits_and_preprocessing, 'static splits/preprocessing');
run(@test_low_rank_operator, 'low-rank operator and Dale signs');
run(@test_full_state_finite_difference, 'full local-state finite difference');
run(@test_lsti_event_sensitivity, 'LSTI event-time sensitivity');
run(@test_phase_metric_identity, 'phase metric identity');
run(@test_target_integrator_refinement, 'target integrator refinement');
if mode == "full"
    run(@test_gpu_reference_equivalence, 'shared timestep CPU/GPU agreement');
    run(@test_checkpoint_restart, 'checkpoint/restart equivalence');
    run(@test_nine_task_smoke, 'nine-task train/test smoke test');
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
            messages{end+1} = sprintf('%s: %s', ME.identifier, ME.message); %#ok<AGROW>
        end
    end
end

function test_default_config()
cfg = banff("config", "lorenz", struct());
assert(cfg.eligibility_mode == "hard_spike");
assert(abs(double(cfg.hard_event_gain) - 1) < 1e-12);
assert(abs(double(cfg.initial_bias) - 20) < 1e-12);
assert(cfg.N_hidden == 32000 && cfg.N_recurrent == 10);
spsa = banff("config", "breast_cancer", struct('method', "spsa"));
assert(spsa.epochs == 50000);
assert(spsa.spsa_schedule_epochs == 50000);
assert(spsa.learning_rate_schedule_epochs == 50000);
assert(~isfield(spsa, 'spsa_continuation_boundary'));
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

function test_gpu_reference_equivalence()
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

function test_checkpoint_restart()
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

function test_nine_task_smoke()
if ~canUseGPU
    warning('BANFF:testNoGPU','Smoke tests skipped because no supported GPU is available.');
    return;
end
root = tempname; mkdir(root); cleanup = onCleanup(@() rmdir(root,'s')); %#ok<NASGU>
tasks = ["breast_cancer","mnist","afro_mnist_vai","abalone","toyota","yacht", ...
    "lorenz","sprott_s","vanderpol"];
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
        o.test_warmup_time = single(.005);
        o.validation_initial_conditions = 1;
        o.test_initial_conditions = 1;
        o.phase_metric = struct('projections',16,'trim_fraction',0, ...
            'subsample',1,'transient_fraction',0,'max_points',100);
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
