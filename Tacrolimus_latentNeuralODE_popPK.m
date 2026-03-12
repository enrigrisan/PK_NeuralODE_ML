%% =========================================================
%  Latent Neural ODE for Population Pharmacokinetics
%  Integration via dlode45 (MATLAB R2023b+)
%  Latent space: z = [z_dyn | z_pk]
%    z_dyn  – dynamic state (current & past concentration)
%    z_pk   – patient-level PK characteristics (from sparse obs)
%% =========================================================
addpath('.\Common_functions');
addpath('.\Tacrolimus_data');

datapath='.\Tacrolimus_data';

%% 0. Configuration
rng(42);
cfg.nPatients   = 200;
cfg.testSingle  = 1000;
cfg.tFull       = single(0:1:48);      % [1 x 49], must be single for dlarray
cfg.nSparseObs  = 10;
cfg.D           = 100;
cfg.dimZdyn     = 8;
cfg.dimZpk      = 4;
cfg.dimZ        = cfg.dimZdyn + cfg.dimZpk;
cfg.dimHidden   = 64;
cfg.nEpochs     = 500;
cfg.lrInit      = 1e-3;
cfg.batchSize   = 16;
cfg.rtolODE     = 1e-3;      % relative tolerance for dlode45
cfg.atolODE     = 1e-4;      % absolute tolerance for dlode45

%% 1. Synthetic PK Data
fprintf('Generating synthetic PK data...\n');
datafile=fullfile(datapath,'dataset2_covariates.csv');
[data, data_val] = generatePKData_fromFile(cfg, datafile, 0);
%data = generatePKData(cfg);

%% 2. Build Network Components
fprintf('Building network components...\n');

% 2a. Dynamic encoder: GRU over full sequence [C(t); t] -> [mu_dyn; logvar_dyn]
dynEncoderNet = buildDynEncoder(2, cfg.dimZdyn * 2, cfg.dimHidden);

% 2b. PK encoder: permutation-invariant set encoder -> [mu_pk; logvar_pk]
pkEncoderNet  = buildPKEncoder(2, cfg.dimZpk  * 2, cfg.dimHidden);

% 2c. ODE function: f(t, z, theta) -> dz/dt
%     Input to the network: [z; t] (dimZ+1 features)
%     dlode45 will call this at arbitrary time points
odeFuncNet    = buildODEFunc(cfg.dimZ, cfg.dimHidden);

% 2d. Decoder: z_dyn(t) -> C_hat(t), applied pointwise over sequence
decoderNet    = buildDecoder(cfg.dimZdyn, cfg.dimHidden);

%% 3. Pack learnable parameters using cell arrays of dlarray (traceable)
params.dynEnc = dynEncoderNet;
params.pkEnc  = pkEncoderNet;
params.odeF   = odeFuncNet;
params.dec    = decoderNet;

%     Extract learnables from each dlnetwork explicitly
learnables.dynEnc = params.dynEnc.Learnables;
learnables.pkEnc  = params.pkEnc.Learnables;
learnables.odeF   = params.odeF.Learnables;
learnables.dec    = params.dec.Learnables;

% Keep network architectures (non-learnable structure) separately
nets.dynEnc = params.dynEnc;
nets.pkEnc  = params.pkEnc;
nets.odeF   = params.odeF;
nets.dec    = params.dec;

%% 4. Training Loop (Adam optimiser)
[nets,lossHistory,val_lossHistory]=TrainNODE(nets,data,data_val,cfg);

% fprintf('Starting training...\n');
% avgGrad   = [];
% avgSqGrad = [];
% 
% 
% lossHistory = zeros(cfg.nEpochs, 1);
% val_lossHistory = zeros(cfg.nEpochs, 1);
% 
% for epoch = 1:cfg.nEpochs
% 
%     idx   = randperm(cfg.nPatients, cfg.batchSize);
%     batch = data(idx);
% 
%     % dlfeval traces learnables (tables of dlarray) -> dlgradient works
%     [loss, gradLearnables] = dlfeval(@modelLoss, nets, learnables, batch, cfg, 1);
%     [val_loss, ~] = dlfeval(@modelLoss, nets, learnables, data_val, cfg, 0);
% 
%     % Flatten all learnables and gradients into a single cell for adamupdate
%     [learnables, avgGrad, avgSqGrad] = updateLearnables( ...
%         learnables, gradLearnables, avgGrad, avgSqGrad, epoch, cfg.lrInit);
% 
% 
%     lossHistory(epoch) = double(extractdata(loss));
%     val_lossHistory(epoch) = double(extractdata(val_loss));
%     if mod(epoch, 20) == 0
%         fprintf('Epoch %3d | Loss = %.4f | Validation Loss = %.4f \n', epoch, lossHistory(epoch), val_lossHistory(epoch));
%     end
% end
% 
% % After training, reassemble full dlnetworks with updated weights
% nets.dynEnc = applyLearnables(nets.dynEnc, learnables.dynEnc);
% nets.pkEnc  = applyLearnables(nets.pkEnc,  learnables.pkEnc);
% nets.odeF   = applyLearnables(nets.odeF,   learnables.odeF);
% nets.dec    = applyLearnables(nets.dec,    learnables.dec);

%% 5. Evaluation
fprintf('\nEvaluating on a test patient...\n');
%testPat = generatePKData_fromFile(cfg, datafile, 1);
%testPat = generatePKData_single(cfg);

figure;
subplot(1,2,1);
plot(log(lossHistory), 'LineWidth', 1.5, 'DisplayName', 'Train Loss');
hold on
plot(log(val_lossHistory), 'LineWidth', 1.5, 'DisplayName', 'Validation Loss');
xlabel('Epoch'); ylabel('ELBO Loss');
legend;
title('Loss'); grid on;

pos=[3,4,7,8];
for tp = 1:length(pos)
    idx = randi([1,length(data_val)]);
    testPat=data_val(idx);
    C_pred  = predictConcentration(nets, testPat, cfg);

    subplot(2,4,pos(tp));
    plot(cfg.tFull, testPat.C_full,  'ko-', 'DisplayName', 'True C(t)'); hold on;
    plot(cfg.tFull(2:end), C_pred,          'r-',  'LineWidth', 2, ...
        'DisplayName', 'Predicted \hat{C}(t)');
    scatter(testPat.t_sparse, testPat.C_sparse, 60, 'b', 'filled', ...
        'DisplayName', 'Sparse obs (PK enc.)');
    xlabel('Time (h)'); ylabel('Concentration');
    %legend; grid on;
    title('Latent Neural ODE – PK Prediction');
end
% ---------------------------------------------------------
% REPARAMETERISE  z = mu + eps * exp(0.5 * logvar)
% ---------------------------------------------------------
function z = reparameterise(mu, logvar)
eps = dlarray(randn(size(mu), 'single'));
z   = mu + exp(0.5 * logvar) .* eps;
end




%% =========================================================
%% NETWORK BUILDERS
%% =========================================================

function net = buildDynEncoder(dimIn, dimOut, dimH)
% GRU processes the full [C(t); t] sequence, returns last hidden state
layers = [
    sequenceInputLayer(dimIn,  'Name', 'seqIn')
    gruLayer(dimH, 'Name', 'gru1', 'OutputMode', 'last')
    fullyConnectedLayer(dimH,  'Name', 'fc1')
    tanhLayer('Name', 'tanh1')
    fullyConnectedLayer(dimOut,'Name', 'fcOut')   % -> [mu_dyn; logvar_dyn]
    ];
net = dlnetwork(layerGraph(layers));
end

function net = buildPKEncoder(dimIn, dimOut, dimH)
% Permutation-invariant: pointwise FC per observation, then mean pool
layers = [
    sequenceInputLayer(dimIn,  'Name', 'sparseIn')
    fullyConnectedLayer(dimH,  'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(dimH,  'Name', 'fc2')
    reluLayer('Name', 'relu2')
    meanPoolSequenceLayer('Name', 'meanPool')     % custom layer
    fullyConnectedLayer(dimOut,'Name', 'fcOut')   % -> [mu_pk; logvar_pk]
    ];
net = dlnetwork(layerGraph(layers));
end

function net = buildODEFunc(dimZ, dimH)
% f([z; t]) -> dz/dt   Input dim = dimZ + 1 (time-augmented)
layers = [
    featureInputLayer(dimZ + 1, 'Name', 'odeIn')
    fullyConnectedLayer(dimH,   'Name', 'fc1')
    tanhLayer('Name', 'tanh1')
    fullyConnectedLayer(dimH,   'Name', 'fc2')
    tanhLayer('Name', 'tanh2')
    fullyConnectedLayer(dimH,   'Name', 'fc3')
    tanhLayer('Name', 'tanh3')
    fullyConnectedLayer(dimZ,   'Name', 'fcOut')
    ];
net = dlnetwork(layerGraph(layers));
end

function net = buildDecoder(dimZdyn, dimH)
% Pointwise decoder applied over the sequence dimension
layers = [
    sequenceInputLayer(dimZdyn, 'Name', 'decIn')
    fullyConnectedLayer(dimH,   'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(dimH,   'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(1,      'Name', 'fcOut')
    softplusLayer('Name', 'sp') % enforce C_hat >= 0
    ];
net = dlnetwork(layerGraph(layers));
end

