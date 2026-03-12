%% =========================================================================
%  Latent Neural ODE for Pharmacokinetics (PK) Modelling
%  =========================================================================
%  Architecture (VAE-style Latent Neural ODE):
%
%    Observed data  C(t)  – drug concentration, sparse per subject
%
%    ENCODER  (RNN / recognition network)
%       Maps sparse, irregularly-sampled {C(t_i)} → posterior
%       q(z0 | C) = N(mu_z0, sigma_z0)
%       Input fed in REVERSE time order (standard practice for ODE-RNN)
%
%    LATENT DYNAMICS  (Neural ODE in latent space)
%       dz/dt = f_theta(z)        z ∈ R^latentDim
%       Solved with dlode45 (Deep Learning Toolbox)
%
%    DECODER  (MLP)
%       C_hat(t) = g_phi(z(t))    projects latent → observed space
%
%    LOSS  =  Reconstruction MSE  +  beta * KL( q || N(0,I) )
%
%  Requirements: MATLAB R2023b+, Deep Learning Toolbox
%  =========================================================================
clear; clc; close all;
rng(42);

%% -------------------------------------------------------------------------
%% 0.  HYPER-PARAMETERS
%% -------------------------------------------------------------------------
cfg.nSubjects   = 1000;        % patients
cfg.tFull       = (0:48)';   % full time grid (h)
cfg.nObs        = 10;        % observed time points per subject (training)
cfg.latentDim   = 4;         % dimension of latent state z
cfg.encoderHid  = 32;        % hidden units in encoder RNN
cfg.odeHid      = 32;        % hidden units in ODE function network
cfg.decoderHid  = 32;        % hidden units in decoder MLP
cfg.beta        = 1e-3;      % KL weight (beta-VAE)
cfg.learnRate   = 5e-4;
cfg.numEpochs   = 5000;
cfg.batchSize   = cfg.nSubjects;   % full-batch (small dataset)
cfg.odeSolver   = 'rk4';          % 'rk4' | 'euler' (faster) | 'dopri5'
cfg.odeRelTol   = 1e-3;
cfg.odeAbsTol   = 1e-5;

%% -------------------------------------------------------------------------
%% 1.  SIMULATE SYNTHETIC PK DATA  (replace with your real dataset)
%% -------------------------------------------------------------------------
%fprintf('Generating synthetic PK data...\n');
%D       = 100;                                        % dose (mg)
%Vd_pop  = 10;  ke_pop  = 0.08;                       % population parameters
%Vd_subj = Vd_pop + 1.5*randn(cfg.nSubjects,1);
%ke_subj = ke_pop + 0.01*abs(randn(cfg.nSubjects,1));

M=readmatrix('dataset2_covariates.csv');
D = 300;
Vd_subj = M(:,12);
ke_subj = M(:,13);
ka = 0.502;
t0 = 0.346;

C_full = zeros(length(cfg.tFull), cfg.nSubjects);
t = cfg.tFull-t0;
for s = 1:cfg.nSubjects
    C_full(:,s) = (D/Vd_subj(s)) .* (exp(-ke_subj(s)*t) -exp(-ka*t)) ...
                  + 0.001*randn(length(cfg.tFull),1);
    C_full(:,s) = max(C_full(:,s), 0);
end

% Normalise concentrations to [0,1] for stable training
C_max   = max(C_full(:));
C_norm  = C_full / C_max;

%% -------------------------------------------------------------------------
%% 2.  INITIALISE NETWORK PARAMETERS
%% -------------------------------------------------------------------------
fprintf('Initialising networks...\n');

%  ---- 2a. Encoder: GRU cell  (processes reversed time series)  -----------
%           Input at each step: [C_obs(t); delta_t]  → 2 features
%           After last step: linear head → [mu_z0; log_sigma_z0]
encInputDim = 2;   % [C value, time gap]
[encParams, encSizes] = initGRU(encInputDim, cfg.encoderHid);
[encHeadParams, encHeadSizes] = initMLP(cfg.encoderHid, [], 2*cfg.latentDim);
%   output: first latentDim → mu, last latentDim → log_sigma

%  ---- 2b. ODE function:  f_theta(z) → dz/dt  ----------------------------
%           Simple MLP with softplus activations
[odeParams, odeSizes] = initMLP(cfg.latentDim, ...
                                [cfg.odeHid, cfg.odeHid], ...
                                cfg.latentDim);

%  ---- 2c. Decoder:  g_phi(z) → C_hat  -----------------------------------
[decParams, decSizes] = initMLP(cfg.latentDim, ...
                                [cfg.decoderHid], ...
                                1);

%  Pack everything into a struct of dlarrays
params.enc     = encParams;
params.encHead = encHeadParams;
params.ode     = odeParams;
params.dec     = decParams;

fprintf('  Encoder GRU params : %d\n', countParams(encParams));
fprintf('  ODE network params : %d\n', countParams(odeParams));
fprintf('  Decoder params     : %d\n', countParams(decParams));

%% -------------------------------------------------------------------------
%% 3.  TRAINING LOOP  (Adam optimiser)
%% -------------------------------------------------------------------------
fprintf('\nStarting training...\n');

% Adam state
adamState.avgGrad   = [];
adamState.avgSqGrad = [];
beta1 = 0.9;  beta2 = 0.999;  eps_ = 1e-8;

lossHistory    = zeros(cfg.numEpochs, 1);
reconHistory   = zeros(cfg.numEpochs, 1);
klHistory      = zeros(cfg.numEpochs, 1);

for epoch = 1:cfg.numEpochs

    %-- Sample 10 random observation time points per subject (skip t=0)
    obsIdx = zeros(cfg.nSubjects, cfg.nObs);
    for s = 1:cfg.nSubjects
        obsIdx(s,:) = sort(randperm(length(cfg.tFull)-1, cfg.nObs) + 1);
    end

    %-- Forward + backward pass via dlfeval
    [loss, recon, kl, grads] = dlfeval(@latentODEloss, ...
        params, C_norm, cfg.tFull, obsIdx, cfg);

    %-- Adam update
    [params, adamState] = adamUpdate(params, grads, adamState, ...
        epoch, cfg.learnRate, beta1, beta2, eps_);

    lossHistory(epoch)  = double(extractdata(loss));
    reconHistory(epoch) = double(extractdata(recon));
    klHistory(epoch)    = double(extractdata(kl));

    if mod(epoch,50)==0 || epoch==1
        fprintf('  Epoch %4d | Loss=%8.4f | Recon=%8.4f | KL=%8.4f\n', ...
            epoch, lossHistory(epoch), reconHistory(epoch), klHistory(epoch));
    end
end

%% -------------------------------------------------------------------------
%% 4.  PREDICTION ON FULL TIME GRID
%% -------------------------------------------------------------------------
fprintf('\nGenerating predictions...\n');

C_pred_norm = zeros(length(cfg.tFull), cfg.nSubjects);

% Use all available time points for encoding at test time
obsIdxFull = repmat((1:length(cfg.tFull))', 1, cfg.nSubjects)';

for s = 1:cfg.nSubjects
    % Encode
    t_s   = cfg.tFull;
    C_s   = C_norm(:, s);
    [mu, ~] = encode(params, C_s, t_s, t_s, cfg);

    % Solve ODE in latent space
    z0    = mu;   % use mean at test time (no sampling)
    tspan = dlarray([cfg.tFull(1), cfg.tFull(end)]);
    odefn = @(t,z,p) odeFunction(t, z, p, odeSizes, cfg.odeHid);
    zSol  = dlode45(odefn, tspan, dlarray(z0), params.ode, ...
                    'DataFormat','CB',...
                    'RelativeTolerance', cfg.odeRelTol, ...
                    'AbsoluteTolerance', cfg.odeAbsTol);

    % Decode at every time step
    % zSol is latentDim x nTimeSteps (evaluated at solver-chosen steps)
    % We re-integrate to get solution at tFull exactly
    zAtFull = solveODEatTimes(params.ode, z0, cfg.tFull, odefn, cfg);
    for ti = 1:length(cfg.tFull)
        C_pred_norm(ti,s) = double(extractdata( ...
            decode(params, zAtFull(:,ti), decSizes, cfg)));
    end
end

C_pred = C_pred_norm * C_max;   % undo normalisation

%% -------------------------------------------------------------------------
%% 5.  PLOTS
%% -------------------------------------------------------------------------

% 5a. Training curves
figure('Name','Training History','Position',[100 100 900 300]);
subplot(1,3,1);
semilogy(lossHistory,'LineWidth',1.5,'Color',[0.2 0.4 0.8]);
xlabel('Epoch'); ylabel('Total Loss'); title('Total Loss'); grid on;

subplot(1,3,2);
semilogy(reconHistory,'LineWidth',1.5,'Color',[0.2 0.7 0.3]);
xlabel('Epoch'); ylabel('Recon Loss'); title('Reconstruction MSE'); grid on;

subplot(1,3,3);
plot(klHistory,'LineWidth',1.5,'Color',[0.8 0.3 0.2]);
xlabel('Epoch'); ylabel('KL Loss'); title('KL Divergence'); grid on;
sgtitle('Latent Neural ODE – Training History');

% 5b. PK concentration predictions
nPlot = min(cfg.nSubjects, 12);
figure('Name','PK Predictions','Position',[100 450 1400 700]);
for s = 1:nPlot
    subplot(3,4,s);
    plot(cfg.tFull, C_full(:,s), 'ko', 'MarkerSize',3,...
         'MarkerFaceColor','k','DisplayName','Observed'); hold on;
    plot(cfg.tFull, C_pred(:,s), 'b-', 'LineWidth',1.8,...
         'DisplayName','Latent Neural ODE');
    xlabel('Time (h)'); ylabel('C (mg/L)');
    title(sprintf('Subject %d',s));
    if s==1, legend('Location','northeast'); end
    ylim([0, max(C_full(:,s))*1.25]);
    grid on;
end
sgtitle('Latent Neural ODE – PK Predictions (Full 0-48h)');

% 5c. Latent space trajectories for first subject
figure('Name','Latent Trajectories');
s = 1;
t_s   = cfg.tFull;
C_s   = C_norm(:, s);
[mu, ~] = encode(params, C_s, t_s, t_s, cfg);
zFull = solveODEatTimes(params.ode, mu, cfg.tFull, ...
    @(t,z,p) odeFunction(t,z,p,odeSizes,cfg.odeHid), cfg);
zFull_data = double(extractdata(zFull));
for d = 1:cfg.latentDim
    subplot(2, ceil(cfg.latentDim/2), d);
    plot(cfg.tFull, zFull_data(d,:), 'LineWidth',1.5);
    xlabel('Time (h)'); ylabel(sprintf('z_%d',d));
    title(sprintf('Latent dim %d (Subject 1)',d));
    grid on;
end
sgtitle('Latent Space Trajectories');

fprintf('\nDone.\n');

%% =========================================================================
%% MAIN LOSS FUNCTION  (called inside dlfeval for autodiff)
%% =========================================================================
function [loss, reconLoss, klLoss, grads] = latentODEloss(...
        params, C_norm, tFull, obsIdx, cfg)

    nSubj   = size(C_norm, 2);
    reconLoss = dlarray(0);
    klLoss    = dlarray(0);
    odefn     = @(t,z,p) odeFunction(t, z, p, [], cfg.odeHid);

    for s = 1:nSubj
        idx_s = obsIdx(s,:);
        t_obs = tFull(idx_s);
        C_obs = dlarray(C_norm(idx_s, s)');   % 1 x nObs

        % --- Encode: RNN over observed points (reversed) ------------------
        [mu, logSigma] = encode(params, dlarray(C_norm(:,s)), ...
                                tFull, t_obs, cfg);

        % --- Reparameterisation trick  z0 ~ N(mu, sigma^2) ---------------
        sigma = exp(0.5 * logSigma);
        z0    = mu + sigma .* randn(size(mu), 'like', mu);   % latentDim x 1

        % --- Solve Neural ODE in latent space from t=0 to t=48 -----------
        tspan = dlarray([tFull(1), tFull(end)]);
        zSol  = dlode45(odefn, tspan, dlarray(z0), params.ode, ...
                        'DataFormat',"CB",...
                        'RelativeTolerance', cfg.odeRelTol, ...
                        'AbsoluteTolerance', cfg.odeAbsTol);

        % Evaluate at observed time points (linearly interpolate solver output)
        % dlode45 returns solution at adaptive steps; use a simple wrapper
        zAtObs = interpolateLatent(zSol, tFull, idx_s);

        % --- Decode -------------------------------------------------------
        C_hat = decodeMulti(params, zAtObs, cfg);  % 1 x nObs

        % --- Reconstruction loss (MSE) ------------------------------------
        reconLoss = reconLoss + mean((C_hat - C_obs).^2, 'all');

        % --- KL divergence  KL( N(mu,sigma^2) || N(0,I) ) ----------------
        klLoss = klLoss + 0.5 * sum(mu.^2 + exp(logSigma) - logSigma - 1);
    end

    reconLoss = reconLoss / nSubj;
    klLoss    = klLoss    / nSubj;
    loss      = reconLoss + cfg.beta * klLoss;

    grads = dlgradient(loss, params);
end

%% =========================================================================
%% NETWORK FORWARD PASSES
%% =========================================================================

function dz = odeFunction(~, z, p, ~, hidDim)
%  f_theta(z) → dz/dt    (autonomous: no explicit time dependence)
%  p is the flat parameter struct for the ODE MLP
    dz = mlpForward(z, p, hidDim, 'tanh');
end

function [mu, logSigma] = encode(params, C_full, tFull, t_obs, cfg)
%  Encode a (possibly irregularly sampled) time series via a GRU
%  Run the GRU in REVERSE time; input at each step = [C(t); delta_t/48]

    nObs = length(t_obs);
    % Map t_obs to indices in tFull
    [~, obsIdx] = ismember(t_obs, tFull);
    if any(obsIdx==0)
        obsIdx = round(interp1(tFull, 1:length(tFull), t_obs, 'linear','extrap'));
    end

    % Reverse ordering
    revIdx  = flip(obsIdx);
    revT    = flip(t_obs);

    h = dlarray(zeros(cfg.encoderHid, 1));   % GRU hidden state

    for k = 1:nObs
        ti   = revIdx(k);
        Cval = C_full(ti);
        if k < nObs
            dt = (revT(k) - revT(k+1)) / 48;
        else
            dt = 0;
        end
        x = dlarray([Cval; dt]);
        h = gruStep(x, h, params.enc, cfg.encoderHid);
    end

    % Linear head: h → [mu; log_sigma]
    % encHead was initialised by initMLP: weights live in p.W{i} cells.
    % Use mlpForward to avoid invalid dlarray-vs-cell multiplication.
    out      = mlpForward(h, params.encHead, [], 'linear');
    mu       = out(1:cfg.latentDim);
    logSigma = out(cfg.latentDim+1:end);
    % Clamp log_sigma for stability
    logSigma = max(min(logSigma, dlarray(2)), dlarray(-4));
end

function C_hat = decode(params, z, decSizes, cfg)
%  Decode a single latent vector z (latentDim x 1) → scalar C_hat
    C_hat = mlpForward(z, params.dec, cfg.decoderHid, 'softplus');
    C_hat = sigmoid(C_hat);   % output in (0,1) since C_norm ∈ [0,1]
end

function C_hat = decodeMulti(params, Z, cfg)
%  Decode multiple latent vectors Z (latentDim x nPts) → 1 x nPts
    nPts  = size(Z, 2);
    C_hat = dlarray(zeros(1, nPts));
    for k = 1:nPts
        C_hat(1,k) = decode(params, Z(:,k), [], cfg);
    end
end

%% =========================================================================
%% GRU CELL
%% =========================================================================

function h_new = gruStep(x, h, p, hidDim)
%  One GRU step.  p contains Wz,Wr,Wh,Uz,Ur,Uh,bz,br,bh
    z   = sigmoid(p.Wz*x + p.Uz*h + p.bz);    % update gate
    r   = sigmoid(p.Wr*x + p.Ur*h + p.br);    % reset gate
    h_c = tanh(   p.Wh*x + p.Uh*(r.*h) + p.bh); % candidate
    h_new = (1-z).*h + z.*h_c;
end

%% =========================================================================
%% HELPER: solve ODE and evaluate at specific times
%% =========================================================================

function zAtTimes = solveODEatTimes(odeParams, z0, tGrid, odefn, cfg)
%  Solve the ODE and return latent state at every point in tGrid
%  Uses simple fixed-step RK4 for guaranteed evaluation at all grid points

    nT   = length(tGrid);
    zCur = dlarray(z0);
    zAtTimes = dlarray(zeros(length(z0), nT));
    zAtTimes(:,1) = zCur;

    for k = 1:nT-1
        dt = tGrid(k+1) - tGrid(k);
        zCur = rk4Step(odefn, tGrid(k), zCur, odeParams, dt);
        zAtTimes(:,k+1) = zCur;
    end
end

function y_next = rk4Step(f, t, y, p, dt)
    k1 = f(t,       y,          p);
    k2 = f(t+dt/2,  y+dt/2*k1,  p);
    k3 = f(t+dt/2,  y+dt/2*k2,  p);
    k4 = f(t+dt,    y+dt*k3,    p);
    y_next = y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
end

function zAtObs = interpolateLatent(zSol, tFull, obsIdx)
%  zSol from dlode45 is evaluated at adaptive time steps internally.
%  Since dlode45 (as of R2023b) returns the solution evaluated at the
%  time points defined by the tspan endpoints, we reconstruct using
%  the rk4 approach above.  This function simply indexes tFull.
%  (For continuous interpolation, use deval-equivalent in DLT.)
    nObs    = length(obsIdx);
    latDim  = size(zSol, 1);
    % zSol has been solved over full [0,48]; it returns 2 columns (endpoints)
    % We rely on solveODEatTimes for pointwise evaluation; this function
    % is kept as a placeholder for dlode45-based continuous interpolation.
    zAtObs  = zSol(:, min(obsIdx, size(zSol,2)));
end

%% =========================================================================
%% MLP FORWARD PASS (generic)
%% =========================================================================

function y = mlpForward(x, p, ~, activation)
%  Forward pass through an MLP stored in p.W{i}, p.b{i}
    nLayers = length(p.W);
    y = x;
    for i = 1:nLayers
        y = p.W{i} * y + p.b{i};
        if i < nLayers
            switch activation
                case 'tanh',     y = tanh(y);
                case 'relu',     y = max(y, dlarray(0));
                case 'softplus', y = log(1 + exp(y));
                case 'linear'    % no activation
            end
        end
    end
end

%% =========================================================================
%% NETWORK INITIALISATION
%% =========================================================================

function [p, sizes] = initGRU(inputDim, hidDim)
    sc = @(r,c) dlarray(0.1*randn(r,c));
    p.Wz = sc(hidDim, inputDim);  p.Uz = sc(hidDim,hidDim); p.bz = dlarray(zeros(hidDim,1));
    p.Wr = sc(hidDim, inputDim);  p.Ur = sc(hidDim,hidDim); p.br = dlarray(zeros(hidDim,1));
    p.Wh = sc(hidDim, inputDim);  p.Uh = sc(hidDim,hidDim); p.bh = dlarray(zeros(hidDim,1));
    sizes = struct('inputDim',inputDim,'hidDim',hidDim);
end

function [p, sizes] = initMLP(inDim, hidDims, outDim)
    dims   = [inDim, hidDims, outDim];
    nLayer = length(dims)-1;
    p.W    = cell(nLayer,1);
    p.b    = cell(nLayer,1);
    for i = 1:nLayer
        sc     = sqrt(2/dims(i));   % He init
        p.W{i} = dlarray(sc * randn(dims(i+1), dims(i)));
        p.b{i} = dlarray(zeros(dims(i+1), 1));
    end
    sizes = struct('dims', dims, 'nLayers', nLayer, ...
                   'inDim', inDim, 'hidDims', hidDims, 'outDim', outDim);
end

%% =========================================================================
%% ADAM UPDATE (handles nested structs of dlarrays)
%% =========================================================================

function [params, state] = adamUpdate(params, grads, state, t, lr, b1, b2, eps)
    if isempty(state.avgGrad)
        state.avgGrad   = zeroLike(grads);
        state.avgSqGrad = zeroLike(grads);
    end
    [params, state.avgGrad, state.avgSqGrad] = ...
        adamUpdateStruct(params, grads, state.avgGrad, state.avgSqGrad, ...
                         t, lr, b1, b2, eps);
end

function [p, m, v] = adamUpdateStruct(p, g, m, v, t, lr, b1, b2, eps)
    if isstruct(p)
        fns = fieldnames(p);
        for k = 1:numel(fns)
            [p.(fns{k}), m.(fns{k}), v.(fns{k})] = ...
                adamUpdateStruct(p.(fns{k}), g.(fns{k}), ...
                                 m.(fns{k}), v.(fns{k}), t, lr, b1, b2, eps);
        end
    elseif iscell(p)
        for k = 1:numel(p)
            [p{k}, m{k}, v{k}] = adamUpdateStruct(p{k}, g{k}, m{k}, v{k}, ...
                                                    t, lr, b1, b2, eps);
        end
    elseif isa(p,'dlarray')
        m  = b1*m + (1-b1)*g;
        v  = b2*v + (1-b2)*g.^2;
        mh = m  / (1-b1^t);
        vh = v  / (1-b2^t);
        p  = p - lr * mh ./ (sqrt(vh) + eps);
    end
end

function z = zeroLike(x)
    if isstruct(x)
        fns = fieldnames(x);
        for k = 1:numel(fns)
            z.(fns{k}) = zeroLike(x.(fns{k}));
        end
    elseif iscell(x)
        z = cell(size(x));
        for k = 1:numel(x)
            z{k} = zeroLike(x{k});
        end
    elseif isa(x,'dlarray')
        z = dlarray(zeros(size(x)));
    else
        z = zeros(size(x));
    end
end

%% =========================================================================
%% UTILITY
%% =========================================================================

function n = countParams(p)
    n = 0;
    if isstruct(p)
        fns = fieldnames(p);
        for k = 1:numel(fns), n = n + countParams(p.(fns{k})); end
    elseif iscell(p)
        for k = 1:numel(p), n = n + countParams(p{k}); end
    elseif isa(p,'dlarray') || isnumeric(p)
        n = numel(p);
    end
end
