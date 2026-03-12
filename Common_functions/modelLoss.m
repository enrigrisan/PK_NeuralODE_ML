%% =========================================================
%% LOCAL FUNCTIONS
%% =========================================================

% ---------------------------------------------------------
% MODEL LOSS  (ELBO = reconstruction + beta * KL)
% nets        : struct of dlnetwork architectures (not differentiated)
% learnables  : struct of learnables tables (traced dlarrays -> differentiable)
% ---------------------------------------------------------
function [loss, gradLearnables] = modelLoss(nets, learnables, batch, cfg, train)

N         = numel(batch);
betaKL    = 0.01;
reconLoss = dlarray(single(0));
klLoss    = dlarray(single(0));

%tSpan = dlarray(cfg.tFull, 'T');
tSpan = dlarray(cfg.tFull(2:end), 'T');

% Rebuild networks with current learnables so forward passes are traced
dynEncNet = applyLearnables(nets.dynEnc, learnables.dynEnc);
pkEncNet  = applyLearnables(nets.pkEnc,  learnables.pkEnc);
odeFNet   = applyLearnables(nets.odeF,   learnables.odeF);
decNet    = applyLearnables(nets.dec,    learnables.dec);

for i = 1:N
    pat = batch(i);

    % ---- 1. Dynamic encoder
    seqInput = dlarray([pat.C_full; cfg.tFull], 'CT');
    dynOut   = predict(dynEncNet, seqInput);
    mu_dyn   = dynOut(1             : cfg.dimZdyn);
    lv_dyn   = dynOut(cfg.dimZdyn+1 : end);

    % ---- 2. PK encoder
    sparseInput = dlarray([pat.C_sparse; pat.t_sparse], 'CT');
    pkOut    = predict(pkEncNet, sparseInput);
    mu_pk    = pkOut(1            : cfg.dimZpk);
    lv_pk    = pkOut(cfg.dimZpk+1 : end);

    % ---- 3. Reparameterise z0 = [z_dyn0 | z_pk]
    z_dyn0 = reparameterise(mu_dyn, lv_dyn);
    z_pk0  = reparameterise(mu_pk,  lv_pk);
    z0     = dlarray([z_dyn0; z_pk0], 'CB');

    % ---- 4. Integrate ODE with dlode45 over t=1..48 only
    tSpanInteg = dlarray(cfg.tFull(1:end), 'T');   % t=1 to t=48, 48 points

    odefun = @(t, z, theta) odeFunction(t, z, theta, cfg);

    Z_integrated = dlode45(odefun, tSpanInteg, z0, odeFNet, ...
        'RelativeTolerance', cfg.rtolODE, ...
        'AbsoluteTolerance', cfg.atolODE);
    % Z_integrated: [dimZ x 1 x 48]

    % ---- 5. Prepend z0 at t=0 to get full trajectory [dimZ x 1 x 49]
    z0_cbt = reshape(z0, [cfg.dimZ, 1, 1]);        % [dimZ x 1 x 1]
    Z_full = cat(3, z0_cbt, Z_integrated);         % [dimZ x 1 x 49]

    % ---- 6. Decode z_dyn -> C_hat over all 49 time points
    Z_dyn_raw = Z_full(1:cfg.dimZdyn, 1, :);       % [dimZdyn x 1 x 49]
    Z_dyn     = dlarray(reshape(stripdims(Z_dyn_raw), cfg.dimZdyn, []), 'CT'); % [dimZdyn x 49]
    C_hat     = predict(decNet, Z_dyn);             % [1 x 49]

    % ---- 7. Reconstruction loss
    C_true    = dlarray(pat.C_full, 'CT');          % [1 x 49]
    %reconLoss = reconLoss + mean(((C_hat - C_true)/mean(C_true)).^2, 'all');
    reconLoss = reconLoss + mean((100*(C_hat - C_true)).^2, 'all');

    % ---- 7. KL divergence
    klLoss = klLoss + klDivergence(mu_dyn, lv_dyn) ...
        + klDivergence(mu_pk,  lv_pk);
end

loss = (reconLoss + betaKL * klLoss) / N;

if (train)
    % dlgradient now works: learnables are traced dlarray tables
    gradLearnables = dlgradient(loss, ...
        {learnables.dynEnc.Value, ...
        learnables.pkEnc.Value,  ...
        learnables.odeF.Value,   ...
        learnables.dec.Value});
else
    gradLearnables =0;
end

end
