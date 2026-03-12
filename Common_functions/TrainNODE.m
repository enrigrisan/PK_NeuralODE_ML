function [nets,lossHistory,val_lossHistory]=TrainNODE(nets,data,data_val,cfg)

%     Extract learnables from each dlnetwork explicitly
learnables.dynEnc = nets.dynEnc.Learnables;
learnables.pkEnc  = nets.pkEnc.Learnables;
learnables.odeF   = nets.odeF.Learnables;
learnables.dec    = nets.dec.Learnables;

%% 4. Training Loop (Adam optimiser)
fprintf('Starting training...\n');
avgGrad   = [];
avgSqGrad = [];
lossHistory = zeros(cfg.nEpochs, 1);
val_lossHistory = zeros(cfg.nEpochs, 1);

for epoch = 1:cfg.nEpochs

    idx   = randperm(cfg.nPatients, cfg.batchSize);
    batch = data(idx);

    % dlfeval traces learnables (tables of dlarray) -> dlgradient works
    [loss, gradLearnables] = dlfeval(@modelLoss, nets, learnables, batch, cfg, 1);
    [val_loss, ~] = dlfeval(@modelLoss, nets, learnables, data_val, cfg, 0);

    % Flatten all learnables and gradients into a single cell for adamupdate
    [learnables, avgGrad, avgSqGrad] = updateLearnables( ...
        learnables, gradLearnables, avgGrad, avgSqGrad, epoch, cfg.lrInit);


    lossHistory(epoch) = double(extractdata(loss));
    val_lossHistory(epoch) = double(extractdata(val_loss));
    if mod(epoch, 20) == 0
        fprintf('Epoch %3d | Loss = %.4f | Validation Loss = %.4f \n', epoch, lossHistory(epoch), val_lossHistory(epoch));
    end
end

% After training, reassemble full dlnetworks with updated weights
nets.dynEnc = applyLearnables(nets.dynEnc, learnables.dynEnc);
nets.pkEnc  = applyLearnables(nets.pkEnc,  learnables.pkEnc);
nets.odeF   = applyLearnables(nets.odeF,   learnables.odeF);
nets.dec    = applyLearnables(nets.dec,    learnables.dec);

