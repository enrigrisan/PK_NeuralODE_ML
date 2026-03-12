% ---------------------------------------------------------
% PREDICTION  (uses reassembled nets after training)
% ---------------------------------------------------------
function C_pred = predictConcentration(nets, testPat, cfg)

tSpan = dlarray(cfg.tFull(2:end), 'T');        % 48 points: t=1..48

seqInput    = dlarray([testPat.C_full; cfg.tFull], 'CT');
dynOut      = predict(nets.dynEnc, seqInput);
mu_dyn      = dynOut(1 : cfg.dimZdyn);

sparseInput = dlarray([testPat.C_sparse; testPat.t_sparse], 'CT');
pkOut       = predict(nets.pkEnc, sparseInput);
mu_pk       = pkOut(1 : cfg.dimZpk);

z0     = dlarray([extractdata(mu_dyn); extractdata(mu_pk)], 'CB');
odefun = @(t, z, theta) odeFunction(t, z, theta, cfg);

Z_integrated = dlode45(odefun, tSpan, z0, nets.odeF, ...
    'RelativeTolerance', cfg.rtolODE, ...
    'AbsoluteTolerance', cfg.atolODE);

z0_cbt    = reshape(z0, [cfg.dimZ, 1, 1]);
Z_full    = cat(3, z0_cbt, Z_integrated);              % [dimZ x 1 x 49]
Z_dyn_raw = Z_full(1:cfg.dimZdyn, 1, :);
Z_dyn     = dlarray(reshape(stripdims(Z_dyn_raw), cfg.dimZdyn, []), 'CT');
C_pred    = double(extractdata(predict(nets.dec, Z_dyn)));
end
