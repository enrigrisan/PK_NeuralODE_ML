% ---------------------------------------------------------
% LATENT SPACE  (uses reassembled nets after training)
% ---------------------------------------------------------
function [mu_dyn, mu_pk] = ExtractLatent(nets, testPat, cfg)

tSpan = dlarray(cfg.tFull(2:end), 'T');        % 48 points: t=1..48

seqInput    = dlarray([testPat.C_full; cfg.tFull], 'CT');
dynOut      = predict(nets.dynEnc, seqInput);
mu_dyn      = dynOut(1 : cfg.dimZdyn);

sparseInput = dlarray([testPat.C_sparse; testPat.t_sparse], 'CT');
pkOut       = predict(nets.pkEnc, sparseInput);
mu_pk       = pkOut(1 : cfg.dimZpk);

%Z=[extractdata(mu_dyn); extractdata(mu_pk)];