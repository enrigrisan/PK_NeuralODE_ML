%% =========================================================
%  Latent Neural ODE for Population Pharmacokinetics
%  Integration via dlode45 (MATLAB R2023b+)
%  Latent space: z = [z_dyn | z_pk]
%    z_dyn  – dynamic state (current & past concentration)
%    z_pk   – patient-level PK characteristics (from sparse obs)
%% =========================================================
function [mu_dyn, mu_pk]=Population_Latent(datafile, nets, nPatients, cfg)

%% 1. Synthetic PK Data
fprintf('Generating synthetic PK data...\n');
M=readmatrix(datafile);
D = 300;
Vd_subj = M(:,12);
ke_subj = M(:,13);
ka = 0.502;
t0 = 0.346;

t    = cfg.tFull(:)';

mu_dyn=zeros(nPatients,8);
mu_pk=zeros(nPatients,4);

for i = 1:nPatients
    that=t-t0;
    Cclean = (D/Vd_subj(i)) .* (exp(-ke_subj(i)*that) -exp(-ka*that)).*(t>=t0);

    % Add proportional lognormal noise
    sigma = 0.05;
    C_full = Cclean .* exp(sigma*randn(size(Cclean)));
    C_full = max(C_full, 1e-6);

    % Sparse random observations for PK encoder (10 random time points)
    sparseIdx     = sort(randperm(numel(t), cfg.nSparseObs));
    pat.t_sparse  = single(t(sparseIdx));
    pat.C_sparse  = single(C_full(sparseIdx));
    pat.C_full    = single(C_full);


    [mu_dyn(i,:), mu_pk(i,:)] = ExtractLatent(nets, pat, cfg);
end