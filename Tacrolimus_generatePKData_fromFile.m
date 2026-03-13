% ---------------------------------------------------------
% DATA GENERATION
% ---------------------------------------------------------
function [data_train, data_test] = Tacrolimus_generatePKData_fromFile(cfg, datafile, test)


M=readmatrix(datafile);
D = 300;
Vd_subj = M(:,12);
ke_subj = M(:,13);
ka = 0.502;
t0 = 0.346;

t    = cfg.tFull(:)';

if test==0
    data_train = struct('C_full',{}, 't_sparse',{}, 'C_sparse',{});
    for i = 1:cfg.nPatients
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

        data_train(i).C_full   = pat.C_full;
        data_train(i).t_sparse = pat.t_sparse;
        data_train(i).C_sparse = pat.C_sparse;
    end

    data_test = struct('C_full',{}, 't_sparse',{}, 'C_sparse',{});
    for i = cfg.nPatients+1:cfg.nPatients+100
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

        data_test(i-cfg.nPatients).C_full   = pat.C_full;
        data_test(i-cfg.nPatients).t_sparse = pat.t_sparse;
        data_test(i-cfg.nPatients).C_sparse = pat.C_sparse;
    end



else

    i=cfg.testSingle;
    that=t-t0
    Cclean = (D/Vd_subj(i)) .* (exp(-ke_subj(i)*that) -exp(-ka*that)).*(t>=t0)

    % Add proportional lognormal noise
    sigma = 0.05;
    C_full = Cclean .* exp(sigma*randn(size(Cclean)));
    C_full = max(C_full, 1e-6);

    % Sparse random observations for PK encoder (10 random time points)
    sparseIdx     = sort(randperm(numel(t), cfg.nSparseObs));
    pat.t_sparse  = single(t(sparseIdx));
    pat.C_sparse  = single(C_full(sparseIdx));
    pat.C_full    = single(C_full);

    data_train.C_full   = pat.C_full;
    data_train.t_sparse = pat.t_sparse;
    data_train.C_sparse = pat.C_sparse;

    data_test=[];
end
