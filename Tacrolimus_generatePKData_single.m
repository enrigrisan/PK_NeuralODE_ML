function pat = Tacrolimus_generatePKData_single(cfg)
    % Random 2-compartment PK parameters (log-normal population)
    CL  = exp(log(5)   + 0.3*randn);   % clearance
    V1  = exp(log(20)  + 0.3*randn);   % central volume
    Q   = exp(log(2)   + 0.3*randn);   % inter-compartmental clearance
    V2  = exp(log(40)  + 0.3*randn);   % peripheral volume
    D   = cfg.D;

    % Analytical solution of 2-cmpt bolus model
    alpha = 0.5*((CL/V1 + Q/V1 + Q/V2) + ...
            sqrt((CL/V1 + Q/V1 + Q/V2)^2 - 4*(CL*Q)/(V1*V2)));
    beta  = 0.5*((CL/V1 + Q/V1 + Q/V2) - ...
            sqrt((CL/V1 + Q/V1 + Q/V2)^2 - 4*(CL*Q)/(V1*V2)));
    A     = (D/V1) * (alpha - Q/V2) / (alpha - beta);
    B     = (D/V1) * (Q/V2 - beta)  / (alpha - beta);

    t    = cfg.tFull(:)';
    Cclean = A*exp(-alpha*t) + B*exp(-beta*t);

    % Add proportional lognormal noise
    sigma = 0.05;
    C_full = Cclean .* exp(sigma*randn(size(Cclean)));
    C_full = max(C_full, 1e-6);

    % Sparse random observations for PK encoder (10 random time points)
    sparseIdx     = sort(randperm(numel(t), cfg.nSparseObs));
    pat.t_sparse  = single(t(sparseIdx));
    pat.C_sparse  = single(C_full(sparseIdx));
    pat.C_full    = single(C_full);
end