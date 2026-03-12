% ---------------------------------------------------------
% DATA GENERATION
% ---------------------------------------------------------
function data = generatePKData(cfg)
    data = struct('C_full',{}, 't_sparse',{}, 'C_sparse',{});
    for i = 1:cfg.nPatients
        pat = generatePKData_single(cfg);
        data(i).C_full   = pat.C_full;
        data(i).t_sparse = pat.t_sparse;
        data(i).C_sparse = pat.C_sparse;
    end
end