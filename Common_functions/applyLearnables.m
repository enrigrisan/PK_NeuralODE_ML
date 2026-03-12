% ---------------------------------------------------------
% HELPER: inject a learnables table into a dlnetwork
% Replaces applyLearnables(net, table) which has changed across
% MATLAB versions and expects exact table schema matching.
% ---------------------------------------------------------
function net = applyLearnables(net, learnTable)
for j = 1:height(learnTable)
    net.Learnables.Value{j} = learnTable.Value{j};
end
end