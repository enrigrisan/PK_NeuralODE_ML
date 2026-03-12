% ---------------------------------------------------------
% UPDATE LEARNABLES  –  apply Adam per-network
% gradLearnables is a cell {gradDynEnc, gradPkEnc, gradOdeF, gradDec}
% each element is a cell array of dlarray gradients matching .Value
% ---------------------------------------------------------
function [learnables, avgGrad, avgSqGrad] = updateLearnables( ...
    learnables, gradLearnables, avgGrad, avgSqGrad, iteration, learnRate)

names = {'dynEnc','pkEnc','odeF','dec'};

if isempty(avgGrad)
    avgGrad   = cell(1,4);
    avgSqGrad = cell(1,4);
end

for k = 1:4
    gradsCell = gradLearnables{k};   % cell of dlarray gradients

    % Wrap gradient cell into a table matching the learnables table
    gradTable = learnables.(names{k});
    for j = 1:height(gradTable)
        gradTable.Value{j} = gradsCell{j};
    end

    [learnables.(names{k}), avgGrad{k}, avgSqGrad{k}] = ...
        adamupdate(learnables.(names{k}), gradTable, ...
        avgGrad{k}, avgSqGrad{k}, iteration, learnRate);
end
end