% ---------------------------------------------------------
% ODE FUNCTION  –  called internally by dlode45
%
%   t     : scalar dlarray (current time, 'U' format)
%   z     : dlarray [dimZ x batchSize], format 'CB'
%   theta : learned parameters of odeFuncNet
%   cfg   : configuration struct (passed via closure)
%
%   Returns dz: same size and format as z
% ---------------------------------------------------------
function dz = odeFunction(t, z, theta, cfg)
% Augment state with current time so the ODE can be non-autonomous
batchSz  = size(z, 2);
t_rep    = repmat(dlarray(t, 'U'), [1, batchSz]);  % [1 x B]
% Build feature input [dimZ+1 x B], format 'CB'
zt_aug   = dlarray([stripdims(z); t_rep], 'CB');
% Forward pass through the ODE network
dz       = predict(theta, zt_aug);                 % [dimZ x B], 'CB'
end