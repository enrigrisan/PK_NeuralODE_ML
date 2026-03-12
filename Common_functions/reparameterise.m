% ---------------------------------------------------------
% REPARAMETERISE  z = mu + eps * exp(0.5 * logvar)
% ---------------------------------------------------------
function z = reparameterise(mu, logvar)
eps = dlarray(randn(size(mu), 'single'));
z   = mu + exp(0.5 * logvar) .* eps;
end