% ---------------------------------------------------------
% KL DIVERGENCE   KL( q(z|x) || N(0,I) )
% ---------------------------------------------------------
function kl = klDivergence(mu, logvar)
kl = -0.5 * sum(1 + logvar - mu.^2 - exp(logvar), 'all');
end