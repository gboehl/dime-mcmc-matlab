function [chains, lprobs, prop_mean, prop_cov] = dime_mcmc(log_prob, init, niter, sigma=1e-5, gammah=nan, aimh_prob=0.1, df_proposal_dist=10, progress=true)
% function [chains, lprobs, prop_mean, prop_cov] = dime_mcmc(log_prob, init, niter, sigma=1e-5, gammah=nan, aimh_prob=0.1, df_proposal_dist=10, progress=true) 
% DIME MCMC sampling
%
% INPUTS
%   o log_prob          [function]  (vectorized) function returning the log-density
%   o init              [array]     The initial ensemble
%   o niter             [int]       The number of iterations to be run.
%   o sigma             [float]     The standard deviation of the Gaussian used to stretch the proposal vector.
%   o gamma             [float]     The mean stretch factor for the proposal vector. By default, it is 2.38 / sqrt(2 ndim) as recommended by ter Braak (2006): http://www.stat.columbia.edu/~gelman/stuff_for_blog/cajo.pdf
%   o aimh_prob         [float]     Probability to draw an adaptive independence Metropolis Hastings (AIMH) proposal. By default this is set to 0.1.
%   o df_proposal_dist  [float]     Degrees of freedom of the multivariate t distribution used for AIMH proposals. Defaults to 10.
%
% OUTPUTS
%   o chains            [array]     The samples (in form of an array of chains)
%   o lprobs            [array]     The log-probabilities
%   o prop_mean         [array]     The last mean of the proposal function.
%   o prop_cov          [array]     The last covariance of the proposal function.

[nchain, ndim] = size(init);

% get some default values
dft = df_proposal_dist;

if isnan(gammah)
    g0 = 2.38 / sqrt(2 * ndim);
else
    g0 = gammah;
end

% initialize
prop_cov = eye(ndim);
prop_mean = zeros(ndim,1);
accepted = ones(nchain,1);
cumlweight = -inf;

% calculate intial values
x = init;
lprob = log_prob(x');

% preallocate
chains = zeros(niter, nchain, ndim);
lprobs = zeros(niter, nchain);

if progress
    pbar = waitbar(0, '');
end

for i = 1:niter

    % get differential evolution proposal
    % draw the indices of the complementary chains
    i1 = (0:nchain-1) + randsample(nchain-1,nchain, true);
    i2 = (0:nchain-1) + randsample(nchain-2,nchain, true);
    i2(i2 > i1) += 1;

    % add small noise and calculate proposal
    f = sigma*normrnd(0,1, nchain, 1);
    q = x + g0 * (x(mod(i1, nchain) + 1,:) - x(mod(i2, nchain) + 1,:)) + f;
    factors = zeros(nchain,1);

    % log weight of current ensemble
    lweight = logsumexp(lprob) + log(sum(accepted)) - log(nchain);

    % calculate stats for current ensemble
    ncov = cov(x);
    nmean = mean(x);

    % update AIMH proposal distribution
    newcumlweight = logsumexp([cumlweight lweight]);
    prop_cov = exp(cumlweight - newcumlweight) * prop_cov + exp(lweight - newcumlweight) * ncov;
    prop_mean = exp(cumlweight - newcumlweight) * prop_mean + exp(lweight - newcumlweight) * nmean;
    cumlweight = newcumlweight;

    % get AIMH proposal
    xchnge = unifrnd(0,1,nchain,1) <= aimh_prob;

    % draw alternative candidates and calculate their proposal density
    xcand = mvtrnd(prop_cov*(dft - 2)/dft, dft, sum(xchnge));
    lprop_old = log(mvtpdf(x(xchnge,:), prop_cov*(dft - 2)/dft, dft));
    lprop_new = log(mvtpdf(xcand, prop_cov*(dft - 2)/dft, dft));

    % update proposals and factors
    q(xchnge,:) = xcand;
    factors(xchnge) = lprop_old - lprop_new;

    % Metropolis-Hasings 
    newlprob = log_prob(q');
    lnpdiff = factors + newlprob - lprob;
    accepted = lnpdiff > log(unifrnd(0,1,nchain,1));
    naccepted = sum(accepted);

    % update chains
    x(accepted,:) = q(accepted,:);
    lprob(accepted) = newlprob(accepted);

    % store
    chains(i,:,:) = x;
    lprobs(i,:) = lprob;

    if progress
        waitbar(i/niter, pbar, sprintf("[ll/MAF: %.3f(%1.0e)/%.0d%%]", max(lprob), std(lprob), 100*naccepted/nchain));
    end
end
if progress
    close(pbar)
end
