function [chains, lprobs, prop_mean, prop_cov] = dime_mcmc(log_prob, init, niter, optaimh_prob, optsigma, optgamma, optdf_proposal_dist, optrho)
% function [chains, lprobs, prop_mean, prop_cov] = dime_mcmc(log_prob, init, niter, optaimh_prob, optsigma, optgamma, optdf_proposal_dist)
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
%   o rho               [float]     The decay parameter for mean and covariance of the AIMH proposals. Defaults to 0.999.
%
% OUTPUTS
%   o chains            [array]     The samples (in form of an array of chains)
%   o lprobs            [array]     The log-probabilities
%   o prop_mean         [array]     The last mean of the proposal function.
%   o prop_cov          [array]     The last covariance of the proposal function.

[nchain, ndim] = size(init);
isplit = fix(nchain/2);

% get some default values
if nargin > 3
    aimh_prob = optaimh_prob;
else
    aimh_prob = 0.1;
end

if nargin > 4
    sigma = optsigma;
else
    sigma = 1e-5;
end

if nargin > 5
    g0 = optgamma;
else
    g0 = 2.38 / sqrt(2 * ndim);
end

if nargin > 6
    dft = optdf_proposal_dist;
else
    dft = 10;
end

if nargin > 7
    rho = optrho;
else
    rho = 0.999;
end

% initialize
prop_cov = eye(ndim);
prop_mean = zeros(ndim,1);
fixPSD = eye(size(prop_cov,1))*1e-15;
naccepted = 1;
cumlweight = -inf;

% calculate intial values
x = init;
lprob = log_prob(x');

% preallocate
chains = zeros(niter, nchain, ndim);
lprobs = zeros(niter, nchain);

pbar = waitbar(0, '');

for i = 1:niter

    % log weight of current ensemble
    lweight = logsumexp(lprob) + log(naccepted) - log(nchain);

    % calculate stats for current ensemble
    ncov = cov(x);
    nmean = mean(x);

    % update AIMH proposal distribution
    newcumlweight = logsumexp([cumlweight lweight]);
    prop_cov = exp(cumlweight - newcumlweight) * prop_cov + exp(lweight - newcumlweight) * ncov;
    prop_mean = exp(cumlweight - newcumlweight) * prop_mean + exp(lweight - newcumlweight) * nmean;
    cumlweight = newcumlweight + log(rho);
    naccepted = 0;

    for complementary_ensemble = [false true]

        % define current ensemble
        if complementary_ensemble
            idcur = 1:isplit+1;
            idref = isplit+1:nchain;
        else
            idcur = isplit+1:nchain;
            idref = 1:isplit+1;
        end
        cursize = length(idcur);
        refsize = nchain - cursize + 1.;

        % get differential evolution proposal
        % draw the indices of the complementary chains
        i1 = (0:cursize-1) + randsample(cursize-1,cursize, true)';
        i2 = (0:cursize-1) + randsample(cursize-2,cursize, true)';
        i2(i2 > i1) = i2(i2 > i1) + 1;

        % add small noise and calculate proposal
        f = sigma*normrnd(0,1, cursize, 1);
        q = x(idcur,:) + g0 * (x(idref(mod(i1, refsize) + 1),:) - x(idref(mod(i2, refsize) + 1),:)) + f;
        factors = zeros(cursize,1);

        % get AIMH proposal
        xchnge = unifrnd(0,1,cursize,1) <= aimh_prob;

        % draw alternative candidates and calculate their proposal density
        xcand = mvtrnd(prop_cov*(dft - 2)/dft + fixPSD, dft, sum(xchnge));
        lprop_old = log(mvtpdf(x(idcur(xchnge),:), prop_cov*(dft - 2)/dft + fixPSD, dft));
        lprop_new = log(mvtpdf(xcand, prop_cov*(dft - 2)/dft + fixPSD, dft));

        % update proposals and factors
        q(xchnge,:) = xcand;
        factors(xchnge) = lprop_old - lprop_new;

        % Metropolis-Hasings 
        newlprob = log_prob(q');
        lnpdiff = factors + newlprob - lprob(idcur);
        accepted = lnpdiff > log(unifrnd(0,1,cursize,1));
        naccepted = naccepted + sum(accepted);

        % update chains
        x(idcur(accepted),:) = q(accepted,:);
        lprob(idcur(accepted)) = newlprob(accepted);
    end

    % store
    chains(i,:,:) = x;
    lprobs(i,:) = lprob;

    waitbar(i/niter, pbar, sprintf("[ll/MAF: %.3f(%1.0e)/%02.f%%]", max(lprob), std(lprob), 100*naccepted/nchain));
end
close(pbar)
