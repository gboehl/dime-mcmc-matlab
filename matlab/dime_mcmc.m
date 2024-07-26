function [chains, lprobs, prop_mean, prop_cov] = dime_mcmc(log_prob, init, niter, opts)
% function [chains, lprobs, prop_mean, prop_cov] = dime_mcmc(log_prob, init, niter, opts)
% DIME MCMC sampling
%
% INPUTS
%   o log_prob          [function]  (vectorized) function returning the log-density
%   o init              [array]     The initial ensemble
%   o niter             [int]       The number of iterations to be run.
%   o opts              [struct]    Struct with tuning options:
%       - opts.aimh_prob         [float]     Probability to draw an adaptive independence Metropolis Hastings (AIMH) proposal. By default this is set to 0.1.
%       - opts.sigma             [float]     The standard deviation of the Gaussian used to stretch the proposal vector.
%       - opts.gamma             [float]     The mean stretch factor for the proposal vector. By default, it is 2.38 / sqrt(2 ndim) as recommended by ter Braak (2006): http://www.stat.columbia.edu/~gelman/stuff_for_blog/cajo.pdf
%       - opts.df_proposal_dist  [float]     Degrees of freedom of the multivariate t distribution used for AIMH proposals. Defaults to 10.
%       - opts.rho               [float]     The decay parameter for mean and covariance of the AIMH proposals. Defaults to 0.999.
%       - opts.show_pbar         [bool]      Whether to show a waitbar. Defaults to true.
%
% OUTPUTS
%   o chains            [array]     The samples (in form of an array of chains)
%   o lprobs            [array]     The log-probabilities
%   o prop_mean         [array]     The last mean of the proposal function.
%   o prop_cov          [array]     The last covariance of the proposal function.

[nchain, ndim] = size(init);
isplit = fix(nchain/2) + 1;

% get some default values
if nargin == 3
    opts.empty = true;
end

if isfield(opts,'aimh_prob')
    aimh_prob = opts.aimh_prob;
else
    aimh_prob = 0.1;
end

if isfield(opts,'sigma')
    sigma = opts.sigma;
else
    sigma = 1e-5;
end

if isfield(opts,'gamma')
    g0 = opts.gamma;
else
    g0 = 2.38 / sqrt(2 * ndim);
end

if isfield(opts,'df_proposal_dist')
    dft = opts.df_proposal_dist;
else
    dft = 10;
end

if isfield(opts,'rho')
    rho = opts.rho;
else
    rho = 0.999;
end

if isfield(opts,'show_pbar')
    show_pbar = opts.show_pbar;
else
    show_pbar = true;
end

% initialize
prop_cov = eye(ndim);
prop_mean = zeros(1,ndim);
fixPSD = eye(size(prop_cov,1))*1e-15;
naccepted = 1;
cumlweight = -inf;

% calculate intial values
x = init;
lprob = log_prob(x');
if any(lprob < -1e6) 
    error("Density of at least one member of the initial ensemble is below -1e6")
end

% preallocate
chains = zeros(niter, nchain, ndim);
lprobs = zeros(niter, nchain);

if show_pbar == 1
    pbar = waitbar(0, '');
end

for i = 1:niter

    for complementary_ensemble = [false true]

        % define current ensemble
        if complementary_ensemble
            idcur = 1:isplit;
            idref = (isplit+1):nchain;
        else
            idcur = (isplit+1):nchain;
            idref = 1:isplit;
        end
        cursize = length(idcur);
        refsize = nchain - cursize;

        % get differential evolution proposal
        % draw the indices of the complementary chains
        i1 = (0:cursize-1) + randi([1 cursize-1], 1, cursize);
        i2 = (0:cursize-1) + randi([1 cursize-2], 1, cursize);
        i2(i2 >= i1) = i2(i2 >= i1) + 1;

        % add small noise and calculate proposal
        f = sigma*randn(cursize, 1);
        q = x(idcur,:) + g0 * (x(idref(mod(i1, refsize) + 1),:) - x(idref(mod(i2, refsize) + 1),:)) + f;
        factors = zeros(cursize,1);

        % get AIMH proposal
        xchnge = rand(cursize,1) <= aimh_prob;

        % draw alternative candidates and calculate their proposal density
        xcand = mvt_rnd(prop_mean, prop_cov*(dft - 2)/dft + fixPSD, dft, sum(xchnge));
        lprop_old = mvt_logpdf(x(idcur(xchnge),:), prop_mean, prop_cov*(dft - 2)/dft + fixPSD, dft);
        lprop_new = mvt_logpdf(xcand, prop_mean, prop_cov*(dft - 2)/dft + fixPSD, dft);

        % update proposals and factors
        q(xchnge,:) = xcand;
        factors(xchnge) = lprop_old - lprop_new;

        % Metropolis-Hasings 
        newlprob = log_prob(q');
        lnpdiff = factors + newlprob - lprob(idcur);
        accepted = lnpdiff > log(rand(cursize,1));
        naccepted = naccepted + sum(accepted);

        % update chains
        x(idcur(accepted),:) = q(accepted,:);
        lprob(idcur(accepted)) = newlprob(accepted);
    end

    % store
    chains(i,:,:) = x;
    lprobs(i,:) = lprob;

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

    if show_pbar == 1
        waitbar(i/niter, pbar, sprintf("%d [ll(std)/MAF: %.3f(%1.0e)/%02.0f%%]", i, max(lprob), std(lprob), 100*naccepted/nchain));
    elseif show_pbar > 1
        fprintf("%d/%d [ll(std)/MAF: %.3f(%1.0e)] %02.0f%%\n", i, niter, max(lprob), std(lprob), 100*naccepted/nchain);
    end
    naccepted = 0;
end
if show_pbar == 1
    close(pbar)
end
