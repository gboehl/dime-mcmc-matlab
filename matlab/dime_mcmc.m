% dime mcmc

function [chains, lprobs, prop_mean, prop_cov] = dime_mcmc(log_prob, init, niter, sigma=1e-5, gammah=nan, aimh_prob=0.1, df_proposal_dist=10, progress=true)

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
        % TODO: this should use logsumexp to avoid numerical errors
        lweight = log(sum(exp(lprob))) + log(sum(accepted)) - log(nchain);

        % calculate stats for current ensemble
        ncov = cov(x);
        nmean = mean(x);

        % update AIMH proposal distribution
        newcumlweight = log(exp(cumlweight) + exp(lweight));
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
end
