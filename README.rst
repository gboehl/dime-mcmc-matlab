dime-mcmc-matlab
================

**Differential-Independence Mixture Ensemble ("DIME") MCMC sampling for matlab**

This is a standalone matlab implementation of the DIME sampler (previously ADEMC sampler) proposed in `Ensemble MCMC Sampling for DSGE Models <https://gregorboehl.com/live/ademc_boehl.pdf>`_. *(Gregor Boehl, 2022, CRC 224 discussion paper series)*.

The sampler has a series of advantages over conventional samplers:

#. DIME MCMC is a (very fast) **global multi-start optimizer** and, at the same time, a **MCMC sampler** that converges to the posterior distribution. This makes any posterior mode density maximization prior to MCMC sampling superfluous.
#. The DIME sampler is pretty robust for odd shaped, **multimodal distributions**.
#. DIME MCMC is **parallelizable**: many chains can run in parallel, and the necessary number of draws decreases almost one-to-one with the number of chains.
#. DIME proposals are generated from an **endogenous and adaptive proposal distribution**, thereby providing close-to-optimal proposal distributions without the need for manual fine-tuning.

Installation
------------

Copy the `matlab <https://github.com/gboehl/dime-mcmc-matlab/tree/main/matlab>`_ folder from this repo somwhere on your PC and add it to you matlab path:

.. code-block:: matlab

    addpath('where/ever/to/dime_mcmc_matlab/matlab')

You can get a zip file containing the complete repo `here <https://github.com/gboehl/dime-mcmc-matlab/archive/refs/heads/main.zip>`_.

Note that you need the statistics toolbox for matlab. Unfortunately this toolbox does not seem to be fully compatible with ``pkg load statistics`` in Octave (sorry, not my fault).

There exist complementary implementations of DIME MCMC in `Python <https://github.com/gboehl/emcwrap>`_ and `Julia <https://github.com/gboehl/DIMESampler.jl>`_ (where you don't need expensive toolboxes).

Usage
-----

The core functionality is included in the function ``dime_mcmc``:

.. code-block:: matlab

    % define your density function
    log_prob = ...

    % define the initial ensemble
    initchain = ...

    % define the number of iterations
    niter = ...

    % off you go sampling
    [chains, lprobs, prop_mean, prop_cov] = dime_mcmc(log_prob, initchain, niter)
    ...


Tutorial
--------

Be sure the files from above are in your matlab path. Then, lets define a challenging example distribution **with three separate modes**:

.. code-block:: matlab

    % make it reproducible
    rng('default'); rng(1);

    % define distribution
    m = 2;
    cov_scale = 0.05;
    weight = [0.33 0.1];
    ndim = 35;

    % define distribution
    log_prob = create_dime_test_func(ndim, weight, m, cov_scale);

``log_prob`` will now return the log-PDF of a 35-dimensional Gaussian mixture.

**Important:** the function returning the log-density must be vectorized, i.e. able to evaluate inputs with size ``(:,ndim)``. If you want to make use of parallelization (which is one of the central advantages of ensemble MCMC), you may want to ensure that this function evaluates its vectorized input in parallel.

Next, define the initial ensemble. In a Bayesian setup, a good initial ensemble would be a sample from the prior distribution. Here, we will go for a sample from a rather flat Gaussian distribution.

.. code-block:: matlab

    initvar = 2;
    nchain = ndim*5; % a sane default
    initcov = eye(ndim)*initvar;
    initmean = zeros(ndim, 1);
    initchain = mvnrnd(initmean, initcov, nchain);

Setting the number of parallel chains to ``5*ndim`` is a sane default. For highly irregular distributions with several modes you should use more chains. Very simple distributions can go with less. 

Now let the sampler run for 5000 iterations.

.. code-block:: matlab

    niter = 5000;
    [chains, lprobs, prop_mean, prop_cov] = dime_mcmc(log_prob, initchain, niter);

The setting of ``aimh_prob`` is the actual default value. For less complex distributions (e.g. distributions closer to Gaussian) a higher value can be chosen, which accelerates burn-in. The information in the progress bar has the structure ``[ll/MAF: <maximum log-prob>(<standard deviation of log-prob>)/<mean acceptance fraction>]...``.

Let's plot the marginal distribution along the first dimension (remember that this actually is a 35-dimensional distribution).

.. code-block:: matlab

    % get sample and analytical marginal pdf
    x = linspace(-4,4,1000);
    mpdf = dime_test_func_marginal_pdf(x, cov_scale, m, weight);
    sample = reshape(chains(end-fix(niter/3):end,:,1), [], 1);

    % calculate a histogram with densities
    bins = linspace(-3,3,50);
    counts = histc(sample, bins);
    density = counts / (sum(counts) * (bins(2)-bins(1)));
    scale = sqrt(prop_cov(1,1)*10/8);

    % plot
    figure;
    hold on
    bar(bins + (bins(2)-bins(1))/2, density)
    plot(x, mpdf)
    plot(x, normpdf(x, 0, sqrt(initvar)))
    plot(x, tpdf((x - prop_mean(1))/scale, 10)/scale)
    xlim([-4 4])
    legend({'Sample', 'Target','Initialization','Final Proposal'},'Location','northwest')
    hold off

.. image:: https://github.com/gboehl/emcwrap/blob/main/docs/dist.png?raw=true
  :width: 800
  :alt: Sample and target distribution

The plot is actually taken from the Python implementation because it looks soo nice.
To ensure proper mixing, let us also have a look at the MCMC traces, again focussing on the first dimension:

.. code-block:: matlab

    figure;
    lines = plot(chains(:,:,1),'-b');
    for i = 1:length(lines)
        lines(i).Color(4) = 0.05;
    end
        
.. image:: https://github.com/gboehl/emcwrap/blob/main/docs/traces.png?raw=true
  :width: 800
  :alt: MCMC traces
  
Note how chains are also switching between the three modes because of the global proposal kernel.

While DIME is a MCMC sampler, it can straightforwardly be used as a global optimization routine. To this end, specify some broad starting region (in a non-Bayesian setup there is no prior) and let the sampler run for an extended number of iterations. Finally, assess whether the maximum value per ensemble did not change much in the last few hundred iterations. In a normal Bayesian setup, plotting the associated log-likelihood over time also helps to assess convergence to the posterior distribution.

.. code-block:: matlab

    figure;
    lines = plot(lprobs, '-b');
    for i = 1:length(lines)
        lines(i).Color(4) = 0.05;
    end

.. image:: https://github.com/gboehl/emcwrap/blob/main/docs/lprobs.png?raw=true
  :width: 800
  :alt: Log-likelihoods

References
----------

If you are using this software in your research, please cite

.. code-block::

    @techreport{boehl2022mcmc,
    title         = {Ensemble MCMC Sampling for DSGE Models},
    author        = {Boehl, Gregor},
    year          = 2022,
    institution   = {CRC224 discussion paper series}
    }
