# Example tutorial notebooks on the use of cd-dynamax 

Each tutorial is tagged with the model family it exercises — `ContDiscreteLinearGaussianSSM` (**CD-LGSSM**), `ContDiscreteNonlinearGaussianSSM` (**CD-NLGSSM**), or `ContDiscreteNonlinearSSM` (**CD-NLSSM**, generic/non-Gaussian emissions), so one can jump to the model type of interest.

- **[CD-LGSSM]** [CD-LGSSM tutorial](./cdlgssm_sgd_fit_to_data_tutorial.ipynb) on how to sample from a continuous-discrete linear SDE model, filter observed data and fit model parameters to data using SGD. It also illustrates how to handle different data-streams, each with their own emission times.

- **[CD-LGSSM]** [GP regression as an LTI CD-LGSSM tutorial](./gp_regression_lti_cdlgssm_tutorial.ipynb) on casting a Gaussian-Process regression problem as an equivalent linear-time-invariant continuous-discrete state space model, solved via Kalman filtering.

- **[CD-NLGSSM]** [Filtering tutorial](./lorenz63_filtering_tutorial.ipynb) on how to filter observed data of a continuous-discrete SDE model, based on different filtering algorithms.

- **[CD-NLGSSM]** [Filtering-based likelihood tutorial](./lorenz63_filter_based_likelihood_tutorial.ipynb) on computing filtering-based likelihoods for continuous-discrete SDEs.

- **[CD-NLGSSM]** [SGD-based model fitting tutorial](./lorenz63_sgd_fit_to_data_tutorial.ipynb) on SGD-based fitting a continuous-discrete SDE model to data.

- **[CD-NLGSSM]** [MCMC-based model fitting tutorial](./lorenz63_mcmc_fit_to_data_tutorial.ipynb) on MCMC-based fitting a continuous-discrete SDE model to data.

- **[CD-NLGSSM]** [SGD-based Neural Network drift fitting tutorial](./lorenz63_nndrift_sgd_fit_to_data_tutorial.ipynb) on how to learn a continuous-discrete SDE drift function using Neural Networks, to fit model to observed data.

- **[CD-NLSSM]** [Differentiable Particle Filter for Poisson data tutorial](./poisson_data_dpf.ipynb) on how to use a differentiable particle filter to fit a continuous-discrete SDE model (the Ornstein–Uhlenbeck process) with Poisson emissions.

- **[CD-NLSSM]** [Comparison of a Differentiable Particle Filter and Ensemble Kalman Filter](./tracking_dpf_enkf.ipynb) a tracking example with non-Gaussian observations, where observations consist of a bearing estimate (that lives in the circle, $S^1$) and a power estimate (that lives in $\mathbb{R}$).

- [A cd-dynamax config walkthrough](./cddynamax_experiment_config_tutorial.ipynb) on loading, overriding, and writing [demos/python/configs](../configs/) config files directly from Python, to use in notebooks or [demos/python/scripts](../scripts/).

