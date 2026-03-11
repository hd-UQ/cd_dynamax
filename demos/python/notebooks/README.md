# Example tutorial notebooks on the use of cd-dynamax 

- [CD-LGGSM tutorial](./cdlgssm_sgg_fit_to_data.ipynb) on how to sample from a continuous-discrete linear SDE model, filter observed data and fit model parameters to data using SGD. It also illustrates how to handle different data-streams, each with their own emission times.

- [Filtering tutorial](./lorenz63_filtering_tutorial.ipynb) on how to filter observed data of a continuous-discrete SDE model, based on different filtering algorithms.

- [Filtering-based likelihood tutorial](./lorenz63_filter_based_likelihood_tutorial.ipynb) on computing filtering-based likelihoods for continuous-discrete SDEs.

- [SGD-based model fitting tutorial](./lorenz63_sgd_fit_to_data_tutorial.ipynb) on SGD-based fitting a continuous-discrete SDE model to data.

- [MCMC-based model fitting tutorial](./lorenz63_mcmc_fit_to_data_tutorial.ipynb) on MCMC-based fitting a continuous-discrete SDE model to data.

- [SGD-based Neural Network drift fitting tutorial](./lorenz63_sgd_fit_to_data_tutorial.ipynb) on how to learn a continuous-discrete SDE drift function using Neural Networks, to fit model to observed data.

- [Differentiable Particle Filter for Poisson data tutorial](./poisson_data_dpf.ipynb) on how to use a differentiable particle filter to fit a continuous-discrete SDE model (the Ornstein–Uhlenbeck process) with Poisson emissions.

- [Comparison of a Differentiable Particle Filter and Ensemble Kalman Filter](./tracking_dpf_enkf.ipynb) a tracking example with non-Gaussian observations, where observations consist of a bearing estimate (that lives in the circle, $S^1$) and a power estimate (that lives in $\mathbb{R}$).

