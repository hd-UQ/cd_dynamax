# Example tutorial notebooks on the use of cd-dynamax 

- [CD-LGGSM tutorial](./cdlgssm_sgg_fit_to_data.ipynb) on how to sample from a continuous-discrete linear SDE model, filter observed data and fit model parameters to data using SGD. It also illustrates how to handle different data-streams, each with their own emission times.

- [Filtering tutorial](./lorenz63_filtering_tutorial.ipynb) on how to filter observed data of a continuous-discrete SDE model, based on different filtering algorithms.

- [Filtering-based likelihood tutorial](./lorenz63_filter_based_likelihood_tutorial.ipynb) on computing filtering-based likelihoods for continuous-discrete SDEs.

- [Filtering-based likelihood tutorial (using new `.build_params` API)](./lorenz63_filter_based_likelihood_tutorial_newAPI.ipynb) on computing filtering-based likelihoods for continuous-discrete SDEs.

- [DPF with Poisson emissions tutorial (using new `.build_params` API)](./dpf_examples.ipynb) on using DPFs with non-Gaussian emissions in two examples: one with Poisson emissions, and the other a tracking example with circular emissions.

- [SGD-based model fitting tutorial](./lorenz63_sgd_fit_to_data_tutorial.ipynb) on SGD-based fitting a continuous-discrete SDE model to data.

- [MCMC-based model fitting tutorial](./lorenz63_mcmc_fit_to_data_tutorial.ipynb) on MCMC-based fitting a continuous-discrete SDE model to data.

- [SGD-based Neural Network drift fitting tutorial](./lorenz63_sgd_fit_to_data_tutorial.ipynb) on how to learn a continuous-discrete SDE drift function using Neural Networks, to fit model to observed data.