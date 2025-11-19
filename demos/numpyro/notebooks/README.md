# Example tutorial notebooks on the use of cd-dynamax 

- [Filtering tutorial](./lorenz63_filtering_tutorial.ipynb) on how to filter observed data of a continuous-discrete SDE model, based on different filtering algorithms.

- [Filtering-based likelihood tutorial](./lorenz63_filter_based_likelihood_tutorial.ipynb) on computing filtering-based likelihoods for continuous-discrete SDEs.

- [Filtering-based likelihood tutorial (using new `.build_params` API)](./lorenz63_filter_based_likelihood_tutorial_newAPI.ipynb) on computing filtering-based likelihoods for continuous-discrete SDEs.

- [SGD-based model fitting tutorial](./lorenz63_sgd_fit_to_data_tutorial.ipynb) on SGD-based fitting a continuous-discrete SDE model to data.

- [MCMC-based model fitting tutorial](./lorenz63_mcmc_fit_to_data_tutorial.ipynb) on MCMC-based fitting a continuous-discrete SDE model to data.

- [SGD-based Neural Network drift fitting tutorial](./lorenz63_sgd_fit_to_data_tutorial.ipynb) on how to learn a continuous-discrete SDE drift function using Neural Networks, to fit model to observed data.

# Tutorial notebooks comparing different latent state inference methods
- [Inferring mechanistic parameters](./lorenz63_nndrift_compare_latents.ipynb) comparing different latent state inference methods (filtering, Euler-Maruyama, Diffrax) when fitting a continuous-discrete SDE model with *mechanistic* drift.
    - We find that the Euler-Maruyama based method performs best in this scenario.
    - All methods are able to accurately recover the mechanistic parameters of the Lorenz63 system.

- [Inferring neural network drift](./lorenz63_mechanistic_compare_latents.ipynb) comparing different latent state inference methods (filtering, Euler-Maruyama, Diffrax) when fitting a continuous-discrete SDE model with *Neural Network* drift.
    - We find that the filtering based method performs best in this scenario.
    - All other methods (Euler-Maruyama, Diffrax) fail to learn an accurate neural network drift function in this setting.