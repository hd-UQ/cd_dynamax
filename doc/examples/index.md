# Examples

We provide a set of tutorial and example notebooks that demonstrate fitting, filtering, and likelihood evaluation with continuous-discrete state-space models in cd_dynamax.

- **[CD-LGSSM SGD fit to data](notebooks/cdlgssm_sgd_fit_to_data_tutorial/)** — Fit a continuous-discrete linear Gaussian SSM to synthetic data using SGD on the marginal log-likelihood (computed via the Kalman filter).

- **[Lorenz63 SGD fit to data](notebooks/lorenz63_sgd_fit_to_data_tutorial/)** — Fit a nonlinear Lorenz 63 model to observed data by maximizing log-likelihood with SGD, using the EnKF to approximate the likelihood.

- **[Lorenz63 neural drift SGD fit](notebooks/lorenz63_nndrift_sgd_fit_to_data_tutorial/)** — Fit a Lorenz 63 model whose drift is parameterized by a neural network, using SGD and the Ensemble Kalman Filter for (approximate) likelihood evaluation.

- **[Lorenz63 MCMC fit to data](notebooks/lorenz63_mcmc_fit_to_data_tutorial/)** — Estimate Lorenz 63 parameters via MCMC (NUTS), with the Extended Kalman Filter used to approximate the log-likelihood.

- **[Lorenz63 filtering](notebooks/lorenz63_filtering_tutorial/)** — Run filtering (and optionally forecasting) on a Lorenz 63 model with partial, noisy observations.

- **[Lorenz63 filter-based likelihood](notebooks/lorenz63_filter_based_likelihood_tutorial/)** — Use the EnKF to compute the (approximate) marginal log-likelihood of observations under a Lorenz 63 model, and sweep over parameters to visualize the likelihood surface.
