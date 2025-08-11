- Data generation:
    - We should generate data using a *hi-fidelity* solver
    - Ensure that data are *read* by default, and only generated if not present

- Numerics Section 1 (Marginalization vs joint inference)
    - Just do NUTS
    - Find a "small enough" time window example where NUTS "works" (for joint + marginal)
    - Compare to longer time window where NUTS fails for joint but works for marginal
    - Try to do it with an *informative* prior (over params and x0) to emphasize value of marginalization

- Numerics Section 2 (Benefits of Continuous-Discrete)
    - EKF:
        - Zeroth order: EKF in discrete time with state-cov = jnp.sqrt(dt) * L_t @ Qc_t @ L_t.T
        - First order: EKF in continuous-time with first-order state approximation
    - EnKF:
        - Zeroth order: EnKF in discrete-time...integrate ODE, then add noise to particles w/ state-cov = jnp.sqrt(dt) * L_t @ Qc_t @ L_t.T
        - First order: Propagate each particle according to SDE solve with drift and diffusion.
            - ** FIND A REFERENCE ** [https://arxiv.org/pdf/2212.02139]
    - UKF [for appendix]:
        - Zeroth order: UKF in discrete time with state-cov = jnp.sqrt(dt) * L_t @ Qc_t @ L_t.T
        - First order: UKF in continuous-time (Sarkka Thesis Algo 3.24)

    - Notes: 
        - Zeroth order ought to fail for longer time steps and/or larger diffusion coefficient (needed L=10ish)
        - EKF First order will fail until we deal with non-PSD forecasted covariance
        - UKF First order needs to be checked carefully (getting NaNs, probably non-PSD covs)
        - To illustrate these differences, we will sweep over many time-steps and a few diffusion coefficients
            - L = 1e-2, 1e-1, 1e0, 1e1, 1e2
            - dt = 1e-3, 1e-2, 1e-1, 1e0
            - keys (10): regenerate data for each key, then use the `10{key}` to seed the random filtering alg.
            - For each L, make a plot over dt of cumulative filtering relative RMSE.

- Numerics Section 3 (Benefits of auto-diff)
    - Scipy vs SGD for MAP optimization:
        - L63 params: rho, sigma, beta
        - NN-drift for L63
    - UQ: SVI (and it matches NUTS)
        - L63 params: rho, sigma, beta
        - NN-drift for L63

- High dimensional inference problem:
    - Lorenz 96 model
    - Consider learning the RHS with a dictionary