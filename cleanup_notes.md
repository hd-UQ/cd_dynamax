- Add psd_covariance checks
    - print warnings if non-psd (with potential solutions)
        - add debugging option
        
- Check for nan loglikelihood values
    - How to raise errors in jax
   
- Dynamax dependencies
    - classes
    - optimizations
    - utils (check psd, lin solves, contrainers, etc.)

- Requirements / environment
    - Versions for: Diffrax, Blackjax, Numpyro [make sure everything else doesn't break]
    
- Moving beyond tensorflow.probability towards unified sampling dists
    - Explore numpyro.distributions
        - can we simply replace them in codebase?
        - in backend? e.g., for sampling paths and computing logprobs?
        - in frondend? e.g., to define priors
