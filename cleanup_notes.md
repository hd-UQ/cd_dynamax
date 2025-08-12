- Add psd_covariance checks
    - print warnings if non-psd (with potential solutions)
        - add debugging option
        
- Check for nan loglikelihood values
    - How to raise errors in jax
   
- Dynamax dependencies
    - Types and parameters
        from dynamax.types import PRNGKey, Scalar
        from dynamax.parameters import ParameterSet, PropertySet

    - utils
        from dynamax.parameters import to_unconstrained, from_unconstrained, log_det_jac_constrain
        from dynamax.utils.utils import ensure_array_has_batch_dim, fallback_hessian
        from dynamax.utils.utils import psd_solve, symmetrize
        from dynamax.utils.bijectors import RealToPSDBijector
        from dynamax.utils.utils import pytree_len

    - classes
        from dynamax.linear_gaussian_ssm.inference import ParamsLGSSMInitial, ParamsLGSSMEmissions
        from dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered, PosteriorGSSMSmoothed

    - optimizations
        from dynamax.utils.optimize import run_sgd as dynamax_run_sgd
        from dynamax.utils.optimize import sample_minibatches

- Requirements / environment
    - Versions for: Diffrax, Blackjax, Numpyro [make sure everything else doesn't break]
    
- Moving beyond tensorflow.probability towards unified sampling dists
    - Explore numpyro.distributions
        - can we simply replace them in codebase?
        - in backend? e.g., for sampling paths and computing logprobs?
        - in frondend? e.g., to define priors
