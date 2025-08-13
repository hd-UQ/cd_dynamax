import jax.numpy as jnp

# cd-dynamax Prior
from ssm_temissions import Prior
# cd-dynamax parameter classes
from continuous_discrete_nonlinear_gaussian_ssm.cdnlgssm_utils import ParamsCDNLGSSM

# Import actual distributions to use by priors
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN

from utils.physics_based_models import LearnableLorenz63_Drift

# An example prior definition for CDNLGSSM dynamics drift: prior over Lorenz63 drift parameters
class CDNLGSSM_Prior(Prior):
    ''' 
        A prior for the CDNLGSSM dynamic drift, based on Lorenz 63
        These are kept within params.dynamics.drift.params
    '''
    def __init__(
            self,
            **kwargs
        ):        
        # Figure out parameter dimensionality
        self.param_dim = 3

        # Define the prior only for the dynamic drift params
        # TODO: use kwargs to define the prior

        # We assume independent priors for each parameter, centered around good values
        # for the Lorenz63 model
        self.dynamic_drift_param_prior = MVN(
            loc=jnp.array([10., 28., 8/3]),
            covariance_matrix=jnp.eye(self.param_dim)
        )      

    # Sampling
    def sample(self, key, M):
        # Sample the state dynamic weigths
        samples=self.dynamic_drift_param_prior.sample(
            (M,), seed=key,
        )

        # Return samples within dictionary
        # which means the rest of params have to be reformatted as a PyTree outside
        return {
            # Note that for CD-NLGSSM we have learnable functions, from which we have sampled just the parameters
            # hence, we need to create such function here
            'dynamics_drift': LearnableLorenz63_Drift(
                sigma=samples[...,0],
                rho=samples[...,1],
                beta=samples[...,2]
            ),
        }
    
    # Compute log_probability of the prior
    def log_prob(self, x):
        # Unbounded uniform assumed for all
            
        # Compute logpdf of the state dynamic drift params
        log_prob=self.dynamic_drift_param_prior.log_prob(
            jnp.stack([
                x['dynamics_drift'].sigma,
                x['dynamics_drift'].rho,
                x['dynamics_drift'].beta
            ], axis=-1)
        )

        return log_prob