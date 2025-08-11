from typing import NamedTuple, Tuple, Optional, Union
from jaxtyping import Array, Float, PyTree
import jax.numpy as jnp

# cd-dynamax Prior
from ssm_temissions import Prior
# cd-dynamax parameter classes
from continuous_discrete_linear_gaussian_ssm.cdlgssm_utils import ParamsCDLGSSM
from continuous_discrete_nonlinear_gaussian_ssm.cdnlgssm_utils import ParamsCDNLGSSM

# Import actual distributions to use by priors
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from tensorflow_probability.substrates.jax.distributions import Gamma

# An example prior definition for CDLGSSM: prior only for dynamics weights
class CDLGSSM_WeightPrior(Prior):
    ''' 
        A prior for the CDLGSSM weights only
    '''
    def __init__(
            self,
            params: ParamsCDLGSSM, **kwargs
        ):
        # Make sure parameters are of the right type
        assert isinstance(params, ParamsCDLGSSM)
        
        # We define the prior only for the dynamic weights
        # Keep track of its shape
        self.dynamic_weights_shape = params.dynamics.weights.shape

        # We define independent priors for each weight
        # TODO: use kwargs to define the prior
        self.dynamic_weights_prior = MVN(
            loc=jnp.zeros(params.dynamics.weights.size),
            covariance_matrix=jnp.eye(params.dynamics.weights.size)
        )

    # Sampling
    def sample(self, key, M):
        # Sample the state dynamic weigths
        dynamic_weights=self.dynamic_weights_prior.sample(
            (M,), seed=key,
        ).reshape(
            (M,)+self.dynamic_weights_shape # Reshape to the original shape
        )
        
        # Return them as a dictionary
        # which means they have to be reformatted as a PyTree outside
        return {'dynamics_weights': dynamic_weights}
    
    # Compute log_probability of the prior
    def log_prob(self, x):
        # Unbounded uniform assumed for all
        # We need to flatten the weights, keeping track of whether we have multiple samples M
        if x.dynamics.weights.ndim > len(self.dynamic_weights_shape):
            # Flatten the weights, keeping the first dimension for M
            dynamic_weights_samples = x.dynamics.weights.reshape(
                (x.dynamics.weights.shape[0], -1)
            )
        else:
            # Flatten the weights, with no need to keep first dimension for M
            dynamic_weights_samples=x.dynamics.weights.ravel()
        
        # Compute logpdf of the state dynamic weigths
        log_prob=self.dynamic_weights_prior.log_prob(
            dynamic_weights_samples
        )

        return log_prob
    
# An example prior definition for CDLGSSM: prior only for dynamics weights
class CDLGSSM_DynamicsPrior(Prior):
    ''' 
        A prior for the CDLGSSM dynamic parameters
    '''
    def __init__(
            self,
            params: ParamsCDLGSSM, **kwargs
        ):
        # Make sure parameters are of the right type
        assert isinstance(params, ParamsCDLGSSM)
        
        # Figure out state dimensionality from params
        self.state_dim = params.dynamics.weights.shape[0]

        # We define the prior for the dynamic weights
        # Keep track of its shape
        self.dynamic_weights_shape = params.dynamics.weights.shape

        # We define independent priors for each weight
        # TODO: use kwargs to define the prior
        self.dynamic_weights_prior = MVN(
            loc=jnp.zeros(params.dynamics.weights.size),
            covariance_matrix=jnp.eye(params.dynamics.weights.size)
        )

        # Draw from a Gamma distribution for the diffusion coefficient
        # Specifically, for the diagonal of the coefficient matrix L_t
        self.dynamics_diffusion_coefficient_prior = Gamma(
            concentration=0.5, # alpha
            rate=1. # beta
        )
              

    # Sampling
    def sample(self, key, M):
        # Sample the state dynamic weigths
        dynamic_weights=self.dynamic_weights_prior.sample(
            (M,), seed=key,
        ).reshape(
            (M,)+self.dynamic_weights_shape # Reshape to the original shape
        )

        # Sample the diffusion coefficient: as diagonal of the coefficient matrix L_t
        diffusion_coefficient=self.dynamics_diffusion_coefficient_prior.sample(
            (M, self.state_dim),
            seed=key,
        )[...,None] * jnp.eye(self.state_dim)[None,...] # Expand to the diagonal of the full L_t matrix

        # Return them as a dictionary
        # which means they have to be reformatted as a PyTree outside
        return {
            'dynamics_weights': dynamic_weights,
            'dynamics_diffusion_coefficient': diffusion_coefficient
        }
    
    # Compute log_probability of the prior
    def log_prob(self, x):
        # Unbounded uniform assumed for all

        # We need to flatten the weights, keeping track of whether we have multiple samples M
        if x.dynamics.weights.ndim > len(self.dynamic_weights_shape):
            # Flatten the weights, keeping the first dimension for M
            dynamic_weights_samples = x.dynamics.weights.reshape(
                (x.dynamics.weights.shape[0], -1)
            )
        else:
            # Flatten the weights, keeping the first dimension for M
            dynamic_weights_samples=x.dynamics.weights.reshape(
                (1, -1)
            )
        
        # Compute logpdf of the state dynamic weigths
        log_prob_weights=self.dynamic_weights_prior.log_prob(
            dynamic_weights_samples
        )

        # We need to get the diagonal of the diffusion coefficient matrix
        # TODO: this might complain for M>1
        diffusion_coefficient_diagonal = jnp.diagonal(
            x.dynamics.diffusion_coefficient,
            axis1=-2, axis2=-1
        )

        # Compute logpdf of the state dynamic weigths
        log_prob_diffusion_coefficient=self.dynamics_diffusion_coefficient_prior.log_prob(
            diffusion_coefficient_diagonal
        ).sum(
            axis=-1 # Sum over state dimension
        )

        # Return the sum
        return log_prob_weights + log_prob_diffusion_coefficient
    
from utils.physics_based_models import LearnableLorenz63_Drift
# An example prior definition for CDNLGSSM dynamics drift: prior over Lorenz63 drift parameters
class CDNLGSSM_DynamicDrift_L63ParamsPrior(Prior):
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