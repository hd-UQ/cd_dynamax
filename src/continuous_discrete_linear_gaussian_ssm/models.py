import pdb
from fastprogress.fastprogress import progress_bar
from functools import partial
from jax import jit
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import tree_map
from jaxtyping import Array, Float, PyTree
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from typing import Any, Optional, Tuple, Union
from typing_extensions import Protocol

import jax.debug as jdb

# From dynamax
from dynamax.parameters import ParameterProperties, ParameterSet
from dynamax.types import PRNGKey, Scalar
from dynamax.utils.bijectors import RealToPSDBijector
from dynamax.utils.distributions import MatrixNormalInverseWishart as MNIW
from dynamax.utils.distributions import NormalInverseWishart as NIW
from dynamax.utils.distributions import mniw_posterior_update, niw_posterior_update
from dynamax.utils.utils import pytree_stack, psd_solve

# Our codebase
from ssm_temissions import SSM
# To avoid unnecessary redefinitions of code,
# We import parameters and posteriors that can be reused from LGSSM first
from dynamax.linear_gaussian_ssm.inference import ParamsLGSSMInitial, ParamsLGSSMEmissions, PosteriorGSSMFiltered, PosteriorGSSMSmoothed
# Param definition
from continuous_discrete_linear_gaussian_ssm.inference import ParamsCDLGSSMDynamics, ParamsCDLGSSM
from continuous_discrete_linear_gaussian_ssm.inference import KFHyperParams
from continuous_discrete_linear_gaussian_ssm.inference import cdlgssm_filter, cdlgssm_smoother
# Unclear why we define this here, but not in models
from continuous_discrete_linear_gaussian_ssm.inference import cdlgssm_joint_sample, cdlgssm_path_sample, cdlgssm_posterior_sample
from continuous_discrete_linear_gaussian_ssm.inference import compute_pushforward

class SuffStatsCDLGSSM(Protocol):
    """A :class:`NamedTuple` with sufficient statistics for CDLGSSM parameter estimation."""
    pass
    
class ContDiscreteLinearGaussianSSM(SSM):
    r"""
    Continuous Discrete Linear Gaussian State Space Model.

    The model is defined in equation (3.134)
    
    # TODO: replace this below
    $$p(z_0) = \mathcal{N}(z_0 \mid m, S)$$
    $$p(z_t \mid z_{t-1}, u_t) = \mathcal{N}(z_t \mid A_t z_{t-1} + B_t u_t + b_t, Q_t)$$
    $$p(y_t \mid z_t) = \mathcal{N}(y_t \mid H_t z_t + D_t u_t + d_t, R_t)$$

    where

    * $z_t$ is a latent state of size `state_dim`,
    * $y_t$ is an emission of size `emission_dim`
    * $u_t$ is an input of size `input_dim` (defaults to 0)
    * $A_t$ = are the dynamics (transition) of the state:
                A_t is the solution to the ODE in eq (3.135)
    * $B$ = optional input-to-state weight matrix
    * $b$ = optional input-to-state bias vector
    * $L$ = diffusion coefficient of the dynamics (system) 
    * $Q$ = diffucion covariance matrix of dynamics (system) ---brownian motion
    * $H$ = emission (observation) matrix
    * $D$ = optional input-to-emission weight matrix
    * $d$ = optional input-to-emission bias vector
    * $R$ = covariance function for emission (observation) noise
    * $m$ = mean of initial state
    * $S$ = covariance matrix of initial state

    The parameters of the model are stored in a :class:`ParamsCDLGSSM`.
    You can create the parameters manually, or by calling :meth:`initialize`.

    :param state_dim: Dimensionality of latent state.
    :param emission_dim: Dimensionality of observation vector.
    :param input_dim: Dimensionality of input vector. Defaults to 0.
    :param has_dynamics_bias: Whether model contains an offset term $b$. Defaults to True.
    :param has_emissions_bias:  Whether model contains an offset term $d$. Defaults to True.

    """
    def __init__(
        self,
        state_dim: int,
        emission_dim: int,
        input_dim: int=0,
        has_dynamics_bias: bool=True,
        has_emissions_bias: bool=True,
        diffeqsolve_settings: dict={},
    ):
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        self.has_dynamics_bias = has_dynamics_bias
        self.has_emissions_bias = has_emissions_bias
        self._diffeqsolve_settings = diffeqsolve_settings

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    @property
    def inputs_shape(self):
        return (self.input_dim,) if self.input_dim > 0 else None

    @property
    def diffeqsolve_settings(self):
        return self._diffeqsolve_settings

    # This is a revised initialize, consistent across cd-dynamax, based on dicts
    def initialize(
        self,
        key: PRNGKey =jr.PRNGKey(0),
        initial_mean: dict = None,
        initial_cov: dict = None,
        dynamics_weights: dict = None,
        dynamics_bias: dict = None,
        dynamics_input_weights: dict = None,
        dynamics_diffusion_coefficient: dict = None,
        dynamics_diffusion_cov: dict = None,
        dynamics_approx_order: Optional[float] = 2.,
        emission_weights: dict = None,
        emission_bias: dict = None,
        emission_input_weights: dict = None,
        emission_cov: dict = None,
    ) -> Tuple[ParamsCDLGSSM, ParamsCDLGSSM]:
        r"""Initialize model parameters that are set to None, and their corresponding properties.

        Args:
            key: Random number key. Defaults to jr.PRNGKey(0).
            initial_mean: parameter $m$. Defaults to None.
            initial_cov: parameter $S$. Defaults to None.
            dynamics_weights: parameter $F$. Defaults to None.
            dynamics_bias: parameter $b$. Defaults to None.
            dynamics_input_weights: parameter $B$. Defaults to None.
            dynamics_diffusion_coefficient: parameter $L$. Defaults to None.
            dynamics_diffusion_cov: parameter $Q$. Defaults to None.
            emission_weights: parameter $H$. Defaults to None.
            emission_bias: parameter $d$. Defaults to None.
            emission_input_weights: parameter $D$. Defaults to None.
            emission_cov: parameter $R$. Defaults to None.

        Returns:
            Tuple[ParamsCDLGSSM, ParamsCDLGSSM]: parameters and their properties.
        """

        ### Arbitrary default values, for demo purposes
        # Default is to have NOTHING learnable.
        ## Initial
        _initial_mean = {
            "params": jnp.zeros(self.state_dim),
            "props": ParameterProperties(trainable=False)
        }

        _initial_cov = {
            "params": jnp.eye(self.state_dim),
            "props": ParameterProperties(
                        trainable=False,
                        constrainer=RealToPSDBijector()
                    )
        }

        ## Dynamics
        _dynamics_weights = {
            "params": -0.1 * jnp.eye(self.state_dim),
            "props": ParameterProperties(trainable=False)
        }
        _dynamics_input_weights = {
            "params": jnp.zeros((self.state_dim, self.input_dim)),
            "props": ParameterProperties(trainable=False)
        }
        _dynamics_bias = {
            "params": jnp.zeros((self.state_dim,)) if self.has_dynamics_bias else None,
            "props": ParameterProperties(trainable=False)
        }
        _dynamics_diffusion_coefficient = {
            "params": 0.1 * jnp.eye(self.state_dim),
            "props": ParameterProperties(trainable=False)
        }
        _dynamics_diffusion_cov = {
            "params": 0.1 * jnp.eye(self.state_dim),
            "props": ParameterProperties(trainable=False, constrainer=RealToPSDBijector())
        }
        
        ## Emission
        _emission_weights = {
            "params": jr.normal(key, (self.emission_dim, self.state_dim)),
            "props": ParameterProperties(trainable=False)
        }
        _emission_input_weights = {
            "params": jnp.zeros((self.emission_dim, self.input_dim)),
            "props": ParameterProperties(trainable=False)
        }
        _emission_bias = {
            "params": jnp.zeros((self.emission_dim,)) if self.has_emissions_bias else None,
            "props": ParameterProperties(trainable=False)
        }
        _emission_cov = {
            "params": 0.1 * jnp.eye(self.emission_dim),
            "props": ParameterProperties(trainable=False, constrainer=RealToPSDBijector())
        }

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0

        # replace defaults as needed
        initial_mean = default(initial_mean, _initial_mean)
        initial_cov = default(initial_cov, _initial_cov)

        dynamics_weights = default(dynamics_weights, _dynamics_weights)
        dynamics_input_weights = default(dynamics_input_weights, _dynamics_input_weights)
        dynamics_bias = default(dynamics_bias, _dynamics_bias)
        dynamics_diffusion_coefficient = default(dynamics_diffusion_coefficient, _dynamics_diffusion_coefficient)
        dynamics_diffusion_cov = default(dynamics_diffusion_cov, _dynamics_diffusion_cov)
        
        emission_weights = default(emission_weights, _emission_weights)
        emission_input_weights = default(emission_input_weights, _emission_input_weights)
        emission_bias = default(emission_bias, _emission_bias)
        emission_cov = default(emission_cov, _emission_cov)
        
        ## Create nested dictionary of params
        params_dict = {"params": {}, "props": {}}
        for key in params_dict.keys():
            params_dict[key] = ParamsCDLGSSM(
                initial=ParamsLGSSMInitial(
                    mean=initial_mean[key],
                    cov=initial_cov[key]
                ),
                dynamics=ParamsCDLGSSMDynamics(
                    weights=dynamics_weights[key],
                    input_weights=dynamics_input_weights[key],
                    bias=dynamics_bias[key],
                    diffusion_coefficient=dynamics_diffusion_coefficient[key],
                    diffusion_cov=dynamics_diffusion_cov[key],
                ),
                emissions=ParamsLGSSMEmissions(
                    weights=emission_weights[key],
                    input_weights=emission_input_weights[key],
                    bias=emission_bias[key],
                    cov=emission_cov[key],
                )
            )

        return params_dict["params"], params_dict["props"]

    def initial_distribution(
        self,
        params: ParamsCDLGSSM,
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> tfd.Distribution:
        return MVN(params.initial.mean, params.initial.cov)

    # NB: The discrete pushforward of the continuous state dynamics
    #       and discrete inputs (controls) 
    def transition_distribution(
        self,
        params: ParamsCDLGSSM,
        state: Float[Array, "state_dim"],
        t0: Optional[Float]=None,
        t1: Optional[Float]=None,
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> tfd.Distribution:
    
        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        # Compute pushforward map:
        # A maps the state from t0 to t1
        # Q is the covariance at t1
        A, Q = compute_pushforward(params, t0, t1, diffeqsolve_settings=self.diffeqsolve_settings)
        # Pushforward the state from t0 to t1, then add controls at t1 
        mean = A @ state + params.dynamics.input_weights @ inputs
        if self.has_dynamics_bias:
            mean += params.dynamics.bias
        
        return MVN(mean, Q)
        
    def emission_distribution(
        self,
        params: ParamsCDLGSSM,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> tfd.Distribution:
        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        mean = params.emissions.weights @ state + params.emissions.input_weights @ inputs
        if self.has_emissions_bias:
            mean += params.emissions.bias
        return MVN(mean, params.emissions.cov)

    
    def sample_dist(
        self,
        params: ParamsCDLGSSM,
        key: PRNGKey,
        num_timesteps: int,
        t_emissions: Optional[Float[Array, "ntime 1"]]=None,
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> Tuple[Float[Array, "num_timesteps state_dim"], Float[Array, "num_timesteps emission_dim"]]:
        print('Sampling from continuous-discrete linear Gaussian SSM distributions')
        return cdlgssm_joint_sample(
            params,
            key,
            num_timesteps,
            t_emissions,
            inputs,
            diffeqsolve_settings=self.diffeqsolve_settings
        )
    
    def sample_path(
        self,
        params: ParamsCDLGSSM,
        key: PRNGKey,
        num_timesteps: int,
        t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
        inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
    ) -> Tuple[Float[Array, "num_timesteps state_dim"],
                Float[Array, "num_timesteps emission_dim"]]:
        r"""Sample from a forward path to produce state and emission trajectories.

        Args:
            params: model parameters
            t_emissions: continuous-time specific time instants of observations: if not None, it is an array 
            inputs: optional array of inputs.

        Returns:
            latent states and emissions

        """
        print('Sampling from continuous-discrete linear Gaussian SSM path')
        return cdlgssm_path_sample(
            params=params,
            key=key,
            num_timesteps=num_timesteps,
            t_emissions=t_emissions,
            inputs=inputs,
            diffeqsolve_settings=self.diffeqsolve_settings
        )
    
    def marginal_log_prob(
        self,
        params: ParamsCDLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        t_emissions: Optional[Float[Array, "ntime 1"]]=None,
        filter_hyperparams: Optional[KFHyperParams]=None,
        inputs: Optional[Float[Array, "ntime input_dim"]] = None
    ) -> Scalar:
        filtered_posterior = cdlgssm_filter(params, emissions, t_emissions, filter_hyperparams, inputs)
        return filtered_posterior.marginal_loglik

    def filter(
        self,
        params: ParamsCDLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        t_emissions: Optional[Float[Array, "ntime 1"]]=None,
        filter_hyperparams: Optional[KFHyperParams]=None,
        inputs: Optional[Float[Array, "ntime input_dim"]] = None
    ) -> PosteriorGSSMFiltered:
        return cdlgssm_filter(params, emissions, t_emissions, filter_hyperparams, inputs)

    def smoother(
        self,
        params: ParamsCDLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        t_emissions: Optional[Float[Array, "ntime 1"]]=None,
        filter_hyperparams: Optional[KFHyperParams]=None,
        inputs: Optional[Float[Array, "ntime input_dim"]] = None
    ) -> PosteriorGSSMSmoothed:
        return cdlgssm_smoother(params, emissions, t_emissions, filter_hyperparams, inputs)

    def posterior_sample(
        self,
        key: PRNGKey,
        params: ParamsCDLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        t_emissions: Optional[Float[Array, "ntime 1"]]=None,
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> Float[Array, "ntime state_dim"]:
        return cdlgssm_posterior_sample(key, params, emissions, t_emissions, inputs)

    def posterior_predictive(
        self,
        params: ParamsCDLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        t_emissions: Optional[Float[Array, "ntime 1"]]=None,
        filter_hyperparams: Optional[KFHyperParams]=None,
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> Tuple[Float[Array, "ntime emission_dim"], Float[Array, "ntime emission_dim"]]:
        r"""Compute marginal posterior predictive smoothing distribution for each observation.

        Args:
            params: model parameters.
            emissions: sequence of observations.
            inputs: optional sequence of inputs.

        Returns:
            :posterior predictive means $\mathbb{E}[y_{t,d} \mid y_{1:T}]$ and standard deviations $\mathrm{std}[y_{t,d} \mid y_{1:T}]$

        """
        posterior = cdlgssm_smoother(params, emissions, t_emissions, filter_hyperparams, inputs)
        H = params.emissions.weights
        b = params.emissions.bias
        R = params.emissions.cov
        emission_dim = R.shape[0]
        smoothed_emissions = posterior.smoothed_means @ H.T + b
        smoothed_emissions_cov = H @ posterior.smoothed_covariances @ H.T + R
        smoothed_emissions_std = jnp.sqrt(
            jnp.array([smoothed_emissions_cov[:, i, i] for i in range(emission_dim)]))
        return smoothed_emissions, smoothed_emissions_std

    # Expectation-maximization (EM) code
    def e_step(
        self,
        params: ParamsCDLGSSM,
        emissions: Union[Float[Array, "num_timesteps emission_dim"],
                         Float[Array, "num_batches num_timesteps emission_dim"]],
        t_emissions: Optional[Union[Float[Array, "num_timesteps 1"],
                        Float[Array, "num_batches num_timesteps 1"]]]=None,
        filter_hyperparams: Optional[KFHyperParams]=None,
        inputs: Optional[Union[Float[Array, "num_timesteps input_dim"],
                               Float[Array, "num_batches num_timesteps input_dim"]]]=None,
    ) -> Tuple[SuffStatsCDLGSSM, Scalar]:
        
        num_timesteps = emissions.shape[0]
        if inputs is None:
            inputs = jnp.zeros((num_timesteps, 0))

        # Run the smoother to get posterior expectations
        posterior = cdlgssm_smoother(params, emissions, t_emissions, filter_hyperparams, inputs)

        # shorthand
        Ex = posterior.smoothed_means
        Exp = posterior.smoothed_means[:-1]
        Exn = posterior.smoothed_means[1:]
        Vx = posterior.smoothed_covariances
        Vxp = posterior.smoothed_covariances[:-1]
        Vxn = posterior.smoothed_covariances[1:]
        Expxn = posterior.smoothed_cross_covariances

        # Append bias to the inputs
        inputs = jnp.concatenate((inputs, jnp.ones((num_timesteps, 1))), axis=1)
        up = inputs[:-1]
        u = inputs
        y = emissions

        # expected sufficient statistics for the initial tfd.Distribution
        Ex0 = posterior.smoothed_means[0]
        Ex0x0T = posterior.smoothed_covariances[0] + jnp.outer(Ex0, Ex0)
        init_stats = (Ex0, Ex0x0T, 1)

        # expected sufficient statistics for the dynamics tfd.Distribution
        # let zp[t] = [x[t], u[t]] for t = 0...T-2
        # let xn[t] = x[t+1]          for t = 0...T-2
        sum_zpzpT = jnp.block([[Exp.T @ Exp, Exp.T @ up], [up.T @ Exp, up.T @ up]])
        sum_zpzpT = sum_zpzpT.at[:self.state_dim, :self.state_dim].add(Vxp.sum(0))
        sum_zpxnT = jnp.block([[Expxn.sum(0)], [up.T @ Exn]])
        sum_xnxnT = Vxn.sum(0) + Exn.T @ Exn
        dynamics_stats = (sum_zpzpT, sum_zpxnT, sum_xnxnT, num_timesteps - 1)
        if not self.has_dynamics_bias:
            dynamics_stats = (sum_zpzpT[:-1, :-1], sum_zpxnT[:-1, :], sum_xnxnT,
                                num_timesteps - 1)

        # more expected sufficient statistics for the emissions
        # let z[t] = [x[t], u[t]] for t = 0...T-1
        sum_zzT = jnp.block([[Ex.T @ Ex, Ex.T @ u], [u.T @ Ex, u.T @ u]])
        sum_zzT = sum_zzT.at[:self.state_dim, :self.state_dim].add(Vx.sum(0))
        sum_zyT = jnp.block([[Ex.T @ y], [u.T @ y]])
        sum_yyT = emissions.T @ emissions
        emission_stats = (sum_zzT, sum_zyT, sum_yyT, num_timesteps)
        if not self.has_emissions_bias:
            emission_stats = (sum_zzT[:-1, :-1], sum_zyT[:-1, :], sum_yyT, num_timesteps)

        return (init_stats, dynamics_stats, emission_stats), posterior.marginal_loglik


    def initialize_m_step_state(
            self,
            params: ParamsCDLGSSM,
            props: ParamsCDLGSSM
    ) -> Any:
        return None

    def m_step(
        self,
        params: ParamsCDLGSSM,
        props: ParamsCDLGSSM,
        batch_stats: SuffStatsCDLGSSM,
        m_step_state: Any
    ) -> Tuple[ParamsCDLGSSM, Any]:
                
        def fit_linear_regression(ExxT, ExyT, EyyT, N):
            # Solve a linear regression given sufficient statistics
            W = psd_solve(ExxT, ExyT).T
            Sigma = (EyyT - W @ ExyT - ExyT.T @ W.T + W @ ExxT @ W.T) / N
            return W, Sigma

        # Sum the statistics across all batches
        stats = tree_map(partial(jnp.sum, axis=0), batch_stats)
        init_stats, dynamics_stats, emission_stats = stats

        # Perform MLE estimation jointly
        sum_x0, sum_x0x0T, N = init_stats
        S = (sum_x0x0T - jnp.outer(sum_x0, sum_x0)) / N
        m = sum_x0 / N

        # TODO: What's the m-step MLE for diffusion_cov and diffusion_coefficient?
        raise ValueError('m_step not implemented yet: what is the MLE for diffusion_cov and diffusion_coefficient?')
        
        FB, Q = fit_linear_regression(*dynamics_stats)
        F = FB[:, :self.state_dim]
        B, b = (FB[:, self.state_dim:-1], FB[:, -1]) if self.has_dynamics_bias \
            else (FB[:, self.state_dim:], None)

        HD, R = fit_linear_regression(*emission_stats)
        H = HD[:, :self.state_dim]
        D, d = (HD[:, self.state_dim:-1], HD[:, -1]) if self.has_emissions_bias \
            else (HD[:, self.state_dim:], None)
        
        params = ParamsCDLGSSM(
            initial=ParamsLGSSMInitial(mean=m, cov=S),
            # TODO: this will crash, as we should provide diffusion_cov and diffusion_coefficient
            dynamics=ParamsCDLGSSMDynamics(weights=F, bias=b, input_weights=B, cov=Q),
            emissions=ParamsLGSSMEmissions(weights=H, bias=d, input_weights=D, cov=R)
        )
        return params, m_step_state

