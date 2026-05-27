# cd-dynamax's abstract SSM class and related types
from abc import ABC
from abc import abstractmethod

# JAX imports
import jax.numpy as jnp
import jax.random as jr
from jax import jit, lax, vmap, grad, value_and_grad
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree

# Type annotations
from jaxtyping import Float, Array
from typing import Optional, Union, Tuple, Any
from typing_extensions import Protocol
from functools import partial

# cd-dynamax relies on tensorflow-probability for distributions, as per dynamax
from tensorflow_probability.substrates.jax import distributions as tfd

# Imports from dynamax
from cd_dynamax.dynamax.types import PRNGKey, Scalar
from cd_dynamax.dynamax.parameters import (
    to_unconstrained,
    from_unconstrained,
    log_det_jac_constrain,
)
from cd_dynamax.dynamax.parameters import ParameterSet, PropertySet
from cd_dynamax.dynamax.utils.utils import ensure_array_has_batch_dim, fallback_hessian

# Imports from the cd-dynamax codebase
from .utils.debug_utils import lax_scan, resolve_verbosity
from .utils.optimize_utils import run_sgd

# Utils for optimization
import optax
import blackjax
from fastprogress.fastprogress import progress_bar

# Used in fit_scipy_* functions
from jaxopt import ScipyMinimize
from scipy.optimize import minimize

DEBUG = False  # By default, debugging is off, e.g., no extra checks in lax_scan


# cd-dynamax's abstract prior class
class Prior(ABC):
    r"""cd-dynamax priors: these prior should have

    * `sample` returns a tensor of samples from this prior
    * `log_prob` returns the log probability of the prior, for a given parameter set
    """

    # Method definitions
    @abstractmethod
    def sample(self, key, M):
        r"""
        A sampling function to be defined by specific prior classes

        Args:
            key: Random number key. Defaults to jr.PRNGKey(0).
            M: number of samples to draw

        Returns:
            PyTree with samples
        """

    @abstractmethod
    def log_prob(self, x):
        r"""
        A function to compute the log_probability of specific prior classes

        Args:
            x: Pytree with samples to evaluate log_prob at

        Returns:
            PyTree with log_prob values
        """


class Posterior(Protocol):
    r"""
    A `NamedTuple` with parameters stored as `jax.Array` in the leaf nodes."""

    pass


class SuffStatsSSM(Protocol):
    r"""A `NamedTuple` with sufficient statistics stored as `jax.Array` in the leaf nodes."""

    pass


# cd-dynamax's abstract SSM class
class SSM(ABC):
    r"""A base cd-dynamax class for continuous-discrete state space models.
    Such models consist of parameters, which we may learn,
    as well as hyperparameters, which specify static properties of the model.
    This base class allows parameters to be indicated in a standardized way
    so that they can easily be converted to/from unconstrained form for optimization.

    **Abstract Methods**

    Models that inherit from `SSM` must implement a few key functions and properties:

    * `initial_distribution` returns the distribution over the initial state given parameters
    * `transition_distribution` returns the conditional distribution over the next state given the current state and parameters
    * `emission_distribution` returns the conditional distribution over the emission given the current state and parameters
    * `log_prior` (optional) returns the log prior probability of the parameters
    * `emission_shape` returns a tuple specification of the emission shape
    * `inputs_shape` returns a tuple specification of the input shape, or `None` if there are no inputs.

    The shape properties are required for properly handling batches of data.

    **Sampling and Computing Log Probabilities**

    Once these have been implemented, subclasses will inherit the ability
    to sample from these models and to compute log joint probabilities from the base class functions:

    * `sample` draws samples of the states and emissions for given parameters
    * `log_prob` computes the log joint probability of the states and emissions for given parameters

    **Inference**

    Many subclasses of SSMs expose basic functions for performing state inference.

    * `marginal_log_prob` computes the marginal log probability of the emissions, summing over latent states
    * `filter` computes the filtered posteriors
    * `smoother` computes the smoothed posteriors

    **Learning**

    Likewise, many SSMs will support learning with stochastic gradient descent (SGD) or Markov Chain Monte Carlo (MCMC).

    The generic SSM class allows to fit the model with the preferred learning algorithm

    For SGD, any subclass that implements `marginal_log_prob` inherits the base class fitting function

    * `fit_sgd` run SGD to minimize the *negative* marginal log probability.

    For black-box optimization, any subclass that implements `marginal_log_prob` inherits the base class fitting function

    * `fit_scipy` run SciPy-based optimization to minimize the *negative* marginal log probability.
    * `fit_scipy_jaxopt` run jaxopt SciPyMinimize optimization to minimize the *negative* marginal log probability.

    For MCMC, any subclass that implements `marginal_log_prob` inherits the base class fitting function
    * `fit_mcmc` run BlackJAX HMC to sample from the posterior over parameters given data.

    """

    # An initialize method, consistent across cd-dynamax, based on dicts
    @abstractmethod
    def initialize(self, *args, **kwargs):
        r"""Initialize the model parameters.

        Args:
            To be defined by the specific model class
        Returns:
            CD-SSM parameters and their properties.
        """

        raise NotImplementedError

    @abstractmethod
    def initial_distribution(
        self, params: ParameterSet, inputs: Optional[Float[Array, " input_dim"]]
    ) -> tfd.Distribution:
        r"""Return an initial distribution over latent states.

        Args:
            params: model parameters $\theta$
            inputs: optional  inputs  $u_t$

        Returns:
            distribution over initial latent state, $p(z_1 \mid u_t, \theta)$.

        """
        raise NotImplementedError

    @abstractmethod
    def transition_distribution(
        self,
        params: ParameterSet,
        state: Float[Array, " state_dim"],
        t0: Optional[Float],
        t1: Optional[Float],
        inputs: Optional[Float[Array, " input_dim"]],
    ) -> tfd.Distribution:
        r"""Return a distribution over next latent state given current state.

        Args:
            params: model parameters $\theta$
            state: current latent state $z_t$
            inputs: current inputs  $u_t$

        Returns:
            conditional distribution of next latent state $p(z_{t+1} \mid z_t, u_t, \theta)$.

        """
        raise NotImplementedError

    @abstractmethod
    def emission_distribution(
        self,
        params: ParameterSet,
        state: Float[Array, " state_dim"],
        inputs: Optional[Float[Array, " input_dim"]] = None,
    ) -> tfd.Distribution:
        r"""Return a distribution over emissions given current state.

        Args:
            params: model parameters $\theta$
            state: current latent state $z_t$
            inputs: current inputs  $u_t$

        Returns:
            conditional distribution of current emission $p(y_t \mid z_t, u_t, \theta)$

        """
        raise NotImplementedError

    @property
    @abstractmethod
    def emission_shape(self) -> Tuple[int]:
        r"""Return a pytree matching the pytree of tuples specifying the shape of a single time step's emissions."""
        raise NotImplementedError

    @property
    def inputs_shape(self) -> Optional[Tuple[int]]:
        r"""Return a pytree matching the pytree of tuples specifying the shape of a single time step's inputs."""
        return None

    @property
    def diffeqsolve_settings(self) -> dict:
        r"""Return a dictionary of settings for the differential equation solver."""
        return {}

    # All SSMs support sampling
    def sample(
        self,
        params: ParameterSet,
        key: PRNGKey,
        num_timesteps: int,
        t_emissions: Optional[Float[Array, "num_timesteps 1"]] = None,
        inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
        transition_type: Optional[str] = "distribution",
        verbosity: Optional[int] = None,
    ) -> Tuple[
        Float[Array, "num_timesteps state_dim"],
        Float[Array, "num_timesteps emission_dim"],
    ]:
        r"""Sample states $z_{1:T}$ and emissions $y_{1:T}$ given parameters $\theta$ and (optionally) inputs $u_{1:T}$.

        Args:
            params: model parameters $\theta$
            key: random number generator
            num_timesteps: number of timesteps $T$
            t_emissions: continuous-time specific time instants: if not None, it is an array
            inputs: inputs $u_{1:T}$
            transition_type: type of transition function, either "distribution" (default) or "path"
                "distribution" samples from the (default Gaussian) transition distribution (default)
                    - This is exact for Linear Gaussian SSMs
                "path" runs an SDE solver to sample the distribution. This is more "exact" (up to discretization error).
                    - Note: this is not supported for Linear Gaussian SSMs.

        Returns:
            latent states and emissions

        """
        verbosity = 1 if verbosity is None else verbosity
        if transition_type == "distribution":
            if verbosity >= 2:
                print(
                    "Sampling from CD distributions: this may be a poor approximation if you're simulating from a non-linear SDE. It is a highly appropriate choice for linear SDEs."
                )
            states, emissions = self.sample_dist(
                params, key, num_timesteps, t_emissions, inputs, verbosity=verbosity
            )
        elif transition_type == "path":
            if verbosity >= 2:
                print(
                    "Sampling from SDE solver path: this may be an unnecessarily poor approximation if you're simulating from a linear SDE. It is an appropriate choice for non-linear SDEs."
                )
            states, emissions = self.sample_path(
                params, key, num_timesteps, t_emissions, inputs, verbosity=verbosity
            )
        else:
            raise ValueError(f"Invalid transition_type: {transition_type}")

        return states, emissions

    # All SSMs support sampling a batch of sequences
    def sample_batch(
        self,
        params: ParameterSet,
        key: PRNGKey,
        num_sequences: int,
        num_timesteps: int,
        t_emissions: Optional[Float[Array, "num_timesteps 1"]] = None,
        inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
        transition_type: Optional[str] = "distribution",
        verbosity: Optional[int] = None,
    ) -> Tuple[
        Float[Array, "num_sequences num_timesteps state_dim"],
        Float[Array, "num_sequences num_timesteps emission_dim"],
    ]:
        r"""Sample a batch of sequences of states and emissions.

        Args:
            params: model parameters $\theta$
            key: random number generator
            num_sequences: number of sequences to sample
            num_timesteps: number of timesteps $T$
            t_emissions: continuous-time specific time instants: if not None, it is an array
            inputs: inputs $u_{1:T}$
            transition_type: type of transition function, either "distribution" (default) or "path"
                "distribution" samples from the (default Gaussian) transition distribution (default)
                    - This is exact for Linear Gaussian SSMs
                "path" runs an SDE solver to sample the distribution. This is more "exact" (up to discretization error).
                    - Note: this is not supported for Linear Gaussian SSMs.

        Returns:
            latent states and emissions

        """

        # Sample each sequence using self.sample and stack them
        def _sample_sequence(key):
            return self.sample(
                params,
                key,
                num_timesteps,
                t_emissions,
                inputs,
                transition_type,
                verbosity=verbosity,
            )

        keys = jr.split(key, num_sequences)
        # use vmap to sample multiple sequences in parallel
        states, emissions = vmap(_sample_sequence)(keys)
        return states, emissions

    # Compute log prior probability for given parameters
    def log_prior(self, params: ParameterSet) -> Scalar:
        r"""Return the log prior probability of model parameters.

        Args:
            params: model parameters $\theta$

        Returns:
            log_prior (Scalar): log prior probability.
        """
        # Default is no prior
        log_prior = 0.0

        # If a prior is specified, compute the log prior
        if self.prior is not None:
            log_prior = self.prior.log_prob(params)

        # Return the log prior
        return log_prior

    # Compute log joint probability of states and emissions
    def log_prob(
        self,
        params: ParameterSet,
        states: Float[Array, "num_timesteps state_dim"],
        emissions: Float[Array, "num_timesteps emission_dim"],
        t_emissions: Optional[Float[Array, "num_timesteps 1"]] = None,
        inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
        key: PRNGKey = jr.PRNGKey(0),
    ) -> Scalar:
        r"""Compute the log joint probability of the states and observations, $\log p(y_{1:T}, z_{1:T} \mid \theta)$.

        Args:
            params: model parameters $\theta$
            states: latent states $z_{1:T}$
            emissions: observed data $y_{1:T}$
            t_emissions: continuous-time specific time instants: if not None, it is an array
            inputs: current inputs  $u_t$
            key: random number generator key

        Returns:
            log joint probability (scalar) of states and emissions

        """

        # Extract initial states, emissions, and inputs
        initial_state = tree_map(lambda x: x[0], states)
        initial_emission = tree_map(lambda x: x[0], emissions)
        initial_input = tree_map(lambda x: x[0], inputs)
        # Compute log prob of initial time step
        lp = self.initial_distribution(params, initial_input).log_prob(initial_state)
        lp += self.emission_distribution(params, initial_state, initial_input).log_prob(
            initial_emission
        )

        # Define the scan step function
        def _step(carry, args):
            lp, prev_state = carry
            state, emission, t0, t1, inpt = args
            lp += self.transition_distribution(
                params, prev_state, t0, t1, inpt
            ).log_prob(state)
            lp += self.emission_distribution(params, state, inpt).log_prob(emission)
            return (lp, state), None

        # Figure out timestamps, as vectors to scan over
        # t_emissions is of shape num_timesteps \times 1
        # t0 and t1 are num_timesteps-1 \times 0
        if t_emissions is not None:
            num_timesteps = t_emissions.shape[0]
            t0 = tree_map(lambda x: x[0:-1, 0], t_emissions)
            t1 = tree_map(lambda x: x[1:, 0], t_emissions)
        else:
            num_timesteps = len(emissions)
            t0 = jnp.arange(num_timesteps - 1)
            t1 = jnp.arange(1, num_timesteps)

        # Scan over time steps
        next_states = tree_map(lambda x: x[1:], states)
        next_emissions = tree_map(lambda x: x[1:], emissions)
        next_inputs = tree_map(lambda x: x[1:], inputs)
        (lp, _), _ = lax.scan(
            _step,
            (lp, initial_state),
            (next_states, next_emissions, t0, t1, next_inputs),
        )

        # Return the log probability
        return lp

    # Some SSMs will implement these inference functions.
    def marginal_log_prob(
        self,
        params: ParameterSet,
        emissions: Float[Array, "ntime emission_dim"],
        t_emissions: Optional[Float[Array, "num_timesteps 1"]] = None,
        filter_hyperparams: Optional[Any] = None,
        inputs: Optional[Float[Array, "ntime input_dim"]] = None,
        key: PRNGKey = jr.PRNGKey(0),
        warn: bool = True,
        verbosity: Optional[int] = None,
    ) -> Scalar:
        r"""Compute log marginal likelihood of observations, $\log \sum_{z_{1:T}} p(y_{1:T}, z_{1:T} \mid \theta)$.

        Args:
            params: model parameters $\theta$
            emissions: emissions $y_{1:T}$
            t_emissions: continuous-time specific time instants: if not None, it is an array
            filter_hyperparams: hyperparameters of the filtering algorithm
            inputs: current inputs  $u_t$
            key: random number generator (for use in randomized methods approximating the marginal likelihood)
            warn: whether to print warnings from filters
        Returns:
            marginal log probability

        """
        warn, verbosity = resolve_verbosity(warn=warn, verbosity=verbosity)
        raise NotImplementedError

    # Compute the score function, i.e., the gradient of the log marginal likelihood
    def score(
        self,
        params: ParameterSet,
        props: PropertySet,
        emissions: Float[Array, "ntime emission_dim"],
        t_emissions: Optional[Float[Array, "num_timesteps 1"]] = None,
        filter_hyperparams: Optional[Any] = None,
        inputs: Optional[Float[Array, "ntime input_dim"]] = None,
        return_log_prob: bool = False,
        warn: bool = True,
        verbosity: Optional[int] = None,
        key: PRNGKey = jr.PRNGKey(0),
    ) -> Scalar:
        r"""Compute the score function, i.e., the gradient of the log marginal likelihood of observations:
            $\nabla \log \sum_{z_{1:T}} p(y_{1:T}, z_{1:T} \mid \theta)$.

        Args:
            params: model parameters $\theta$
            props: properties specifying which parameters should be learned
            emissions: observed data $y_{1:T}$
            t_emissions: continuous-time specific time instants: if not None, it is an array
            filter_hyperparams: hyperparameters of the filtering algorithm
            inputs: current inputs  $u_t$
            return_log_prob: whether or not to return the log probability in addition to its gradient
            warn: whether to print warnings from filters
            key: random number generator (for use in randomized methods approximating the marginal likelihood)

        Returns:
            gradient of marginal log probability

        Note: We need to exclude non-trainable parameters from the gradient computation
        because sometimes objects like integers or booleans are included in the parameter set.
        These are not differentiable and will cause the gradient to be None
        even with stop_gradient applied.

        """
        warn, verbosity = resolve_verbosity(warn=warn, verbosity=verbosity)
        # Extract only trainable parameters
        trainable_params = tree_map(
            lambda p, prop: p if prop.trainable else None, params, props
        )

        # Define a helper function to compute log_prob given only trainable parameters
        def _log_prob(trainable_params):
            # Recombine: fill in non-trainable values from original `params`
            full_params = tree_map(
                lambda full, trained, prop: trained if prop.trainable else full,
                params,
                trainable_params,
                props,
            )
            return self.marginal_log_prob(
                full_params,
                emissions,
                t_emissions,
                filter_hyperparams,
                inputs,
                key=key,
                warn=warn, verbosity=verbosity,
            )

        if return_log_prob:
            logp, grads = value_and_grad(_log_prob)(trainable_params)
            return logp, grads
        else:
            grads = grad(_log_prob)(trainable_params)
            return grads

    # Inference algorithms
    # All SSMs support filtering
    def filter(
        self,
        params: ParameterSet,
        emissions: Float[Array, "ntime emission_dim"],
        t_emissions: Optional[Float[Array, "num_timesteps 1"]] = None,
        filter_hyperparams: Optional[Union[Any]] = None,
        inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    ) -> Posterior:
        r"""Compute filtering distributions, $p(z_t \mid y_{1:t}, u_{1:t}, \theta)$ for $t=1,\ldots,T$.

        Args:
            params: model parameters $\theta$
            emissions: observed data $y_{1:T}$
            t_emissions: continuous-time specific time instants: if not None, it is an array
            filter_hyperparams: hyperparameters of the filtering algorithm
            inputs: current inputs  $u_t$

        Returns:
            filtering distributions

        """
        raise NotImplementedError

    # All SSMs support smoothing
    def smoother(
        self,
        params: ParameterSet,
        emissions: Float[Array, "ntime emission_dim"],
        t_emissions: Optional[Float[Array, "num_timesteps 1"]] = None,
        filter_hyperparams: Optional[Union[Any]] = None,
        inputs: Optional[Float[Array, "ntime input_dim"]] = None,
    ) -> Posterior:
        r"""Compute smoothing distributions, $p(z_t \mid y_{1:T}, u_{1:T}, \theta)$ for $t=1,\ldots,T$.

        Args:
            params: model parameters $\theta$
            emissions: observed data $y_{1:T}$
            t_emissions: continuous-time specific time instants: if not None, it is an array
            filter_hyperparams: hyperparameters of the filtering algorithm
            inputs: current inputs  $u_t$

        Returns:
            smoothing distributions

        """
        raise NotImplementedError

    # Learning algorithms
    # Fit model using SGD
    def fit_sgd(
        self,
        initial_params: ParameterSet,
        props: PropertySet,
        emissions: Union[
            Float[Array, "num_timesteps emission_dim"],
            Float[Array, "num_batches num_timesteps emission_dim"],
        ],
        t_emissions: Optional[
            Union[
                Float[Array, "num_timesteps 1"],
                Float[Array, "num_batches num_timesteps 1"],
            ]
        ] = None,
        filter_hyperparams: Optional[Any] = None,
        inputs: Optional[
            Union[
                Float[Array, "num_timesteps input_dim"],
                Float[Array, "num_batches num_timesteps input_dim"],
            ]
        ] = None,
        optimizer: optax.GradientTransformation = optax.adam(1e-3),
        batch_size: int = 1,
        num_epochs: int = 50,
        shuffle: bool = False,
        return_param_history: bool = False,
        return_grad_history: bool = False,
        warn: bool = True,
        verbosity: Optional[int] = None,
        key: PRNGKey = jr.PRNGKey(0),
    ) -> Tuple[ParameterSet, Float[Array, " niter"]]:
        r"""Compute parameter MLE/MAP estimate using Stochastic Gradient Descent (SGD).

        SGD aims to find parameters that maximize the marginal log probability,

        $$\theta^\star = \mathrm{argmax}_\theta \; \log p(y_{1:T}, \theta \mid u_{1:T})$$

        by minimizing the _negative_ of that quantity.

        *Note:* ``emissions`` *and* ``inputs`` *can either be single sequences or batches of sequences.*

        On each iteration, the algorithm grabs a *minibatch* of sequences and takes a gradient step.

        One pass through the entire set of sequences is called an *epoch*.

        Args:
            initial_params: model parameters $\theta$
            props: model properties specifying which parameters should be learned
            emissions: one or more sequences of observed data
            t_emissions: continuous-time specific time instants: if not None, it is an array
            filter_hyperparams: if needed, hyperparameters of the filtering algorithm
            inputs: one or more sequences of corresponding inputs
            optimizer: an `optax` optimizer for minimization
            batch_size: number of sequences per minibatch
            num_epochs: number of epochs of SGD to run
            shuffle: whether or not to shuffle the sequences on each
            return_param_history: whether to return the history of parameters
            return_grad_history: whether to return the history of gradients
            key: a random number generator for selecting minibatches

        Returns:
            tuple of new parameters and losses (negative scaled marginal log probs) over the course of SGD iterations.
                if interested in the history of parameters and gradients, these are returned as well.

        """
        warn, verbosity = resolve_verbosity(warn=warn, verbosity=verbosity)
        # Make sure the emissions and inputs have batch dimensions
        batch_emissions = ensure_array_has_batch_dim(emissions, self.emission_shape)
        batch_t_emissions = ensure_array_has_batch_dim(t_emissions, (1,))
        batch_inputs = ensure_array_has_batch_dim(inputs, self.inputs_shape)

        # Convert initial parameters to unconstrained space
        initial_unc_params = to_unconstrained(initial_params, props)
        # build initial_unc_params_trainable from initial_unc_params and props
        # by setting untrainable parameters to None
        initial_unc_params_trainable = tree_map(
            lambda param, prop: param if prop.trainable else None,
            initial_unc_params,
            props,
        )

        # The log likelihood that SGD computes on a minibatch
        def _loss_fn(unc_params_trainable, minibatch):
            # Combine the trainable and non-trainable parameters, then convert them to constrained space
            unc_params = tree_map(
                lambda initial, trained, prop: trained if prop.trainable else initial,
                initial_unc_params,
                unc_params_trainable,
                props,
            )
            params = from_unconstrained(unc_params, props)

            # Extract minibatch data
            minibatch_emissions, minibatch_t_emissions, minibatch_inputs = minibatch
            scale = len(batch_emissions) / len(minibatch_emissions)
            # Compute marginal log likelihoods for the minibatch
            minibatch_lls = vmap(
                partial(
                    self.marginal_log_prob,
                    params,
                    filter_hyperparams=filter_hyperparams,
                    warn=warn, verbosity=verbosity,
                )  # partial with fixed params arg and filter_hyperparams kwarg
            )(
                # arguments to vmap over
                emissions=minibatch_emissions,
                t_emissions=minibatch_t_emissions,
                inputs=minibatch_inputs,
            )
            # Compute the (scaled) negative log posterior, including prior
            lp = self.log_prior(params) + minibatch_lls.sum() * scale
            # Return negative log posterior scaled by total number of data points
            return -lp / batch_emissions.size

        # Run SGD over dataset: emissions, their timestamps, and inputs
        dataset = (batch_emissions, batch_t_emissions, batch_inputs)
        (
            unc_params_trainable,
            losses,
            unc_params_trainable_history,
            grad_trainable_history,
        ) = run_sgd(
            _loss_fn,
            initial_unc_params_trainable,
            dataset,
            optimizer=optimizer,
            batch_size=batch_size,
            num_epochs=num_epochs,
            shuffle=shuffle,
            return_param_history=return_param_history,
            return_grad_history=return_grad_history,
            key=key,
        )

        # Untrainable parameters will appear as None in history
        # We will fill in these none values with the initial unconstrained parameters,
        # and broadcast them to the correct shape.
        # It will appear as though the sampler has not updated these parameters
        #  (in fact,it is ignoring them altogether, and we add them here for easy downstream usage).
        unc_params_fitted = tree_map(
            lambda initial, fitted: initial if fitted is None else fitted,
            initial_unc_params,
            unc_params_trainable,
        )
        # Convert unconstrained parameters back to constrained space
        params_fitted = from_unconstrained(unc_params_fitted, props)

        n_fits = len(
            losses
        )  # could differ from num_epochs if NaN's cause an early quit?
        if unc_params_trainable_history is not None:
            unc_params_history = tree_map(
                lambda initial, fitted: (
                    jnp.broadcast_to(
                        jnp.array(initial), (n_fits,) + jnp.array(initial).shape
                    )
                    if fitted is None
                    else fitted
                ),
                initial_unc_params,
                unc_params_trainable_history,
            )
            # Convert unconstrained parameters back to constrained space
            params_history = from_unconstrained(unc_params_history, props)

        if grad_trainable_history is not None:
            grad_history = tree_map(
                lambda initial, fitted: (
                    jnp.broadcast_to(
                        jnp.zeros_like(initial), (n_fits,) + jnp.array(initial).shape
                    )
                    if fitted is None
                    else fitted
                ),
                initial_unc_params,
                grad_trainable_history,
            )

        # If interested in history of parameters and gradients
        if return_param_history and return_grad_history:
            # Return all
            return params_fitted, losses, params_history, grad_history
        # If not interested in history of parameters
        elif not return_param_history and return_grad_history:
            return params_fitted, losses, grad_history
        # If not interested in history of gradients
        elif return_param_history and not return_grad_history:
            return params_fitted, losses, params_history
        # If not interested in history of parameters and gradients
        else:
            return params_fitted, losses

    # Fit model using SciPy optimizer
    def fit_scipy(
        self,
        initial_params: ParameterSet,
        props: PropertySet,
        emissions: Union[
            Float[Array, "num_timesteps emission_dim"],
            Float[Array, "num_batches num_timesteps emission_dim"],
        ],
        t_emissions: Optional[
            Union[
                Float[Array, "num_timesteps 1"],
                Float[Array, "num_batches num_timesteps 1"],
            ]
        ] = None,
        filter_hyperparams: Optional[Any] = None,
        inputs: Optional[
            Union[
                Float[Array, "num_timesteps input_dim"],
                Float[Array, "num_batches num_timesteps input_dim"],
            ]
        ] = None,
        method: str = "Nelder-Mead",
        options: dict = {"maxiter": 5000},
        return_param_history: bool = False,
        warn: bool = True,
        verbosity: Optional[int] = None,
    ) -> Tuple[ParameterSet, Float[Array, " niter"], Optional[ParameterSet]]:
        """
        Compute parameter MLE/MAP estimate using SciPy-based chosen method.
        SciPy-based optimizers aim to find parameters that maximize the marginal log probability,

        $$\theta^\star = \mathrm{argmax}_\theta \; \log p(y_{1:T}, \theta \mid u_{1:T})$$

        by minimizing the _negative_ of that quantity.

        *Note:* ``emissions`` *and* ``inputs`` *can either be single sequences or batches of sequences.*

        Args:
            initial_params: model parameters $\theta$
            props: model properties specifying which parameters should be learned
            emissions: one or more sequences of observed data
            t_emissions: continuous-time specific time instants: if not None, it is an array
            filter_hyperparams: if needed, hyperparameters of the filtering algorithm
            inputs: one or more sequences of corresponding inputs
            method: optimization method to use, e.g. "Nelder-Mead", "BFGS", etc.
            options: dictionary of options to pass to the SciPy optimizer
            return_param_history: whether to return the history of parameters
            warn: whether to print warnings from filters
        Returns:
            params_fitted: final fitted parameters (PyTree, constrained)
            losses: array of loss values per iteration
            params_history (optional): constrained parameter history (if requested)
        """
        warn, verbosity = resolve_verbosity(warn=warn, verbosity=verbosity)

        # Ensure batch dims
        batch_emissions = ensure_array_has_batch_dim(emissions, self.emission_shape)
        batch_t_emissions = ensure_array_has_batch_dim(t_emissions, (1,))
        batch_inputs = ensure_array_has_batch_dim(inputs, self.inputs_shape)

        # Convert initial parameters to unconstrained space
        initial_unc_params = to_unconstrained(initial_params, props)
        initial_unc_params_trainable = tree_map(
            lambda param, prop: param if prop.trainable else None,
            initial_unc_params,
            props,
        )

        # Fill untrainable params so we can flatten to a single vector
        filled_unc_params = tree_map(
            lambda init, p: init if p is None else p,
            initial_unc_params,
            initial_unc_params_trainable,
        )

        # Ravel/unravel parameters for SciPy
        flat_init, unravel_fn = ravel_pytree(filled_unc_params)

        # Base loss for SciPy in PyTree form
        def _loss_fn(unc_params_trainable):
            unc_params = tree_map(
                lambda init, trained, prop: trained if prop.trainable else init,
                initial_unc_params,
                unc_params_trainable,
                props,
            )
            params = from_unconstrained(unc_params, props)

            # Compute marginal log likelihoods for the full batch
            lls = vmap(
                partial(
                    self.marginal_log_prob,
                    params,
                    filter_hyperparams=filter_hyperparams,
                    warn=warn, verbosity=verbosity,
                )
            )(
                emissions=batch_emissions,
                t_emissions=batch_t_emissions,
                inputs=batch_inputs,
            )
            lp = self.log_prior(params) + lls.sum()
            return -lp / batch_emissions.size

        # Flat loss for SciPy
        def flat_loss_fn(flat_params):
            pytree_params = unravel_fn(flat_params)
            return float(_loss_fn(pytree_params))

        # History storage
        loss_history = []
        param_history = [] if return_param_history else None

        # Callback to store loss and parameters at each iteration
        def callback(flat_params):
            pytree_params = unravel_fn(flat_params)
            loss_history.append(flat_loss_fn(flat_params))
            if return_param_history:
                param_history.append(pytree_params)

        # Run SciPy optimizer
        result = minimize(
            flat_loss_fn, flat_init, method=method, callback=callback, options=options
        )

        # Final fitted parameters (combine fixed + learned)
        unc_params_fitted = unravel_fn(result.x)
        unc_params_fitted = tree_map(
            lambda init, trained: init if trained is None else trained,
            initial_unc_params,
            unc_params_fitted,
        )
        params_fitted = from_unconstrained(unc_params_fitted, props)

        # Convert loss history
        losses = jnp.array(loss_history)
        if return_param_history:
            # Fill missing params before storing
            param_history = [
                tree_map(
                    lambda init, p: init if p is None else p,
                    initial_unc_params,
                    ph,
                )
                for ph in param_history
            ]

            # Stack across iterations (same structure now)
            unc_params_history = tree_map(lambda *xs: jnp.stack(xs), *param_history)
            params_history = from_unconstrained(unc_params_history, props)
            return params_fitted, losses, params_history, result
        else:
            return params_fitted, losses, result

    # Fit model using jaxopt SciPyMinimize
    def fit_scipy_jaxopt(
        self,
        initial_params: ParameterSet,
        props: PropertySet,
        emissions: Union[
            Float[Array, "num_timesteps emission_dim"],
            Float[Array, "num_batches num_timesteps emission_dim"],
        ],
        t_emissions: Optional[
            Union[
                Float[Array, "num_timesteps 1"],
                Float[Array, "num_batches num_timesteps 1"],
            ]
        ] = None,
        filter_hyperparams: Optional[Any] = None,
        inputs: Optional[
            Union[
                Float[Array, "num_timesteps input_dim"],
                Float[Array, "num_batches num_timesteps input_dim"],
            ]
        ] = None,
        method: str = "nelder-mead",
        options: dict = {"maxiter": 100},
    ) -> Tuple[ParameterSet, Float[Array, " niter"]]:
        r"""Compute parameter MLE/ MAP estimate using SciPy-based chosen method from jaxopt.

        ScipyMinimize aims to find parameters that maximize the marginal log probability,

        $$\theta^\star = \mathrm{argmax}_\theta \; \log p(y_{1:T}, \theta \mid u_{1:T})$$

        by minimizing the _negative_ of that quantity.

        *Note:* ``emissions`` *and* ``inputs`` *can either be single sequences or batches of sequences.*

        Args:
            initial_params: model parameters $\theta$
            props: properties specifying which parameters should be learned
            emissions: one or more sequences of emissions
            t_emissions: continuous-time specific time instants: if not None, it is an array
            filter_hyperparams: if needed, hyperparameters of the filtering algorithm
            inputs: one or more sequences of corresponding inputs
            method: optimization method to use, e.g. "Nelder-Mead", "BFGS", etc.
            options: dictionary of options to pass to the SciPy optimizer

        Returns:
            tuple of new parameters and losses (negative scaled marginal log probs) over the course of Nelder-Mead iterations.
        """

        # Ensure batch dims
        batch_emissions = ensure_array_has_batch_dim(emissions, self.emission_shape)
        batch_t_emissions = ensure_array_has_batch_dim(t_emissions, (1,))
        batch_inputs = ensure_array_has_batch_dim(inputs, self.inputs_shape)

        # Convert initial parameters to unconstrained space
        initial_unc_params = to_unconstrained(initial_params, props)
        initial_unc_params_trainable = tree_map(
            lambda param, prop: param if prop.trainable else None,
            initial_unc_params,
            props,
        )

        # Define base loss for SciPyMinimize in PyTree form
        def _loss_fn(unc_params_trainable):
            unc_params = tree_map(
                lambda init, trained, prop: trained if prop.trainable else init,
                initial_unc_params,
                unc_params_trainable,
                props,
            )
            params = from_unconstrained(unc_params, props)

            # Compute marginal log likelihoods for the full batch
            lls = vmap(
                partial(
                    self.marginal_log_prob,
                    params,
                    filter_hyperparams=filter_hyperparams,
                )
            )(
                emissions=batch_emissions,
                t_emissions=batch_t_emissions,
                inputs=batch_inputs,
            )
            lp = self.log_prior(params) + lls.sum()
            return -lp / batch_emissions.size

        # Run Nelder-Mead optimization
        solver = ScipyMinimize(fun=_loss_fn, method=method, options=options)
        solver.implicit_diff = False
        result = solver.run(init_params=initial_unc_params_trainable)

        # Convert loss history to jnp array
        final_loss = jnp.array(result.state.fun_val)

        # Final fitted params (combine fixed + learned)
        unc_params_fitted = tree_map(
            lambda initial, fitted: initial if fitted is None else fitted,
            initial_unc_params,
            result.params,
        )
        params_fitted = from_unconstrained(unc_params_fitted, props)

        # Can only return final params and loss
        # TODO: maybe there is a way to return history?
        # If you want history, use fit_scipy instead.
        # Note that their implementations (perhaps due to versioning) differ empirically,
        # so it is not a drop-in replacement.
        return params_fitted, final_loss

    # Fit model using MCMC, based on Blackjax
    def fit_mcmc(
        self,
        initial_params: ParameterSet,
        props: PropertySet,
        emissions: Union[
            Float[Array, "num_timesteps emission_dim"],
            Float[Array, "num_batches num_timesteps emission_dim"],
        ],
        t_emissions: Optional[
            Union[
                Float[Array, "num_timesteps 1"],
                Float[Array, "num_batches num_timesteps 1"],
            ]
        ] = None,
        filter_hyperparams: Optional[Any] = None,
        inputs: Optional[
            Union[
                Float[Array, "num_timesteps input_dim"],
                Float[Array, "num_batches num_timesteps input_dim"],
            ]
        ] = None,
        mcmc_algorithm={
            "type": "nuts",
            "n_samples": 100,
            "warmup_samples": 10,  # Number of warmup steps,
            "parameters": {},  # Additional parameters for the MCMC algorithm
        },
        verbose=True,
        warn=True,
        verbosity: Optional[int] = None,
        key: PRNGKey = jr.PRNGKey(0),
    ) -> Tuple[
        ParameterSet,
        ParameterSet,
        Float[Array, " num_steps"],
        Float[Array, " n_mcmc_samples"],
    ]:
        r"""Generate samples from the posterior using Markov Chain Monte Carlo (MCMC).

        Args:
            initial_params: initial parameters $\theta$
            props: properties specifying which parameters should be learned
            emissions: one or more sequences of observed data
            t_emissions: continuous-time specific time instants: if not None, it is an array
            filter_hyperparams: if needed, hyperparameters of the filtering algorithm
            inputs: one or more sequences of corresponding inputs
            mcmc_algorithm: dictionary specifying the MCMC algorithm to use and its settings, based on Blackjax.
                It should contain the following keys:
                    type: type of MCMC algorithm (e.g. "nuts")
                    n_samples: number of samples to draw
                    warmup_samples: number of warmup steps
                    parameters: additional parameters for the MCMC algorithm
            verbose: whether or not to show a progress bar
            warn: whether to print warnings from filters
            key: a random number generator

        Returns:
            tuple of samples and log probabilities of the samples
        """
        warn, verbosity = resolve_verbosity(warn=warn, verbosity=verbosity)

        ## cd-dynamax specific code
        # Make sure the emissions and inputs have batch dimensions
        batch_emissions = ensure_array_has_batch_dim(emissions, self.emission_shape)
        batch_t_emissions = ensure_array_has_batch_dim(t_emissions, (1,))
        batch_inputs = ensure_array_has_batch_dim(inputs, self.inputs_shape)

        # Transform initial parameters to unconstrained space
        initial_unc_params = to_unconstrained(initial_params, props)
        # and build initial_unc_params_trainable from initial_unc_params and props
        # by setting untrainable parameters to None
        initial_unc_params_trainable = tree_map(
            lambda param, prop: param if prop.trainable else None,
            initial_unc_params,
            props,
        )

        # The log likelihood that MCMC samples from
        def _logprob(unc_params_trainable):
            # Combine the trainable and non-trainable parameters, then convert them to constrained space
            unc_params = tree_map(
                lambda initial, trained, prop: trained if prop.trainable else initial,
                initial_unc_params,
                unc_params_trainable,
                props,
            )
            params = from_unconstrained(unc_params, props)
            # Compute marginal log likelihoods for the full batch
            batch_lls = vmap(
                partial(
                    self.marginal_log_prob,
                    params,
                    filter_hyperparams=filter_hyperparams,
                    warn=warn, verbosity=verbosity,
                )  # partial with fixed params arg and filter_hyperparams kwarg
            )(
                # arguments to vmap over
                emissions=batch_emissions,
                t_emissions=batch_t_emissions,
                inputs=batch_inputs,
            )
            # Compute the log posterior, including prior
            lp = self.log_prior(params) + batch_lls.sum()
            lp += log_det_jac_constrain(params, props)
            return lp

        ## Blackjax specific code
        # Helper function for the HMC algorithm
        def _run_hmc(
            mcmc_algo,
            mcmc_algorithm,
            _logprob,
            initial_unc_params_trainable,
            verbose,
            key,
        ):
            # Initialize MCMC using window_adaptation
            # https://blackjax-devs.github.io/blackjax/examples/quickstart.html#use-stan-s-window-adaptation
            warmup = blackjax.window_adaptation(
                algorithm=mcmc_algo,
                logdensity_fn=_logprob,
                progress_bar=verbose,
                **mcmc_algorithm["parameters"],
            )

            # Set-up warmup
            warmup_key, key = jr.split(key)
            # Run warmup
            (warmup_state, warmup_kernel_params), warmup_info = warmup.run(
                warmup_key,
                position=initial_unc_params_trainable,
                num_steps=mcmc_algorithm["warmup_samples"],
            )

            # Set-up HMC
            # MCMC-HMC sampling kernel, based on warmup kernel params
            hmc_kernel_step = mcmc_algo(_logprob, **warmup_kernel_params).step

            # HMC sampling step
            @jit
            def __hmc_step(state, carry):
                step_key = carry  # Only keys passed via lax_scan
                state, _ = hmc_kernel_step(step_key, state)
                return state, state

            # Set-up random keys for MCMC sampler
            hmc_keys = jr.split(key, mcmc_algorithm["n_samples"])
            # Run HMC inference loop
            print("Running HMC inference loop...")
            # Using our lax_scan
            _, states = lax_scan(__hmc_step, warmup_state, hmc_keys, debug=DEBUG)

            # Sampled warm-up states
            warmup_params = warmup_info.state.position
            mcmc_params = states.position
            # Keep-track of the samples' log probabilities
            warmup_log_probs = jnp.array(warmup_info.state.logdensity)
            mcmc_log_probs = jnp.array(states.logdensity)

            return warmup_params, mcmc_params, warmup_log_probs, mcmc_log_probs

        # Helper function for the MH algorithm
        def _run_mh(
            mcmc_algo,
            mcmc_algorithm,
            _logprob,
            initial_unc_params_trainable,
            verbose,
            key,
        ):
            # MH algo, with proposal in mcmc_algorithm['parameters']['proposal']
            mh = mcmc_algo(_logprob, eval(mcmc_algorithm["parameters"]["proposal"]))

            # MH sampling kernel step
            mh_kernel_step = mh.step

            # MH sampling step
            @jit
            def _mh_step(state, carry):
                step_key = carry  # Only keys passed via lax_scan
                state, _ = mh_kernel_step(step_key, state)
                return state, state

            # Set-up warmup/burn-in
            warmup_key, key = jr.split(key)

            # Set-up random keys for MH burn-in
            warmup_keys = jr.split(warmup_key, mcmc_algorithm["warmup_samples"])
            # Run MH burn-in loop
            print("Running MH burn-in loop...")
            # Original, using our lax_scan
            warmup_final_state, warmup_states = lax_scan(
                _mh_step,
                mh.init(initial_unc_params_trainable),
                warmup_keys,
                debug=DEBUG,
            )

            # Set-up random keys for MCMC sampler
            mh_keys = jr.split(key, mcmc_algorithm["n_samples"])
            # Run MH inference loop
            print("Running MH inference loop...")
            # Using our lax_scan
            _, states = lax_scan(_mh_step, warmup_final_state, mh_keys, debug=DEBUG)

            # Sampled warm-up states
            warmup_params = warmup_states.position
            mcmc_params = states.position
            # Keep-track of the samples' log probabilities
            warmup_log_probs = jnp.array(warmup_states.logdensity)
            mcmc_log_probs = jnp.array(states.logdensity)

            return warmup_params, mcmc_params, warmup_log_probs, mcmc_log_probs

        # Instantiate blackjax MCMC algorithm
        mcmc_algo = eval("blackjax.{}".format(mcmc_algorithm["type"].lower()))

        # Run MCMC, based on algorithm type
        if (
            mcmc_algorithm["type"].lower() == "nuts"
            or mcmc_algorithm["type"].lower() == "hmc"
        ):
            warmup_params, mcmc_params, warmup_log_probs, mcmc_log_probs = _run_hmc(
                mcmc_algo,
                mcmc_algorithm,
                _logprob,
                initial_unc_params_trainable,
                verbose,
                key,
            )
        elif (
            mcmc_algorithm["type"].lower() == "additive_step_random_walk"
            or mcmc_algorithm["type"].lower() == "rmh"
        ):
            warmup_params, mcmc_params, warmup_log_probs, mcmc_log_probs = _run_mh(
                mcmc_algo,
                mcmc_algorithm,
                _logprob,
                initial_unc_params_trainable,
                verbose,
                key,
            )
        else:
            raise ValueError(
                "Unknown MCMC algorithm type: {}".format(mcmc_algorithm["type"])
            )

        ### Convert MCMC samples to constrained space
        # Untrainable parameters will appear as None in param_samples
        # We will fill in these none values with the initial unconstrained parameters,
        # and broadcast them to the correct shape.
        # It will appear as though the sampler has not updated these parameters
        # (in fact, it is ignoring them altogether, and we add them here for easy downstream usage).
        def _sampled_or_initial_over_tree(initial, sampled):
            return tree_map(
                lambda i, s: (
                    jnp.broadcast_to(
                        jnp.array(i),
                        (mcmc_algorithm["n_samples"],) + jnp.array(i).shape,
                    )
                    if s is None
                    else s
                ),
                initial,
                sampled,
            )

        # Warm-up or burn-in
        warmup_unc_param_samples = _sampled_or_initial_over_tree(
            initial_unc_params, warmup_params
        )
        warmup_param_samples = from_unconstrained(warmup_unc_param_samples, props)

        # Main loop
        mcmc_unc_param_samples = _sampled_or_initial_over_tree(
            initial_unc_params, mcmc_params
        )
        mcmc_param_samples = from_unconstrained(mcmc_unc_param_samples, props)

        return (
            warmup_param_samples,
            mcmc_param_samples,
            warmup_log_probs,
            mcmc_log_probs,
        )

    # For EM, subclasses must implement E- and M-steps
    # E-step
    def e_step(
        self,
        params: ParameterSet,
        emissions: Float[Array, "num_timesteps emission_dim"],
        t_emissions: Optional[Float[Array, "num_timesteps 1"]] = None,
        inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
    ) -> Tuple[SuffStatsSSM, Scalar]:
        r"""Perform an E-step to compute expected sufficient statistics under the posterior, $p(z_{1:T} \mid y_{1:T}, u_{1:T}, \theta)$.

        Args:
            params: model parameters $\theta$
            emissions: emissions $y_{1:T}$
            t_emissions: continuous-time specific time instants: if not None, it is an array
            inputs: optional inputs $u_{1:T}$

        Returns:
            Expected sufficient statistics under the posterior.

        """
        raise NotImplementedError

    # M-step
    def m_step(
        self,
        params: ParameterSet,
        props: PropertySet,
        batch_stats: SuffStatsSSM,
        m_step_state: Any,
    ) -> ParameterSet:
        r"""Perform an M-step to find parameters that maximize the expected log joint probability.

        Specifically, compute

        $$\theta^\star = \mathrm{argmax}_\theta \; \mathbb{E}_{p(z_{1:T} \mid y_{1:T}, u_{1:T}, \theta)} \big[\log p(y_{1:T}, z_{1:T}, \theta \mid u_{1:T}) \big]$$

        Args:
            params: model parameters $\theta$
            props: properties specifying which parameters should be learned
            batch_stats: sufficient statistics from each sequence
            m_step_state: any required state for optimizing the model parameters.

        Returns:
            new parameters

        """
        raise NotImplementedError

    # EM fitting function
    def fit_em(
        self,
        params: ParameterSet,
        props: PropertySet,
        emissions: Union[
            Float[Array, "num_timesteps emission_dim"],
            Float[Array, "num_batches num_timesteps emission_dim"],
        ],
        t_emissions: Optional[
            Union[
                Float[Array, "num_timesteps 1"],
                Float[Array, "num_batches num_timesteps 1"],
            ]
        ] = None,
        inputs: Optional[
            Union[
                Float[Array, "num_timesteps input_dim"],
                Float[Array, "num_batches num_timesteps input_dim"],
            ]
        ] = None,
        num_iters: int = 50,
        verbose: bool = True,
    ) -> Tuple[ParameterSet, Float[Array, " num_iters"]]:
        r"""Compute parameter MLE/ MAP estimate using Expectation-Maximization (EM).

        EM aims to find parameters that maximize the marginal log probability,

        $$\theta^\star = \mathrm{argmax}_\theta \; \log p(y_{1:T}, \theta \mid u_{1:T})$$

        It does so by iteratively forming a lower bound (the "E-step") and then maximizing it (the "M-step").

        *Note:* ``emissions`` *and* ``inputs`` *can either be single sequences or batches of sequences.*

        Args:
            params: model parameters $\theta$
            props: properties specifying which parameters should be learned
            emissions: one or more sequences of emissions
            t_emissions: continuous-time specific time instants: if not None, it is an array
            inputs: one or more sequences of corresponding inputs
            num_iters: number of iterations of EM to run
            verbose: whether or not to show a progress bar

        Returns:
            tuple of new parameters and log likelihoods over the course of EM iterations.

        """

        # Make sure the emissions and inputs have batch dimensions
        batch_emissions = ensure_array_has_batch_dim(emissions, self.emission_shape)
        batch_t_emissions = ensure_array_has_batch_dim(t_emissions, (1,))
        batch_inputs = ensure_array_has_batch_dim(inputs, self.inputs_shape)

        @jit
        def em_step(params, m_step_state):
            batch_stats, lls = vmap(partial(self.e_step, params))(
                batch_emissions, batch_t_emissions, batch_inputs
            )
            lp = self.log_prior(params) + lls.sum()
            params, m_step_state = self.m_step(params, props, batch_stats, m_step_state)
            return params, m_step_state, lp

        log_probs = []
        m_step_state = self.initialize_m_step_state(params, props)
        pbar = progress_bar(range(num_iters)) if verbose else range(num_iters)
        for _ in pbar:
            params, m_step_state, marginal_loglik = em_step(params, m_step_state)
            log_probs.append(marginal_loglik)
        return params, jnp.array(log_probs)

    # Fisher information matrix
    def fisher_information(
        self,
        params: ParameterSet,
        props: PropertySet,
        emissions: Union[
            Float[Array, "num_timesteps emission_dim"],
            Float[Array, "num_batches num_timesteps emission_dim"],
        ],
        t_emissions: Optional[
            Union[
                Float[Array, "num_timesteps 1"],
                Float[Array, "num_batches num_timesteps 1"],
            ]
        ] = None,
        filter_hyperparams: Optional[Any] = None,
        inputs: Optional[
            Union[
                Float[Array, "num_timesteps input_dim"],
                Float[Array, "num_batches num_timesteps input_dim"],
            ]
        ] = None,
        warn: bool = True,
        verbosity: Optional[int] = None,
    ):
        r"""Compute the observed Fisher information matrix for the model parameters.

        Args:
            params: model parameters $\theta$
            props: properties specifying which parameters should be learned
            emissions: one or more sequences of emissions
            t_emissions: continuous-time specific time instants: if not None, it is an array
            filter_hyperparams: if needed, hyperparameters of the filtering algorithm
            inputs: one or more sequences of corresponding inputs
            warn: whether to print warnings from filters
        Returns:
            Fisher information PyTree.

        NOTE:
            The hessian computation requires reverse-mode autodiff.
        """
        warn, verbosity = resolve_verbosity(warn=warn, verbosity=verbosity)

        # Make sure the emissions and inputs have batch dimensions
        batch_emissions = ensure_array_has_batch_dim(emissions, self.emission_shape)
        batch_t_emissions = ensure_array_has_batch_dim(t_emissions, (1,))
        batch_inputs = ensure_array_has_batch_dim(inputs, self.inputs_shape)

        # Transform initial parameters to unconstrained space
        initial_unc_params = to_unconstrained(params, props)
        # build initial_unc_params_trainable from initial_unc_params and props
        # by setting untrainable parameters to None
        initial_unc_params_trainable = tree_map(
            lambda param, prop: param if prop.trainable else None,
            initial_unc_params,
            props,
        )

        # The log probability function
        def _logprob(unc_params_trainable):
            # Combine the trainable and non-trainable parameters, then convert them to constrained space
            unc_params = tree_map(
                lambda initial, trained, prop: trained if prop.trainable else initial,
                initial_unc_params,
                unc_params_trainable,
                props,
            )
            params = from_unconstrained(unc_params, props)
            batch_lls = vmap(
                partial(
                    self.marginal_log_prob,
                    params,
                    filter_hyperparams=filter_hyperparams,
                    warn=warn, verbosity=verbosity,
                )  # partial with fixed params arg and filter_hyperparams kwarg
            )(
                # arguments to vmap over
                emissions=batch_emissions,
                t_emissions=batch_t_emissions,
                inputs=batch_inputs,
            )
            # Compute the log posterior, including prior
            lp = self.log_prior(params) + batch_lls.sum()
            lp += log_det_jac_constrain(params, props)
            return lp

        # Compute the Fisher information matrix using autodiff
        hess, method_used = fallback_hessian(
            f=_logprob, params=initial_unc_params_trainable
        )
        neg_hess = tree_map(lambda x: -x, hess)
        return neg_hess

    # Prior-averaged Fisher information matrix
    def prior_averaged_fisher(
        self,
        prior: Prior,
        initial_params: ParameterSet,
        num_timesteps: int,
        t_emissions: Optional[
            Union[
                Float[Array, "num_timesteps 1"],
                Float[Array, "n_samples num_timesteps 1"],
            ]
        ] = None,
        inputs: Optional[
            Union[
                Float[Array, "num_timesteps input_dim"],
                Float[Array, "n_samples num_timesteps input_dim"],
            ]
        ] = None,
        filter_hyperparams: Optional[Union[Any]] = None,
        transition_type: Optional[str] = "distribution",
        n_samples: int = 100,
        key: PRNGKey = jr.PRNGKey(0),
    ):
        """
        Compute the expected Fisher information matrix with respect to the prior.

        Note: for the Fisher information computation, as we need to compute gradients w.rto the parameters,
        user must ensure that the sampled parameters are trainable

        Args:
            prior: prior distribution
            initial_params: initial parameters to use for those not sampled from the prior
            num_timesteps: number of timesteps $T$
            t_emissions: continuous-time specific time instants: if not None, it is an array
            inputs: inputs $u_{1:T}$
            filter_hyperparams: hyperparameters of the filtering algorithm
            transition_type: type of transition function, either "distribution" (default) or "path"
                "distribution" samples from the (default Gaussian) transition distribution (default)
                    - This is exact for Linear Gaussian SSMs
                "path" runs an SDE solver to sample the distribution. This is more "exact" (up to discretization error).
            n_samples: number of samples to draw from the prior
            key: random number generator

        Returns:
            Fisher information PyTree averaged over the prior.
        """

        # Sample parameters from the prior
        prior_key, key = jr.split(key)
        sampled_ssm_params, sampled_ssm_props = self.sample_prior(
            prior=prior,
            init_params=initial_params,
            M=n_samples,
            key=prior_key,
        )  # Sampled parameters are a PyTree, where each leave has axis=0 for the M samples

        # Sampling states and emissions from each model, given drawn parameteres
        # split the key for each sample
        sampling_key, key = jr.split(key)
        per_sample_keys = jr.split(sampling_key, n_samples)
        # Sample from each model, given drawn parameteres, using vmap
        sampled_states, sampled_emissions = vmap(
            self.sample, in_axes=(0, 0, None, None, None, None)
        )(
            sampled_ssm_params,
            per_sample_keys,
            num_timesteps,
            t_emissions,
            inputs,
            transition_type,
        )

        # Compute the outer product of the gradients of the log likelihood (score) at each emission sample with respect to the parameters
        # For each computation of the score, we will use an independent key in filter hyperparams.
        # This is relevant when the marginal log likelihood computation is random (e.g. via EnKF or particle filter)
        filtering_key, key = jr.split(key)
        per_sample_filtering_keys = jr.split(filtering_key, n_samples)

        # Compute the score for each model, given sampled parameteres, using vmap
        sampled_scores = vmap(
            self.score, in_axes=(0, None, 0, None, None, None, None, 0)
        )(
            sampled_ssm_params,
            sampled_ssm_props,
            sampled_emissions,
            t_emissions,
            filter_hyperparams,
            inputs,
            False,
            per_sample_filtering_keys,
        )

        # Compute H, the outer product of the scores per-sample, for each PyTree leaf
        sampled_H = tree_map(
            vmap(lambda x: jnp.outer(x, x), in_axes=(0)), sampled_scores
        )

        # Now, average sampled H over the samples, for each leaf of the PyTree
        H = tree_map(lambda x: jnp.mean(x, axis=0), sampled_H)
        # Return the expected Fisher information matrix with respect to the prior
        return H
