from abc import ABC
from abc import abstractmethod
from fastprogress.fastprogress import progress_bar
from functools import partial
import jax.numpy as jnp
import jax.random as jr
from jax import jit, lax, vmap
from jax.tree_util import tree_map
from jaxtyping import Float, Array, PyTree
import optax
from tensorflow_probability.substrates.jax import distributions as tfd
from typing import Optional, Union, Tuple, Any
from typing_extensions import Protocol

from dynamax.parameters import to_unconstrained, from_unconstrained
from dynamax.parameters import ParameterSet, PropertySet
from dynamax.types import PRNGKey, Scalar
from dynamax.utils.optimize import run_sgd
from dynamax.utils.utils import ensure_array_has_batch_dim

from utils.diffrax_utils import diffeqsolve

class Posterior(Protocol):
    """A :class:`NamedTuple` with parameters stored as :class:`jax.DeviceArray` in the leaf nodes."""
    pass

class SuffStatsSSM(Protocol):
    """A :class:`NamedTuple` with sufficient statics stored as :class:`jax.DeviceArray` in the leaf nodes."""
    pass

class SSM(ABC):
    r"""A base class for state space models. Such models consist of parameters, which
    we may learn, as well as hyperparameters, which specify static properties of the
    model. This base class allows parameters to be indicated a standardized way
    so that they can easily be converted to/from unconstrained form for optimization.

    **Abstract Methods**

    Models that inherit from `SSM` must implement a few key functions and properties:

    * :meth:`initial_distribution` returns the distribution over the initial state given parameters
    * :meth:`transition_distribution` returns the conditional distribution over the next state given the current state and parameters
    * :meth:`emission_distribution` returns the conditional distribution over the emission given the current state and parameters
    * :meth:`log_prior` (optional) returns the log prior probability of the parameters
    * :attr:`emission_shape` returns a tuple specification of the emission shape
    * :attr:`inputs_shape` returns a tuple specification of the input shape, or `None` if there are no inputs.

    The shape properties are required for properly handling batches of data.

    **Sampling and Computing Log Probabilities**

    Once these have been implemented, subclasses will inherit the ability to sample
    and compute log joint probabilities from the base class functions:

    * :meth:`sample` draws samples of the states and emissions for given parameters
    * :meth:`log_prob` computes the log joint probability of the states and emissions for given parameters

    **Inference**

    Many subclasses of SSMs expose basic functions for performing state inference.

    * :meth:`marginal_log_prob` computes the marginal log probability of the emissions, summing over latent states
    * :meth:`filter` computes the filtered posteriors
    * :meth:`smoother` computes the smoothed posteriors

    **Learning**

    Likewise, many SSMs will support learning with expectation-maximization (EM) or stochastic gradient descent (SGD).

    For expectation-maximization, subclasses must implement the E- and M-steps.

    * :meth:`e_step` computes the expected sufficient statistics for a sequence of emissions, given parameters
    * :meth:`m_step` finds new parameters that maximize the expected log joint probability

    Once these are implemented, the generic SSM class allows to fit the model with EM

    * :meth:`fit_em` run EM to find parameters that maximize the likelihood (or posterior) probability.

    For SGD, any subclass that implements :meth:`marginal_log_prob` inherits the base class fitting function

    * :meth:`fit_sgd` run SGD to minimize the *negative* marginal log probability.

    """

    @abstractmethod
    def initial_distribution(
        self,
        params: ParameterSet,
        inputs: Optional[Float[Array, "input_dim"]]
    ) -> tfd.Distribution:
        r"""Return an initial distribution over latent states.

        Args:
            params: model parameters $\theta$
            inputs: optional  inputs  $u_t$

        Returns:
            distribution over initial latent state, $p(z_1 \mid \theta)$.

        """
        raise NotImplementedError

    @abstractmethod
    def transition_distribution(
        self,
        params: ParameterSet,
        state: Float[Array, "state_dim"],
        t0: Optional[Float],
        t1: Optional[Float],
        inputs: Optional[Float[Array, "input_dim"]]
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
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "input_dim"]]=None
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

    def log_prior(
        self,
        params: ParameterSet
    ) -> Scalar:
        r"""Return the log prior probability of any model parameters.

        Returns:
            lp (Scalar): log prior probability.
        """
        return 0.0

    @property
    @abstractmethod
    def emission_shape(self) -> Tuple[int]:
        r"""Return a pytree matching the pytree of tuples specifying the shape of a single time step's emissions.

        For example, a `GaussianHMM` with $D$ dimensional emissions would return `(D,)`.

        """
        raise NotImplementedError

    @property
    def inputs_shape(self) -> Optional[Tuple[int]]:
        r"""Return a pytree matching the pytree of tuples specifying the shape of a single time step's inputs.

        """
        return None

    # All SSMs support sampling
    def sample(
        self,
        params: ParameterSet,
        key: PRNGKey,
        num_timesteps: int,
        t_emissions: Optional[Float[Array, "num_timesteps 1"]] = None,
        inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
        transition_type: Optional[str] = "distribution",
    ) -> Tuple[Float[Array, "num_timesteps state_dim"], Float[Array, "num_timesteps emission_dim"]]:
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
        def _step(prev_state, args):
            key, t0, t1, inpt = args
            key1, key2 = jr.split(key, 2)
            if transition_type == "distribution":
                print("Sampling from transition distribution (this may be a poor approximation if you're simulating from a non-linear SDE). It is a highly appropriate choice for linear SDEs.")
                state = self.transition_distribution(params, prev_state, t0, t1, inpt).sample(seed=key2)
            elif transition_type == "path":
                print("Sampling from SDE solver path (this may be an unnecessarily poor approximation if you're simulating from a linear SDE). It is an appropriate choice for non-linear SDEs.")
                def drift(t, y, args):
                    return params.dynamics.drift.f(y, inpt, t)

                def diffusion(t, y, args):
                    Qc_t = params.dynamics.diffusion_cov.f(None, inpt, t)
                    L_t = params.dynamics.diffusion_coefficient.f(None, inpt, t)
                    Q_sqrt = jnp.linalg.cholesky(Qc_t)
                    combined_diffusion = L_t @ Q_sqrt
                    return combined_diffusion

                state = diffeqsolve(key=key2, drift=drift, diffusion=diffusion, t0=t0, t1=t1, y0=prev_state)[0]
            else:
                raise ValueError("transition_type must be either 'distribution' or 'path'")

            emission = self.emission_distribution(params, state, inpt).sample(seed=key1)
            return state, (state, emission)

        # Sample the initial state
        key1, key2, key = jr.split(key, 3)
        initial_input = tree_map(lambda x: x[0], inputs)
        initial_state = self.initial_distribution(params, initial_input).sample(seed=key1)
        initial_emission = self.emission_distribution(params, initial_state, initial_input).sample(seed=key2)

        # Figure out timestamps, as vectors to scan over
        # t_emissions is of shape num_timesteps \times 1
        # t0 and t1 are num_timesteps-1 \times 0
        if t_emissions is not None:
            num_timesteps = t_emissions.shape[0]
            t0 = tree_map(lambda x: x[0:-1,0], t_emissions)
            t1 = tree_map(lambda x: x[1:,0], t_emissions)
        else:
            t0 = jnp.arange(num_timesteps-1)
            t1 = jnp.arange(1,num_timesteps)

        # Sample the remaining emissions and states
        next_keys = jr.split(key, num_timesteps - 1)
        next_inputs = tree_map(lambda x: x[1:], inputs)
        _, (next_states, next_emissions) = lax.scan(_step, initial_state, (next_keys, t0, t1, next_inputs))

        # Concatenate the initial state and emission with the following ones
        expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
        states = tree_map(expand_and_cat, initial_state, next_states)
        emissions = tree_map(expand_and_cat, initial_emission, next_emissions)
        return states, emissions

    def log_prob(
        self,
        params: ParameterSet,
        states: Float[Array, "num_timesteps state_dim"],
        emissions: Float[Array, "num_timesteps emission_dim"],
        t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
        inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
    ) -> Scalar:
        r"""Compute the log joint probability of the states and observations"""

        def _step(carry, args):
            lp, prev_state = carry
            state, emission, t0, t1, inpt = args
            lp += self.transition_distribution(params, prev_state, t0, t1, inpt).log_prob(state)
            lp += self.emission_distribution(params, state, inpt).log_prob(emission)
            return (lp, state), None

        # Compute log prob of initial time step
        initial_state = tree_map(lambda x: x[0], states)
        initial_emission = tree_map(lambda x: x[0], emissions)
        initial_input = tree_map(lambda x: x[0], inputs)
        lp = self.initial_distribution(params, initial_input).log_prob(initial_state)
        lp += self.emission_distribution(params, initial_state, initial_input).log_prob(initial_emission)

        # Figure out timestamps, as vectors to scan over
        # t_emissions is of shape num_timesteps \times 1
        # t0 and t1 are num_timesteps-1 \times 0
        if t_emissions is not None:
            num_timesteps = t_emissions.shape[0]
            t0 = tree_map(lambda x: x[0:-1,0], t_emissions)
            t1 = tree_map(lambda x: x[1:,0], t_emissions)
        else:
            num_timesteps = len(emissions)
            t0 = jnp.arange(num_timesteps-1)
            t1 = jnp.arange(1,num_timesteps)

        # Scan over remaining time steps
        next_states = tree_map(lambda x: x[1:], states)
        next_emissions = tree_map(lambda x: x[1:], emissions)
        next_inputs = tree_map(lambda x: x[1:], inputs)
        (lp, _), _ = lax.scan(_step, (lp, initial_state), (next_states, next_emissions, t0, t1, next_inputs))
        return lp

    # Some SSMs will implement these inference functions.
    def marginal_log_prob(
        self,
        params: ParameterSet,
        filter_hyperparams: Optional[Any],
        emissions: Float[Array, "ntime emission_dim"],
        t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
        inputs: Optional[Float[Array, "ntime input_dim"]]=None,
    ) -> Scalar:
        r"""Compute log marginal likelihood of observations, $\log \sum_{z_{1:T}} p(y_{1:T}, z_{1:T} \mid \theta)$.

        Args:
            params: model parameters $\theta$
            t_emissions: continuous-time specific time instants: if not None, it is an array 
            state: current latent state $z_t$
            inputs: current inputs  $u_t$

        Returns:
            marginal log probability

        """
        raise NotImplementedError

    def filter(
        self,
        params: ParameterSet,
        filter_hyperparams: Optional[Union[Any]],
        emissions: Float[Array, "ntime emission_dim"],
        t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
        inputs: Optional[Float[Array, "ntime input_dim"]]=None,
    ) -> Posterior:
        r"""Compute filtering distributions, $p(z_t \mid y_{1:t}, u_{1:t}, \theta)$ for $t=1,\ldots,T$.

        Args:
            params: model parameters $\theta$
            t_emissions: continuous-time specific time instants: if not None, it is an array 
            state: current latent state $z_t$
            inputs: current inputs  $u_t$

        Returns:
            filtering distributions

        """
        raise NotImplementedError

    def smoother(
        self,
        params: ParameterSet,
        filter_hyperparams: Optional[Union[Any]],
        emissions: Float[Array, "ntime emission_dim"],
        t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
        inputs: Optional[Float[Array, "ntime input_dim"]]=None,
    ) -> Posterior:
        r"""Compute smoothing distribution, $p(z_t \mid y_{1:T}, u_{1:T}, \theta)$ for $t=1,\ldots,T$.

        Args:
            params: model parameters $\theta$
            t_emissions: continuous-time specific time instants: if not None, it is an array 
            state: current latent state $z_t$
            inputs: current inputs  $u_t$

        Returns:
            smoothing distributions

        """
        raise NotImplementedError

    # Learning algorithms
    def e_step(
        self,
        params: ParameterSet,
        emissions: Float[Array, "num_timesteps emission_dim"],
        t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
        inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None
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

    def m_step(
        self,
        params: ParameterSet,
        props: PropertySet,
        batch_stats: SuffStatsSSM,
        m_step_state: Any
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

    def fit_em(
        self,
        params: ParameterSet,
        props: PropertySet,
        emissions: Union[Float[Array, "num_timesteps emission_dim"],
                         Float[Array, "num_batches num_timesteps emission_dim"]],
        t_emissions: Optional[Union[Float[Array, "num_timesteps 1"],
                        Float[Array, "num_batches num_timesteps 1"]]]=None,
        inputs: Optional[Union[Float[Array, "num_timesteps input_dim"],
                               Float[Array, "num_batches num_timesteps input_dim"]]]=None,
        num_iters: int=50,
        verbose: bool=True
    ) -> Tuple[ParameterSet, Float[Array, "num_iters"]]:
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
            batch_stats, lls = vmap(partial(self.e_step, params))(batch_emissions, batch_t_emissions, batch_inputs)
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

    def fit_sgd(
        self,
        params: ParameterSet,
        props: PropertySet,
        emissions: Union[Float[Array, "num_timesteps emission_dim"],
                         Float[Array, "num_batches num_timesteps emission_dim"]],
        filter_hyperparams: Optional[Any],
        t_emissions: Optional[Union[Float[Array, "num_timesteps 1"],
                        Float[Array, "num_batches num_timesteps 1"]]]=None,
        inputs: Optional[Union[Float[Array, "num_timesteps input_dim"],
                               Float[Array, "num_batches num_timesteps input_dim"]]]=None,
        optimizer: optax.GradientTransformation=optax.adam(1e-3),
        batch_size: int=1,
        num_epochs: int=50,
        shuffle: bool=False,
        key: PRNGKey=jr.PRNGKey(0)
    ) -> Tuple[ParameterSet, Float[Array, "niter"]]:
        r"""Compute parameter MLE/ MAP estimate using Stochastic Gradient Descent (SGD).

        SGD aims to find parameters that maximize the marginal log probability,

        $$\theta^\star = \mathrm{argmax}_\theta \; \log p(y_{1:T}, \theta \mid u_{1:T})$$

        by minimizing the _negative_ of that quantity.

        *Note:* ``emissions`` *and* ``inputs`` *can either be single sequences or batches of sequences.*

        On each iteration, the algorithm grabs a *minibatch* of sequences and takes a gradient step.
        One pass through the entire set of sequences is called an *epoch*.

        Args:
            params: model parameters $\theta$
            props: properties specifying which parameters should be learned
            emissions: one or more sequences of emissions
            filter_hyperparams: if needed, hyperparameters of the filtering algorithm
            t_emissions: continuous-time specific time instants: if not None, it is an array 
            inputs: one or more sequences of corresponding inputs
            optimizer: an `optax` optimizer for minimization
            batch_size: number of sequences per minibatch
            num_epochs: number of epochs of SGD to run
            key: a random number generator for selecting minibatches
            verbose: whether or not to show a progress bar

        Returns:
            tuple of new parameters and losses (negative scaled marginal log probs) over the course of SGD iterations.

        """
        # Make sure the emissions and inputs have batch dimensions
        batch_emissions = ensure_array_has_batch_dim(emissions, self.emission_shape)
        batch_t_emissions = ensure_array_has_batch_dim(t_emissions, (1,))
        batch_inputs = ensure_array_has_batch_dim(inputs, self.inputs_shape)

        unc_params = to_unconstrained(params, props)

        def _loss_fn(unc_params, minibatch):
            """Default objective function."""
            params = from_unconstrained(unc_params, props)
            minibatch_emissions, minibatch_t_emissions, minibatch_inputs = minibatch
            scale = len(batch_emissions) / len(minibatch_emissions)
            minibatch_lls = vmap(partial(self.marginal_log_prob, params, filter_hyperparams))(minibatch_emissions, minibatch_t_emissions, minibatch_inputs)
            lp = self.log_prior(params) + minibatch_lls.sum() * scale
            return -lp / batch_emissions.size

        dataset = (batch_emissions, batch_t_emissions, batch_inputs)
        unc_params, losses = run_sgd(_loss_fn,
                                     unc_params,
                                     dataset,
                                     optimizer=optimizer,
                                     batch_size=batch_size,
                                     num_epochs=num_epochs,
                                     shuffle=shuffle,
                                     key=key)

        params = from_unconstrained(unc_params, props)
        return params, losses
