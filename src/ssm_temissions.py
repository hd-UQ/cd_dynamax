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

from dynamax.parameters import to_unconstrained, from_unconstrained, log_det_jac_constrain
from dynamax.parameters import ParameterSet, PropertySet
from dynamax.types import PRNGKey, Scalar
from dynamax.utils.utils import ensure_array_has_batch_dim, pytree_stack

# From our codebase
from utils.diffrax_utils import diffeqsolve
from utils.debug_utils import lax_scan
from utils.optimize_utils import run_sgd

import blackjax
from fastprogress.fastprogress import progress_bar

DEBUG = False

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

    @property
    def diffeqsolve_settings(self) -> dict:
        r"""Return a dictionary of settings for the differential equation solver.

        """
        return {}

    def sample_batch(
        self,
        params: ParameterSet,
        key: PRNGKey,
        num_sequences: int,
        num_timesteps: int,
        t_emissions: Optional[Float[Array, "num_timesteps 1"]] = None,
        inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
        transition_type: Optional[str] = "distribution"
    ) -> Tuple[Float[Array, "num_sequences num_timesteps state_dim"], Float[Array, "num_sequences num_timesteps emission_dim"]]:

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
            return self.sample(params, key, num_timesteps, t_emissions, inputs, transition_type)

        keys = jr.split(key, num_sequences)
        # use vmap to sample multiple sequences in parallel
        states, emissions = vmap(_sample_sequence)(keys)
        return states, emissions

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
        if transition_type == "distribution":
            print("Sampling from CD distributions: this may be a poor approximation if you're simulating from a non-linear SDE. It is a highly appropriate choice for linear SDEs.")
            states, emissions = self.sample_dist(
                params,
                key,
                num_timesteps,
                t_emissions,
                inputs
            )
        elif transition_type == "path":
            print("Sampling from SDE solver path: this may be an unnecessarily poor approximation if you're simulating from a linear SDE. It is an appropriate choice for non-linear SDEs.")
            states, emissions = self.sample_path(
                params,
                key,
                num_timesteps,
                t_emissions,
                inputs
            )
        else:
            raise ValueError(f"Invalid transition_type: {transition_type}")
        
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
        emissions: Float[Array, "ntime emission_dim"],
        t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
        filter_hyperparams: Optional[Any]=None,
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
        emissions: Float[Array, "ntime emission_dim"],
        t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
        filter_hyperparams: Optional[Union[Any]]=None,
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
        emissions: Float[Array, "ntime emission_dim"],
        t_emissions: Optional[Float[Array, "num_timesteps 1"]]=None,
        filter_hyperparams: Optional[Union[Any]]=None,
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
        t_emissions: Optional[Union[Float[Array, "num_timesteps 1"],
                        Float[Array, "num_batches num_timesteps 1"]]]=None,
        filter_hyperparams: Optional[Any]=None,
        inputs: Optional[Union[Float[Array, "num_timesteps input_dim"],
                               Float[Array, "num_batches num_timesteps input_dim"]]]=None,
        optimizer: optax.GradientTransformation=optax.adam(1e-3),
        batch_size: int=1,
        num_epochs: int=50,
        shuffle: bool=False,
        return_param_history: bool=False,
        return_grad_history: bool=False,
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
            t_emissions: continuous-time specific time instants: if not None, it is an array
            filter_hyperparams: if needed, hyperparameters of the filtering algorithm
            inputs: one or more sequences of corresponding inputs
            optimizer: an `optax` optimizer for minimization
            batch_size: number of sequences per minibatch
            num_epochs: number of epochs of SGD to run
            return_param_history: whether to return the history of parameters
            return_grad_history: whether to return the history of gradients
            key: a random number generator for selecting minibatches
            verbose: whether or not to show a progress bar

        Returns:
            tuple of new parameters and losses (negative scaled marginal log probs) over the course of SGD iterations.
            if interested in the history of parameters and gradients, these are returned as well.

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
            minibatch_lls = vmap(
                partial(
                    self.marginal_log_prob,
                    params,
                    filter_hyperparams=filter_hyperparams,
                    ) # partial with fixed params arg and filter_hyperparams kwarg
                )(
                # arguments to vmap over
                emissions=minibatch_emissions,
                t_emissions=minibatch_t_emissions,
                inputs=minibatch_inputs
            )
            lp = self.log_prior(params) + minibatch_lls.sum() * scale
            return -lp / batch_emissions.size

        dataset = (batch_emissions, batch_t_emissions, batch_inputs)
        unc_params, losses, unc_params_history, grad_history = run_sgd(_loss_fn,
                                    unc_params,
                                    dataset,
                                    optimizer=optimizer,
                                    batch_size=batch_size,
                                    num_epochs=num_epochs,
                                    shuffle=shuffle,
                                    return_param_history=return_param_history,
                                    return_grad_history=return_grad_history,
                                    key=key
                                )

        # Convert unconstrained parameters back to constrained space
        params = from_unconstrained(unc_params, props)
        params_history = from_unconstrained(unc_params_history, props)

        # If interested in history of parameters and gradients
        if return_param_history and return_grad_history:
            # Return all
            return params, losses, params_history, grad_history
        # If not interested in history of parameters
        elif not return_param_history and return_grad_history:
            return params, losses, grad_history
        # If not interested in history of gradients
        elif return_param_history and not return_grad_history:
            return params, losses, params_history
        # If not interested in history of parameters and gradients
        else:
            return params, losses

    def fit_mcmc(
            self,
            initial_params: ParameterSet,
            props: PropertySet,
            emissions: Union[Float[Array, "num_timesteps emission_dim"],
                            Float[Array, "num_batches num_timesteps emission_dim"]],
            t_emissions: Optional[Union[Float[Array, "num_timesteps 1"],
                            Float[Array, "num_batches num_timesteps 1"]]]=None,
            filter_hyperparams: Optional[Any]=None,
            inputs: Optional[Union[Float[Array, "num_timesteps input_dim"],
                                Float[Array, "num_batches num_timesteps input_dim"]]]=None,
            n_mcmc_samples: int=500,
            mcmc_algorithm={
                "type": "nuts",
                "parameters": {
                    "num_steps": 4 # Number of warmup steps
                }
            },
            verbose=True,
            key: PRNGKey=jr.PRNGKey(0)
        ) -> Tuple[ParameterSet, ParameterSet, Float[Array, "num_steps"], Float[Array, "n_mcmc_samples"]]:
            r"""Generate samples from the posterior using Hamiltonian Monte Carlo (HMC).

            Args:
                initial_params: initial parameters $\theta$
                props: properties specifying which parameters should be learned
                emissions: one or more sequences of emissions
                t_emissions: continuous-time specific time instants: if not None, it is an array
                filter_hyperparams: if needed, hyperparameters of the filtering algorithm
                inputs: one or more sequences of corresponding inputs
                num_samples: number of samples to draw
                warmup_steps: number of warmup steps
                num_integration_steps: number of integration steps in the HMC sampler
                verbose: whether or not to show a progress bar
                key: a random number generator

            Returns:
                tuple of samples and log probabilities of the samples
            """

            ## cd-dynamax specific code
            # Make sure the emissions and inputs have batch dimensions
            batch_emissions = ensure_array_has_batch_dim(emissions, self.emission_shape)
            batch_t_emissions = ensure_array_has_batch_dim(t_emissions, (1,))
            batch_inputs = ensure_array_has_batch_dim(inputs, self.inputs_shape)

            initial_unc_params = to_unconstrained(initial_params, props)

            # build initial_unc_params_trainable from initial_unc_params and props
            # by setting trainable parameters to None
            initial_unc_params_trainable = tree_map(
                lambda param, prop: param if prop.trainable else None, initial_unc_params, props
            )

            # The log likelihood that the HMC samples from
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
                        filter_hyperparams=filter_hyperparams
                        ) # partial with fixed params arg and filter_hyperparams kwarg
                    )(
                    # arguments to vmap over
                    emissions=batch_emissions,
                    t_emissions=batch_t_emissions,
                    inputs=batch_inputs
                )
                lp = self.log_prior(params) + batch_lls.sum()
                lp += log_det_jac_constrain(params, props)
                return lp

            ## Blackjax - MCMC specific code
            # Instantiate blackjax MCMC algorithm
            mcmc_algo = eval(
                'blackjax.{}'.format(
                    mcmc_algorithm['type'].lower()
                )
            )
            
            # Initialize MCMC using window_adaptation
            # https://blackjax-devs.github.io/blackjax/examples/quickstart.html#use-stan-s-window-adaptation
            warmup = blackjax.window_adaptation(
                algorithm=mcmc_algo,
                logprob_fn=_logprob,
                progress_bar=verbose,
                **mcmc_algorithm['parameters'],
            )

            # Set-up warmup
            warmup_key, key = jr.split(key)
            # Run warmup
            warmup_state, warmup_kernel, \
                 (warmup_states, warmup_info, warmup_window_adaptation_state) \
                    = warmup.run(
                warmup_key,
                initial_unc_params_trainable,
            )

            # Set-up MCMC
            # MCMC sampling kernel, based on warmup kernel
            mcmc_kernel = warmup_kernel

            # MCMC sampling step
            @jit
            def _mcmc_step(state, step_key):
                state, _ = mcmc_kernel(step_key, state)
                return state, state

            # Set-up random keys for MCMC sampler
            mcmc_keys = jr.split(key, n_mcmc_samples)
            # Run MCMC inference loop
            print('Running MCMC inference loop...')
            _, states = lax_scan(
                _mcmc_step,
                warmup_state,
                mcmc_keys,
                debug=DEBUG
            )

            '''
            print('Trying blackjax scanbar MCMC inference loop...')
            from blackjax.progress_bar import gen_scan_fn
            scan_fn = gen_scan_fn(
                n_mcmc_samples,
                progress_bar=True
            )

            _, states = scan_fn(
                _mcmc_step,
                warmup_state,
                mcmc_keys,
            )
            '''

            # Convert MCMC samples to constrained space
            warmup_param_samples = from_unconstrained(warmup_states.position, props)
            mcmc_param_samples = from_unconstrained(states.position, props)

            # Compute log probabilities of the samples
            warmup_log_probs = jnp.array(-warmup_states.potential_energy)
            mcmc_log_probs = jnp.array(-states.potential_energy)
            
            # Un-trained parameters will appear as None in param_samples
            # We will fill in these none values with the initial parameters,
            # and broadcast them to the correct shape.
            # It will appear as though the sampler has not updated these parameters (in fact,
            # it is ignoring them altogether, and we add them here for easy downstream usage).
            warmup_param_samples = tree_map(
                lambda initial, sampled: (
                    jnp.broadcast_to(
                        jnp.array(initial),
                        (mcmc_algorithm["parameters"]["num_steps"],) + jnp.array(initial).shape) if sampled is None else sampled
                ),
                initial_params,
                warmup_param_samples,
            )
            
            mcmc_param_samples = tree_map(
                lambda initial, sampled: (
                    jnp.broadcast_to(
                        jnp.array(initial),
                        (n_mcmc_samples,) + jnp.array(initial).shape) if sampled is None else sampled
                ),
                initial_params,
                mcmc_param_samples,
            )

            return warmup_param_samples, mcmc_param_samples, warmup_log_probs, mcmc_log_probs
