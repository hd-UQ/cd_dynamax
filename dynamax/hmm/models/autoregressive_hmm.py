import jax.numpy as jnp
import jax.random as jr
from jax import lax, tree_map
from typing import NamedTuple
from dynamax.hmm.models.abstractions import HMM
from dynamax.hmm.models.initial import StandardHMMInitialState, ParamsStandardHMMInitialState
from dynamax.hmm.models.transitions import StandardHMMTransitions, ParamsStandardHMMTransitions
from dynamax.hmm.models.linreg_hmm import (LinearRegressionHMMEmissions,
                                           ParamsLinearRegressionHMMEmissions)
from dynamax.parameters import ParameterProperties
from dynamax.utils import PSDToRealBijector

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


class ParamsLinearAutoregressiveHMM(NamedTuple):
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsLinearRegressionHMMEmissions


class LinearAutoregressiveHMMEmissions(LinearRegressionHMMEmissions):
    """A linear autoregressive HMM (ARHMM) is a special case of a
    linear regression HMM where the inputs (i.e. features)
    are functions of the past emissions.
    """

    def __init__(self,
                 num_states,
                 emission_dim,
                 num_lags=1):
        self.num_lags = num_lags
        self.emission_dim = emission_dim
        input_dim = num_lags * emission_dim
        super().__init__(num_states, input_dim, emission_dim)

    def initialize(self,
                   key=jr.PRNGKey(0),
                   method="prior",
                   emission_weights=None,
                   emission_biases=None,
                   emission_covariances=None,
                   emissions=None):
        if method.lower() == "kmeans":
            assert emissions is not None, "Need emissions to initialize the model with K-Means!"
            from sklearn.cluster import KMeans
            km = KMeans(self.num_states).fit(emissions.reshape(-1, self.emission_dim))
            _emission_weights = jnp.zeros((self.num_states, self.emission_dim, self.emission_dim * self.num_lags))
            _emission_biases = jnp.array(km.cluster_centers_)
            _emission_covs = jnp.tile(jnp.eye(self.emission_dim)[None, :, :], (self.num_states, 1, 1))

        elif method.lower() == "prior":
            # technically there's an MNIW prior, but that's a bit complicated...
            key1, key2, key = jr.split(key, 3)
            _emission_weights = jnp.zeros((self.num_states, self.emission_dim, self.emission_dim * self.num_lags))
            _emission_weights = _emission_weights.at[:, :, :self.emission_dim].set(0.95 * jnp.eye(self.emission_dim))
            _emission_weights += 0.01 * jr.normal(key1, (self.num_states, self.emission_dim, self.emission_dim * self.num_lags))
            _emission_biases = jr.normal(key2, (self.num_states, self.emission_dim))
            _emission_covs = jnp.tile(jnp.eye(self.emission_dim), (self.num_states, 1, 1))
        else:
            raise Exception("Invalid initialization method: {}".format(method))

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0
        params = ParamsLinearRegressionHMMEmissions(
            weights=default(emission_weights, _emission_weights),
            biases=default(emission_biases, _emission_biases),
            covs=default(emission_covariances, _emission_covs))
        props = ParamsLinearRegressionHMMEmissions(
            weights=ParameterProperties(),
            biases=ParameterProperties(),
            covs=ParameterProperties(constrainer=tfb.Invert(PSDToRealBijector)))
        return params, props


class LinearAutoregressiveHMM(HMM):
    def __init__(self,
                 num_states,
                 emission_dim,
                 num_lags=1,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1
                 ):
        self.emission_dim = emission_dim
        self.num_lags = num_lags
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, transition_matrix_concentration=transition_matrix_concentration)
        emission_component = LinearAutoregressiveHMMEmissions(num_states, emission_dim, num_lags=num_lags)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    @property
    def inputs_shape(self):
        """Return a pytree matching the pytree of tuples specifying the shape(s)
        of a single time step's inputs.
        """
        return (self.num_lags * self.emission_dim,)


    def initialize(self,
                   key=jr.PRNGKey(0),
                   method="prior",
                   initial_probs=None,
                   transition_matrix=None,
                   emission_weights=None,
                   emission_biases=None,
                   emission_covariances=None,
                   emissions=None):
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Note: in the future we may support more initialization schemes, like K-Means.

        Args:
            key (PRNGKey, optional): random number generator for unspecified parameters. Must not be None if there are any unspecified parameters. Defaults to jr.PRNGKey(0).
            method (str, optional): method for initializing unspecified parameters. Currently, only "prior" is allowed. Defaults to "prior".
            initial_probs (array, optional): manually specified initial state probabilities. Defaults to None.
            transition_matrix (array, optional): manually specified transition matrix. Defaults to None.
            emission_weights (array, optional): manually specified emission weights. Defaults to None.
            emission_biases (array, optional): manually specified emission biases. Defaults to None.
            emission_covariance (array, optional): manually specified emission covariance. Defaults to None.
            emissions (array, optional): emissions for initializing the parameters with kmeans. Defaults to None.

        Returns:
            params: nested dataclasses of arrays containing model parameters.
            props: a nested dictionary of ParameterProperties to specify parameter constraints and whether or not they should be trained.
        """
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_weights=emission_weights, emission_biases=emission_biases, emission_covariances=emission_covariances, emissions=emissions)
        return ParamsLinearAutoregressiveHMM(**params), ParamsLinearAutoregressiveHMM(**props)

    def sample(self, params, key, num_timesteps, prev_emissions=None):
        """
        """
        if prev_emissions is None:
            # Default to zeros
            prev_emissions = jnp.zeros((self.num_lags, self.emission_dim))

        def _step(carry, key):
            prev_state, prev_emissions = carry
            key1, key2 = jr.split(key, 2)
            state = self.transition_distribution(params, prev_state).sample(seed=key2)
            emission = self.emission_distribution(params, state, inputs=jnp.ravel(prev_emissions)).sample(seed=key1)
            next_prev_emissions = jnp.row_stack([emission, prev_emissions[:-1]])
            return (state, next_prev_emissions), (state, emission)

        # Sample the initial state
        key1, key2, key = jr.split(key, 3)
        initial_state = self.initial_distribution(params).sample(seed=key1)
        initial_emission = self.emission_distribution(params, initial_state, inputs=jnp.ravel(prev_emissions)).sample(seed=key2)
        initial_prev_emissions = jnp.row_stack([initial_emission, prev_emissions[:-1]])

        # Sample the remaining emissions and states
        next_keys = jr.split(key, num_timesteps - 1)
        _, (next_states, next_emissions) = lax.scan(
            _step, (initial_state, initial_prev_emissions), next_keys)

        # Concatenate the initial state and emission with the following ones
        expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
        states = tree_map(expand_and_cat, initial_state, next_states)
        emissions = tree_map(expand_and_cat, initial_emission, next_emissions)
        return states, emissions

    def compute_inputs(self, emissions, prev_emissions=None):
        """Helper function to compute the matrix of lagged emissions. These are the
        inputs to the fitting functions.

        Args:
            emissions (array): (T, D) array of emissions

        Returns:
            lagged emissions (array): (T, D*num_lags) array of lagged emissions
        """
        if prev_emissions is None:
            # Default to zeros
            prev_emissions = jnp.zeros((self.num_lags, self.emission_dim))

        padded_emissions = jnp.vstack((prev_emissions, emissions))
        num_timesteps = len(emissions)
        return jnp.column_stack([padded_emissions[lag:lag+num_timesteps]
                                 for lag in reversed(range(self.num_lags))])
