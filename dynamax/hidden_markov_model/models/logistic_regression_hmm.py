import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb
from jaxtyping import Float, Array
from dynamax.parameters import ParameterProperties
from dynamax.hidden_markov_model.models.abstractions import HMM, HMMEmissions
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState, ParamsStandardHMMInitialState
from dynamax.hidden_markov_model.models.transitions import StandardHMMTransitions, ParamsStandardHMMTransitions
import optax
from typing import NamedTuple, Union


class ParamsLogisticRegressionHMMEmissions(NamedTuple):
    weights: Union[Float[Array, "state_dim input_dim"], ParameterProperties]
    biases: Union[Float[Array, "state_dim"], ParameterProperties]


class LogisticRegressionHMMEmissions(HMMEmissions):

    def __init__(self,
                 num_states,
                 input_dim,
                 emission_matrices_scale=1e8,
                 m_step_optimizer=optax.adam(1e-2),
                 m_step_num_iters=50):
        super().__init__(m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        self.num_states = num_states
        self.input_dim = input_dim
        self.emission_weights_scale = emission_matrices_scale

    @property
    def emission_shape(self):
        return ()

    @property
    def inputs_shape(self):
        return (self.input_dim,)

    def initialize(self,
                   key=jr.PRNGKey(0),
                   method="prior",
                   emission_weights=None,
                   emission_biases=None,
                   emissions=None,
                   inputs=None):

        if method.lower() == "kmeans":
            assert emissions is not None, "Need emissions to initialize the model with K-Means!"
            assert inputs is not None, "Need inputs to initialize the model with K-Means!"
            from sklearn.cluster import KMeans

            flat_emissions = emissions.reshape(-1,)
            flat_inputs = inputs.reshape(-1, self.input_dim)
            km = KMeans(self.num_states).fit(flat_inputs)
            _emission_weights = jnp.zeros((self.num_states, self.input_dim))
            _emission_biases = jnp.array([tfb.Sigmoid().inverse(flat_emissions[km.labels_ == k].mean())
                                          for k in range(self.num_states)])

        elif method.lower() == "prior":
            # TODO: Use an MNIW prior
            key1, key2, key = jr.split(key, 3)
            _emission_weights = 0.01 * jr.normal(key1, (self.num_states, self.input_dim))
            _emission_biases = jr.normal(key2, (self.num_states,))

        else:
            raise Exception("Invalid initialization method: {}".format(method))

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0
        params = ParamsLogisticRegressionHMMEmissions(
            weights=default(emission_weights, _emission_weights),
            biases=default(emission_biases, _emission_biases))
        props = ParamsLogisticRegressionHMMEmissions(
            weights=ParameterProperties(),
            biases=ParameterProperties())
        return params, props

    def log_prior(self, params):
        return tfd.Normal(0, self.emission_weights_scale).log_prob(params.weights).sum()

    def distribution(self, params, state, inputs):
        logits = params.weights[state] @ inputs + params.biases[state]
        return tfd.Bernoulli(logits=logits)


class ParamsLogisticRegressionHMM(NamedTuple):
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsLogisticRegressionHMMEmissions


class LogisticRegressionHMM(HMM):
    def __init__(self,
                 num_states,
                 input_dim,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_matrices_scale=1e8,
                 m_step_optimizer=optax.adam(1e-2),
                 m_step_num_iters=50):
        self.inputs_dim = input_dim
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, transition_matrix_concentration=transition_matrix_concentration)
        emission_component = LogisticRegressionHMMEmissions(num_states, input_dim, emission_matrices_scale=emission_matrices_scale, m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    @property
    def inputs_shape(self):
        return (self.inputs_dim,)

    def initialize(self,
                   key=jr.PRNGKey(0),
                   method="prior",
                   initial_probs=None,
                   transition_matrix=None,
                   emission_weights=None,
                   emission_biases=None,
                   emissions=None,
                   inputs=None):
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
            emissions (array, optional): emissions for initializing the parameters with kmeans. Defaults to None.
            inputs (array, optional): inputs for initializing the parameters with kmeans. Defaults to None.

        Returns:
            params: nested dataclasses of arrays containing model parameters.
            props: a nested dictionary of ParameterProperties to specify parameter constraints and whether or not they should be trained.
        """
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_weights=emission_weights, emission_biases=emission_biases, emissions=emissions, inputs=inputs)
        return ParamsLogisticRegressionHMM(**params), ParamsLogisticRegressionHMM(**props)
