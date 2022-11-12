import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from dynamax.parameters import ParameterProperties, ParameterSet, PropertySet
from dynamax.hidden_markov_model.models.abstractions import HMMEmissions, HMM
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState, ParamsStandardHMMInitialState
from dynamax.hidden_markov_model.models.transitions import StandardHMMTransitions, ParamsStandardHMMTransitions
from dynamax.utils.utils import pytree_sum
from jaxtyping import Float, Array
from typing import NamedTuple, Optional, Tuple, Union


class ParamsBernoulliHMMEmissions(NamedTuple):
    probs: Union[Float[Array, "emission_dim"], ParameterProperties]


class ParamsBernoulliHMM(NamedTuple):
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsBernoulliHMMEmissions


class BernoulliHMMEmissions(HMMEmissions):

    def __init__(self,
                 num_states,
                 emission_dim,
                 emission_prior_concentration1=1.1,
                 emission_prior_concentration0=1.1):
        """_summary_
        Args:
            emission_probs (_type_): _description_
        """
        self.num_states = num_states
        self.emission_dim = emission_dim
        self.emission_prior_concentration0 = emission_prior_concentration0
        self.emission_prior_concentration1 = emission_prior_concentration1

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    def distribution(self, params, state, inputs=None):
        # This model assumes the emissions are a vector of conditionally independent
        # Bernoulli observations. The `reinterpreted_batch_ndims` argument tells
        # `tfd.Independent` that only the last dimension should be considered a "batch"
        # of conditionally independent observations.
        return tfd.Independent(tfd.Bernoulli(probs=params.probs[state]), reinterpreted_batch_ndims=1)

    def log_prior(self, params):
        prior = tfd.Beta(self.emission_prior_concentration1,
                         self.emission_prior_concentration0)
        return prior.log_prob(params.probs).sum()

    def initialize(self, key=jr.PRNGKey(0), method="prior", emission_probs=None):
        if emission_probs is None:
            if method.lower() == "prior":
                prior = tfd.Beta(self.emission_prior_concentration1, self.emission_prior_concentration0)
                emission_probs = prior.sample(seed=key, sample_shape=(self.num_states, self.emission_dim))
            elif method.lower() == "kmeans":
                raise NotImplementedError("kmeans initialization is not yet implemented!")
            else:
                raise Exception("invalid initialization method: {}".format(method))
        else:
            assert emission_probs.shape == (self.num_states, self.emission_dim)
            assert jnp.all(emission_probs >= 0)
            assert jnp.all(emission_probs <= 0)

        # Add parameters to the dictionary
        params = ParamsBernoulliHMMEmissions(probs=emission_probs)
        props = ParamsBernoulliHMMEmissions(probs=ParameterProperties(constrainer=tfb.Sigmoid()))
        return params, props

    def collect_suff_stats(self, params, posterior, emissions, inputs=None):
        expected_states = posterior.smoothed_probs
        sum_x = jnp.einsum("tk, ti->ki", expected_states, jnp.where(jnp.isnan(emissions), 0, emissions))
        sum_1mx = jnp.einsum("tk, ti->ki", expected_states, jnp.where(jnp.isnan(emissions), 0, 1 - emissions))
        return (sum_x, sum_1mx)

    def initialize_m_step_state(self, params, props):
        return None

    def m_step(self, params, props, batch_stats, m_step_state):
        if props.probs.trainable:
            sum_x, sum_1mx = pytree_sum(batch_stats, axis=0)
            probs = tfd.Beta(
                self.emission_prior_concentration1 + sum_x,
                self.emission_prior_concentration0 + sum_1mx).mode()
            params = params._replace(probs=probs)
        return params, m_step_state


class BernoulliHMM(HMM):
    r"""An HMM with conditionally independent Bernoulli emissions.

    Let $y_t \in \{0,1\}^N$ denote a binary vector of emissions at time $t$. In this model,
    the emission distribution is,

    $$p(y_t \mid z_t, \theta) = \prod_{n=1}^N \mathrm{Bern}(y_{tn} \mid \theta_{z_t,n})$$
    $$p(\theta) = \prod_{k=1}^K \prod_{n=1}^N \mathrm{Beta}(\theta_{k,n}; \gamma_0, \gamma_1)$$

    with $\theta_{k,n} \in [0,1]$ for $k=1,\ldots,K$ and $n=1,\ldots,N$ are the
    *emission probabilities* and $\gamma_0, \gamma_1$ are their prior pseudocounts.

    :param num_states: number of discrete states $K$
    :param emission_dim: number of conditionally independent emissions $N$
    :param initial_probs_concentration: $\alpha$
    :param transition_matrix_concentration: $\beta$
    :param emission_prior_concentration0: $\gamma_0$
    :param emission_prior_concentration1: $\gamma_1$

    """
    def __init__(self, num_states: int,
                 emission_dim: int,
                 initial_probs_concentration=1.1,
                 transition_matrix_concentration=1.1,
                 emission_prior_concentration0=1.1,
                 emission_prior_concentration1=1.1):
        self.emission_dim = emission_dim
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, transition_matrix_concentration=transition_matrix_concentration)
        emission_component = BernoulliHMMEmissions(num_states, emission_dim, emission_prior_concentration0=emission_prior_concentration0, emission_prior_concentration1=emission_prior_concentration1)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    def initialize(self,
                   key: jr.PRNGKey=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: Optional[Float[Array, "num_states"]]=None,
                   transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
                   emission_probs: Optional[Float[Array, "num_states emission_dim"]]=None
    ) -> Tuple[ParameterSet, PropertySet]:
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Note: in the future we may support more initialization schemes, like K-Means.

        Args:
            key: random number generator for unspecified parameters. Must not be None if there are any unspecified parameters. Defaults to jr.PRNGKey(0).
            method: method for initializing unspecified parameters. Currently, only "prior" is allowed. Defaults to "prior".
            initial_probs: manually specified initial state probabilities. Defaults to None.
            transition_matrix: manually specified transition matrix. Defaults to None.
            emission_probs: manually specified emission probabilities. Defaults to None.

        Returns:
            Model parameters and their properties.
        """
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_probs=emission_probs)
        return ParamsBernoulliHMM(**params), ParamsBernoulliHMM(**props)
