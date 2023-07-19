import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.distributions as tfd
from jaxtyping import Float, Array
from dynamax.parameters import ParameterProperties
from dynamax.hidden_markov_model.models.abstractions import HMM, HMMEmissions, HMMParameterSet, HMMPropertySet
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState, ParamsStandardHMMInitialState
from dynamax.hidden_markov_model.models.transitions import StandardHMMTransitions, ParamsStandardHMMTransitions
from dynamax.types import Scalar
import optax
from typing import NamedTuple, Optional, Tuple, Union


class ParamsCategoricalRegressionHMMEmissions(NamedTuple):
    weights: Union[Float[Array, "state_dim num_classes feature_dim"], ParameterProperties]
    biases: Union[Float[Array, "state_dim num_classes"], ParameterProperties]


class ParamsCategoricalRegressionHMM(NamedTuple):
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsCategoricalRegressionHMMEmissions


class CategoricalRegressionHMMEmissions(HMMEmissions):

    def __init__(self,
                 num_states,
                 num_classes,
                 input_dim,
                 m_step_optimizer=optax.adam(1e-2),
                 m_step_num_iters=50):
        """_summary_

        Args:
            emission_probs (_type_): _description_
        """
        super().__init__(m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        self.num_states = num_states
        self.num_classes = num_classes
        self.feature_dim = input_dim

    @property
    def emission_shape(self):
        return ()

    @property
    def inputs_shape(self):
        return (self.feature_dim,)

    def log_prior(self, params):
        return 0.0

    def initialize(self, key=jr.PRNGKey(0), method="prior", emission_weights=None, emission_biases=None):
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Note: in the future we may support more initialization schemes, like K-Means.

        Args:
            key (PRNGKey, optional): random number generator for unspecified parameters. Must not be None if there are any unspecified parameters. Defaults to jr.PRNGKey(0).
            method (str, optional): method for initializing unspecified parameters. Currently, only "prior" is allowed. Defaults to "prior".
            emission_weights (array, optional): manually specified emission weights. Defaults to None.
            emission_biases (array, optional): manually specified emission biases. Defaults to None.

        Returns:
            params: nested dataclasses of arrays containing model parameters.
            props: a nested dictionary of ParameterProperties to specify parameter constraints and whether or not they should be trained.
        """
        if method.lower() == "prior":
            # technically there's no prior, so just sample standard normals
            key1, key2, key = jr.split(key, 3)
            _emission_weights = jr.normal(key1, (self.num_states, self.num_classes, self.feature_dim))
            _emission_biases = jr.normal(key2, (self.num_states, self.num_classes))

        else:
            raise Exception("Invalid initialization method: {}".format(method))

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0
        params = ParamsCategoricalRegressionHMMEmissions(
            weights=default(emission_weights, _emission_weights),
            biases=default(emission_biases, _emission_biases))
        props = ParamsCategoricalRegressionHMMEmissions(
            weights=ParameterProperties(),
            biases=ParameterProperties())
        return params, props

    def distribution(self, params, state, inputs=None):
        logits = params.weights[state] @ inputs + params.biases[state]
        return tfd.Categorical(logits=logits)


class CategoricalRegressionHMM(HMM):
    r"""An HMM whose emissions come from a categorical regression with state-dependent weights.
    This is also known as a *switching multiclass logistic regression* model.

    Let $y_t \in \{1, \ldots, C\}$ and $u_t \in \mathbb{R}^M$ denote categorical emissions
    and inputs at time $t$, respectively. In this model, the emission distribution is,

    $$p(y_t \mid z_t, u_t, \theta) = \mathrm{Cat}(y_{t} \mid \mathrm{softmax}(W_{z_t} u_t + b_{z_t}))$$

    with *emission weights* $W_k \in \mathbb{R}^{C \times M}$ and *emission biases* $b_k \in \mathbb{R}^C$.

    This model does not have a prior.

    :param num_states: number of discrete states $K$
    :param num_classes: number of emission classes $C$
    :param input_dim: input dimension $M$
    :param initial_probs_concentration: $\alpha$
    :param transition_matrix_concentration: $\beta$
    :param transition_matrix_stickiness: optional hyperparameter to boost the concentration on the diagonal of the transition matrix.
    :param m_step_optimizer: ``optax`` optimizer, like Adam.
    :param m_step_num_iters: number of optimizer steps per M-step.

    """
    def __init__(self,
                 num_states: int,
                 num_classes: int,
                 input_dim: int,
                 initial_probs_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_stickiness: Scalar=0.0,
                 m_step_optimizer: optax.GradientTransformation=optax.adam(1e-2),
                 m_step_num_iters: int=50):
        self.input_dim = input_dim
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, concentration=transition_matrix_concentration, stickiness=transition_matrix_stickiness)
        emission_component = CategoricalRegressionHMMEmissions(num_states, num_classes, input_dim, m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    @property
    def inputs_shape(self):
        return (self.input_dim,)

    def initialize(self,
                   key: jr.PRNGKey=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: Optional[Float[Array, "num_states"]]=None,
                   transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
                   emission_weights: Optional[Float[Array, "num_states num_classes input_dim"]]=None,
                   emission_biases: Optional[Float[Array, "num_states num_classes"]]=None,
        ) -> Tuple[HMMParameterSet, HMMPropertySet]:
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Args:
            key: random number generator for unspecified parameters. Must not be None if there are any unspecified parameters.
            method: method for initializing unspecified parameters. Both "prior" and "kmeans" are supported.
            initial_probs: manually specified initial state probabilities.
            transition_matrix: manually specified transition matrix.
            emission_weights: manually specified emission weights.
            emission_biases: manually specified emission biases.

        Returns:
            Model parameters and their properties.

        """
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key=key3, method=method, emission_weights=emission_weights, emission_biases=emission_biases)
        return ParamsCategoricalRegressionHMM(**params), ParamsCategoricalRegressionHMM(**props)
