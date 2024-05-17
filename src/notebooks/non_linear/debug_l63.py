import sys
sys.path.append("../../..")
sys.path.append("../..")


from jax import numpy as jnp
from jax import vmap
import jax.random as jr
from jax import jit, vmap
from jax import lax, value_and_grad, tree_map

from typing import NamedTuple, Tuple, Optional, Union
from flax import linen as nn
from itertools import count
from dynamax.parameters import to_unconstrained, from_unconstrained, log_det_jac_constrain
from dynamax.utils.utils import ensure_array_has_batch_dim, pytree_stack

# use custom src codebase
from utils.plotting_utils import *

# from utils.utils import monotonically_increasing
from continuous_discrete_nonlinear_gaussian_ssm import ContDiscreteNonlinearGaussianSSM
from continuous_discrete_nonlinear_gaussian_ssm.models import *


import dynamax
from dynamax.parameters import log_det_jac_constrain

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = [16, 9]

state_dim = 3
emission_dim = 1
num_sequences = 1000
T = 1
num_timesteps = int(T / 0.01)
t_emissions = jnp.array(sorted(jr.uniform(jr.PRNGKey(0), (num_timesteps, 1), minval=0, maxval=T)))
# drop duplicates
t_emissions = jnp.unique(t_emissions)[:,None]
num_timesteps = len(t_emissions)
keys = map(jr.PRNGKey, count())

# Create a model with oscillatory dynamics
## GOAL is to only learn parameters of the drift function and assume all other parameters are known.


dynamics_drift = {
    "params": LearnableLorenz63(sigma=10.0, rho=28.0, beta=8/3),
    "props": LearnableLorenz63(sigma=ParameterProperties(),
                               rho=ParameterProperties(),
                               beta=ParameterProperties()) 
}

dynamics_diffusion_coefficient = {
    "params": LearnableMatrix(params=jnp.eye(state_dim)),
    "props": LearnableMatrix(params=ParameterProperties(trainable=False))
}

dynamics_diffusion_cov = {
    "params": LearnableMatrix(params=jnp.eye(state_dim)),
    "props": LearnableMatrix(params=ParameterProperties(constrainer=RealToPSDBijector(), trainable=False))
}

emission_function = {
    "params": LearnableLinear(weights=jnp.eye(emission_dim, state_dim), bias=jnp.zeros(emission_dim)),
    "props": LearnableLinear(weights=ParameterProperties(trainable=False), bias=ParameterProperties(trainable=False))
}

emission_cov = {
    "params": LearnableMatrix(params=jnp.eye(emission_dim)),
    "props": LearnableMatrix(params=ParameterProperties(constrainer=RealToPSDBijector(), trainable=False))
}

initial_mean = {"params": jnp.zeros(state_dim),
                "props": ParameterProperties(trainable=False)}

initial_cov = {"params": 100*jnp.eye(state_dim),
                "props": ParameterProperties(trainable=False, constrainer=RealToPSDBijector())}


true_model = ContDiscreteNonlinearGaussianSSM(state_dim, emission_dim)
true_params, true_props = true_model.initialize(
    next(keys),
    initial_mean=initial_mean,
    initial_cov=initial_cov,
    dynamics_drift=dynamics_drift,
    dynamics_diffusion_coefficient=dynamics_diffusion_coefficient,
    dynamics_diffusion_cov=dynamics_diffusion_cov,
    emission_function=emission_function,
    emission_cov=emission_cov,
)

# sample true states and emissions. Using transition_type="path" to solve SDE.
true_states, emissions = true_model.sample_batch(
    true_params, next(keys), num_sequences, num_timesteps, t_emissions, transition_type="path"
)

# compute the log likelihood of the true model
## WARNING: only computing it for the first sequence
# this is to avoid batching for now...
for state_order in ["zeroth", "first", "second"]:
    print(f"Computing log likelihood for {state_order} order EKF.")
    filter_hyperparams = EKFHyperParams(state_order=state_order)

    first_emissions = emissions[0]
    ll_true = true_model.marginal_log_prob(
        params=true_params, filter_hyperparams=filter_hyperparams, emissions=first_emissions, t_emissions=t_emissions
    )
    print(f"Log likelihood of true model (approximated by EKF): {-ll_true}")


# ## Create a class for a learnable neural network, which we will use to parameterize the drift function

# define a normalizer class which stores the mean and standard deviation of the data
# it will have a method to normalize the data and another to denormalize it
class Normalizer(NamedTuple):
    mean: jnp.ndarray
    std: jnp.ndarray

    def normalize(self, data):
        return (data - self.mean) / self.std

    def denormalize(self, data):
        return data * self.std + self.mean

print("Warning: cheating by using the true states mean and std to normalize the states.")
my_normalizer = Normalizer(mean=true_states.mean(axis=(0,1)), std=true_states.std(axis=(0,1)))

@jit
def adjust_rhs(x, rhs, lower_bound=-100, upper_bound=100, espilon=1e-10):
    """
    Adjust the right-hand side of the ODE to ensure that the state 
    remains within the bounds [-100, 100]
    """

    ## NB: Can use jax.lax.clamp to do this more efficiently
    

    # Use a small epsilon to ensure numerical stability
    # Smoothly adjust the bounds to avoid gradient discontinuities
    # safe_lower_bound = jnp.where(x <= lower_bound, lower_bound + epsilon, x)
    # safe_upper_bound = jnp.where(x >= upper_bound, upper_bound - epsilon, x)

    # Conditionally adjust rhs using the safe bounds
    # rhs = jnp.where(safe_lower_bound <= lower_bound, jnp.maximum(rhs, 0), rhs)
    # rhs = jnp.where(safe_upper_bound >= upper_bound, jnp.minimum(rhs, 0), rhs)

    # adjust_min = x <= lower_bound
    # adjust_max = x >= upper_bound
    # rhs = jnp.where(adjust_min, jnp.maximum(rhs, 0), rhs)
    # rhs = jnp.where(adjust_max, jnp.minimum(rhs, 0), rhs)
    return rhs


class LearnableNN_TwoLayerGeLU(NamedTuple):
    """Two-layer neural network with Gaussian Error Linear Units
    weights1: weights of the first layer
    bias1: bias of the first layer
    weights2: weights of the second layer
    bias2: bias of the second layer

    f(x) = weights2 @ gelu(weights1 @ x + bias1) + bias2
    """

    weights1: Union[Float[Array, "hidden_dim input_dim"], ParameterProperties]
    bias1: Union[Float[Array, "hidden_dim"], ParameterProperties]
    weights2: Union[Float[Array, "output_dim hidden_dim"], ParameterProperties]
    bias2: Union[Float[Array, "output_dim"], ParameterProperties]
    scale: Union[Float, ParameterProperties]

    def f(self, x, u=None, t=None):
        '''This rhs operates in original space, so we need to normalize the input x first.'''

        # # first, clamp all x components to be within [-100, 100]
        # x = jnp.clip(x, -100, 100)
        x_normalized = my_normalizer.normalize(x)

        # compute derivative given by NN
        rhs = self.weights2 @ nn.gelu(self.weights1 @ x_normalized + self.bias1) + self.bias2

        # un-normalize the rhs and multiply by 10^scale
        rhs = rhs * my_normalizer.std * jnp.power(10, self.scale)

        return adjust_rhs(x, rhs)

# ## Define bad parameter set
new_params = ParamsCDNLGSSM(
    initial=ParamsLGSSMInitial(
        mean=jnp.array([0.0, 0.0, 0.0]),
        cov=jnp.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]),
    ),
    dynamics=ParamsCDNLGSSMDynamics(
        drift=LearnableNN_TwoLayerGeLU(
            weights1=jnp.array(
                [
                    [-0.09618049, -0.51569897, 0.24136686],
                    [0.9450796, 1.3519114, 0.19802144],
                    [-0.05342332, -0.14780462, -0.7245296],
                    [-0.36849234, 0.62802565, 0.20715408],
                    [-0.49130943, -1.0126867, -1.3491994],
                    [0.8521092, -0.78537834, 0.7155881],
                    [-0.76622623, -0.12774178, -1.01384],
                    [-0.4277388, 0.57263494, -0.55070764],
                    [1.9272785, 0.2300885, -0.35206327],
                    [0.02106795, -1.5295459, -0.8156675],
                    [0.57002157, 1.6865013, -0.02586436],
                    [-0.83643544, -0.6072524, 0.6926793],
                    [0.30118793, 0.647623, 0.7569656],
                    [-0.52852356, -0.18216379, 0.26737297],
                    [1.8086166, -0.38924438, 1.281889],
                    [0.99501103, 0.00269605, 0.16720587],
                    [-0.02648759, 0.8370276, 0.39732492],
                    [-0.33850017, -0.5173549, 0.6109256],
                    [-0.69618744, 0.88613963, 0.13466708],
                    [-0.78111035, 0.40116292, 0.25791332],
                    [0.9166225, 0.8639344, 0.6593652],
                    [-0.75124484, 0.54718304, 0.4500338],
                    [-0.99765694, -0.31343877, 1.2181777],
                    [-0.23146433, 0.4152232, -0.864154],
                    [-0.362102, -0.32531476, 1.0601325],
                    [-0.01614203, -0.8401461, 0.9883657],
                    [-0.3660067, -0.924267, 0.5297535],
                    [-0.16821717, -1.0970813, -0.9427991],
                    [0.4142191, -0.3386449, -1.0180024],
                    [-0.650903, -0.03522696, -1.6251227],
                    [0.02817957, -1.4134698, 0.6566164],
                    [-1.4348187, -0.6617468, 0.9970957],
                    [1.3310343, -0.22725026, -1.0269009],
                    [-0.9511327, 0.4886929, 0.12323312],
                    [1.3149629, 0.08380658, 0.1452246],
                    [-0.72148377, 0.37688267, -1.3649459],
                    [-0.04463482, -0.22438252, -0.6143299],
                    [0.9736543, 0.6474263, -0.16621913],
                    [0.70256996, -0.7736834, 0.5235938],
                    [0.97892535, 0.08240159, 0.25392073],
                    [0.16757911, -0.6722603, -1.6465843],
                    [0.30732825, -0.05132433, 0.857229],
                    [0.17216274, 0.58979577, 0.7237272],
                    [-0.8716645, 0.76828384, 1.5337913],
                    [-0.3449903, -0.65782964, 0.79576266],
                    [-0.10955516, 0.32162288, -0.54427487],
                    [-0.54340225, 1.9441457, 1.683218],
                    [-0.03756475, -0.4139751, 0.86002994],
                    [0.3870767, -1.225448, 0.69391006],
                    [0.42021775, -0.50241125, -0.8686773],
                ],
            ),
            bias1=jnp.array(
                [
                    -0.4911987,
                    0.45916802,
                    -0.09253469,
                    -0.1301824,
                    -0.15691942,
                    -0.29079548,
                    -0.37290585,
                    -0.04042498,
                    0.4659113,
                    0.36804974,
                    0.19003159,
                    0.21274355,
                    -0.56768054,
                    -0.43781382,
                    0.50611204,
                    -0.02147902,
                    -0.38756862,
                    0.09218834,
                    0.46690747,
                    -0.30948746,
                    -0.12051558,
                    -0.30601153,
                    -0.2781745,
                    0.60438234,
                    -0.25925636,
                    -0.18054532,
                    0.02130086,
                    0.34136146,
                    0.03142443,
                    -0.07919644,
                    -0.36738414,
                    -0.25737956,
                    -0.2025466,
                    -0.42205328,
                    0.30250922,
                    0.08632556,
                    0.29119304,
                    0.02290984,
                    -0.10311417,
                    0.28066704,
                    0.16911839,
                    0.29324874,
                    -0.4608032,
                    -0.33346748,
                    -0.24925442,
                    -0.3671042,
                    -0.5419366,
                    -0.38385752,
                    -0.00609869,
                    0.05823664,
                ],
            ),
            weights2=jnp.array(
                [
                    [
                        -0.36993665,
                        0.11541048,
                        0.38307217,
                        0.48211232,
                        0.45650345,
                        -0.890558,
                        0.13204156,
                        0.06995704,
                        -0.10877819,
                        0.6890369,
                        0.16016759,
                        -0.71713686,
                        0.33091098,
                        -0.04646308,
                        -0.52129096,
                        -0.36356634,
                        0.16591297,
                        -0.33467025,
                        -0.30493695,
                        0.05359869,
                        0.50149375,
                        0.30816406,
                        -0.15791675,
                        -0.09652498,
                        -0.23420116,
                        -0.17018546,
                        -0.485532,
                        0.4052401,
                        0.77575344,
                        0.19252874,
                        -0.07894012,
                        -0.2194967,
                        0.09211113,
                        0.18383104,
                        -0.09754829,
                        -0.00180552,
                        0.31173855,
                        -0.45646724,
                        -0.4829965,
                        -0.74340665,
                        0.7452059,
                        -0.23047309,
                        0.35947558,
                        0.41634345,
                        0.12020309,
                        -0.20971236,
                        0.49231744,
                        -0.24428444,
                        -0.50080967,
                        0.76692957,
                    ],
                    [
                        0.22760816,
                        -0.80147594,
                        0.35056633,
                        -0.15119132,
                        0.27670926,
                        0.37824708,
                        0.21244119,
                        -0.21309087,
                        -0.44221494,
                        0.31324536,
                        -0.5124975,
                        0.26644516,
                        -0.11724719,
                        0.6349943,
                        0.09138875,
                        0.15301862,
                        -0.21332796,
                        0.40317836,
                        0.12951411,
                        0.20515616,
                        -0.87896484,
                        0.5936799,
                        0.5835366,
                        -0.3451974,
                        0.3856366,
                        0.52980787,
                        0.49151063,
                        0.17267054,
                        0.38694176,
                        0.10625781,
                        0.20629553,
                        0.3011698,
                        0.09076123,
                        0.11131746,
                        -0.29279873,
                        -0.08796256,
                        0.1569084,
                        -0.45298752,
                        0.15043056,
                        -0.39114738,
                        0.2286047,
                        0.47483107,
                        -0.3926533,
                        0.42057666,
                        0.66156936,
                        0.1430995,
                        0.27697626,
                        0.3969916,
                        0.22285038,
                        0.14430788,
                    ],
                    [
                        0.35392624,
                        -0.7695391,
                        -0.41437694,
                        -0.35927558,
                        -0.49264586,
                        0.49925268,
                        -0.2657759,
                        -0.6061649,
                        0.0418812,
                        0.23572065,
                        -0.6032816,
                        -0.13204005,
                        -0.2234995,
                        -0.1371735,
                        0.26169276,
                        0.557582,
                        -0.7434315,
                        -0.10799786,
                        0.15583253,
                        0.16887103,
                        -0.49089,
                        0.12605475,
                        -0.03148818,
                        -0.2289765,
                        -0.42078897,
                        0.39111412,
                        0.39623457,
                        0.09845531,
                        -0.4530118,
                        -1.0744587,
                        0.22699036,
                        -0.16500154,
                        -0.5089227,
                        0.1656643,
                        0.2414226,
                        -0.54656476,
                        -0.09216134,
                        -0.3951335,
                        0.32760996,
                        0.5137172,
                        -0.46826428,
                        -0.14056414,
                        -0.6498051,
                        -0.41157344,
                        -0.00212859,
                        -0.48042625,
                        -0.55699813,
                        0.22114234,
                        0.14584987,
                        -0.5378949,
                    ],
                ],
            ),
            bias2=jnp.array([0.08113164, -0.31454903, 0.12585898]),
            scale=jnp.array(0.11300036),
        ),
        diffusion_coefficient=LearnableMatrix(
            params=jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], )
        ),
        diffusion_cov=LearnableMatrix(params=jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], )),
        approx_order=jnp.array(2.0),
    ),
    emissions=ParamsCDNLGSSMEmissions(
        emission_function=LearnableLinear(
            weights=jnp.array([[1.0, 0.0, 0.0]]), bias=jnp.array([0.0])
        ),
        emission_cov=LearnableMatrix(params=jnp.array([[1.0]])),
    ),
)

# ## Compute gradients of loss @ bad parameter set
test_model = ContDiscreteNonlinearGaussianSSM(state_dim, emission_dim)


## Now do the same but only for a single sequence of emissions
em_ind = 2
batch_inputs = ensure_array_has_batch_dim(None, test_model.inputs_shape)
batch_emissions = ensure_array_has_batch_dim(emissions[em_ind], test_model.emission_shape)
# batch_t_emissions = ensure_array_has_batch_dim(t_emissions, (1,))
batch_t_emissions = jnp.repeat(t_emissions[jnp.newaxis, :, :], batch_emissions.shape[0], axis=0)


for state_order in ["zeroth", "first", "second"]:
    print(f"Computing log likelihood and its gradients for {state_order} order EKF on single emission sequence.")
    filter_hyperparams = EKFHyperParams(state_order=state_order)

    def _new_loss_fn(my_params):
        batch_lls = vmap(
            partial(test_model.marginal_log_prob, my_params, filter_hyperparams=filter_hyperparams),
        )(emissions=batch_emissions, t_emissions=batch_t_emissions, inputs=batch_inputs)
        lp = test_model.log_prior(my_params) + batch_lls.sum()
        return -lp / len(batch_emissions)
    loss_grad_fn = value_and_grad(_new_loss_fn)
    ## USING THE MANUALLY LOADED PARAMETERS
    this_loss, grads = loss_grad_fn(new_params)
    print(this_loss)
    print(jnp.max(jnp.abs(grads.dynamics.drift.bias2)))
    print(jnp.max(jnp.abs(grads.dynamics.drift.bias1)))
    print(jnp.max(jnp.abs(grads.dynamics.drift.weights1)))
    print(jnp.max(jnp.abs(grads.dynamics.drift.weights2)))
