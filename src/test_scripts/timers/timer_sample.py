import sys
sys.path.append("../")
sys.path.append("../..")

import dynamax

from jax import numpy as jnp
from jax import jit, vmap
import jax.random as jr
from matplotlib import pyplot as plt

import argparse


# use custom src codebase
from utils.plotting_utils import *

# from utils.utils import monotonically_increasing
from continuous_discrete_nonlinear_gaussian_ssm import ContDiscreteNonlinearGaussianSSM
from continuous_discrete_nonlinear_gaussian_ssm.models import *

from itertools import count

from dynamax.parameters import log_det_jac_constrain

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = [16, 9]


# %%
# Simulate synthetic data from true model
state_dim = 3
emission_dim = 1
keys = map(jr.PRNGKey, count())


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
true_params, _ = true_model.initialize(
    next(keys),
    initial_mean=initial_mean,
    initial_cov=initial_cov,
    dynamics_drift=dynamics_drift,
    dynamics_diffusion_coefficient=dynamics_diffusion_coefficient,
    dynamics_diffusion_cov=dynamics_diffusion_cov,
    emission_function=emission_function,
    emission_cov=emission_cov,
)

# Setup the parser
parser = argparse.ArgumentParser(description="Run model simulation")
parser.add_argument('-T', '--total_time', default=3, type=float, help="Total simulation time")
parser.add_argument('-N', '--num_sequences', default=10, type=int, help="Number of sequences to simulate")
parser.add_argument('-i', '--num_iterations', default=10, type=int, help="Number of iterations to run")
args = parser.parse_args()

num_sequences = args.num_sequences
T = args.total_time
num_timesteps = int(T / 0.01)
t_emissions = jnp.array(sorted(jr.uniform(jr.PRNGKey(0), (num_timesteps, 1), minval=0, maxval=T)))
# drop duplicates
t_emissions = jnp.unique(t_emissions)[:, None]
num_timesteps = len(t_emissions)

def run_sample():
    true_states, emissions = true_model.sample_batch(
        true_params, next(keys), num_sequences, num_timesteps, t_emissions, transition_type="path"
    )

import timeit
# Time the execution
execution_time = timeit.timeit(run_sample, number=args.num_iterations)
avg_execution_time = execution_time / args.num_iterations
print(f"Average execution time: {avg_execution_time} seconds")
