import jax.numpy as jnp

from dynamax.nonlinear_gaussian_ssm.unscented_kalman_filter import unscented_kalman_smoother, UKFHyperParams
from dynamax.nonlinear_gaussian_ssm.sarkka_lib import ukf, uks
from dynamax.nonlinear_gaussian_ssm.inference_test import random_nlgssm_args


from dynamax.utils import has_tpu

if has_tpu():
    def allclose(x, y):
        print(jnp.max(x-y))
        return jnp.allclose(x, y, atol=1e-1)
else:
    def allclose(x,y):
        return jnp.allclose(x, y, atol=1e-1)

def test_ukf_nonlinear(key=0, num_timesteps=15):
    nlgssm_args, _, emissions = random_nlgssm_args(key=key, num_timesteps=num_timesteps)
    hyperparams = UKFHyperParams()

    # Run UKF from sarkka-jax library
    means_ukf, covs_ukf = ukf(*nlgssm_args, *hyperparams, emissions)
    # Run UKS from sarkka-jax library
    means_uks, covs_uks = uks(*nlgssm_args, *hyperparams, emissions)
    # Run UKS from dynamax
    uks_post = unscented_kalman_smoother(nlgssm_args, emissions, hyperparams)

    # Compare filter results
    assert allclose(means_ukf, uks_post.filtered_means)
    assert allclose(covs_ukf, uks_post.filtered_covariances)
    assert allclose(means_uks, uks_post.smoothed_means)
    assert allclose(covs_uks, uks_post.smoothed_covariances)