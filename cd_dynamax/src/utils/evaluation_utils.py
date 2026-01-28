# This file contains utility functions for evaluating the performance of cd dynamax models

# Imports
import jax.numpy as jnp


### Evaluation
# Compute RMSE
def compute_rmse(y, y_est):
    return jnp.sqrt(jnp.sum((y - y_est) ** 2) / len(y))


# Compute RMSE of estimate and print comparison with
# standard deviation of measurement noise
def compute_and_print_rmse_comparison(y, y_est, R, est_type=""):
    rmse_est = compute_rmse(y, y_est)
    print(f"{f'The RMSE of the {est_type} estimate is':<40}: {rmse_est:.2f}")
    print(f"{'The std of measurement noise is':<40}: {jnp.sqrt(R):.2f}")
