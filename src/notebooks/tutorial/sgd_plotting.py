# Imports
import numpy as np
from matplotlib import pyplot as plt
import jax


def plot_learning_curve(
    marginal_lls,
    true_model,
    true_params,
    # TODO: what are these for?
    test_model,
    test_params,
    true_emissions,
    t_emissions,
):
    plt.figure()
    plt.xlabel("iteration")
    nsteps = len(marginal_lls)
    true_logjoint = true_model.log_prior(true_params) + true_model.marginal_log_prob(
        true_params, true_emissions, t_emissions
    )
    plt.axhline(true_logjoint, color="k", linestyle=":", label="true")
    plt.plot(marginal_lls, label="estimated")
    plt.ylabel("marginal joint probability")

    # Adjust y-axis limits
    y_min = min(min(marginal_lls), true_logjoint) * 1.1  # 10% lower than the smallest value
    y_max = max(max(marginal_lls), true_logjoint) * 0.9  # 10% higher than the largest value
    plt.ylim([y_min, y_max])
    plt.yscale("symlog")
    plt.autoscale(enable=True, axis="x", tight=True)
    plt.legend()
