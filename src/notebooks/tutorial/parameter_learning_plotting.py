# Imports
import numpy as np
from matplotlib import pyplot as plt
import jax

## Plotting Utilities

# Plot the marginal log likelihood learning curve
def plot_mll_learning_curve(
    true_model,
    true_params,
    true_emissions,
    t_emissions,
    marginal_lls,
):
    plt.figure()
    plt.xlabel("Iterations")
    true_logjoint = true_model.log_prior(true_params) + true_model.marginal_log_prob(
        true_params, true_emissions, t_emissions
    )
    plt.axhline(
        true_logjoint,
        color="k",
        linestyle=":",
        label="Truth: {}".format(
            np.round(true_logjoint, 2)
        ),
    )
    plt.plot(
        marginal_lls,
        label="Estimated: {}".format(
            np.round(marginal_lls[-1], 2)
        ),
    )
    plt.ylabel("Marginal log joint probability")
    plt.title("Marginal log joint probability over iterations")

    # Adjust y-axis limits
    y_min = min(min(marginal_lls), true_logjoint) * 1.1  # 10% lower than the smallest value
    y_max = max(max(marginal_lls), true_logjoint) * 0.9  # 10% higher than the largest value
    plt.ylim([y_min, y_max])
    plt.yscale("symlog")
    plt.autoscale(enable=True, axis="x", tight=True)
    plt.legend()

# Plot the parameter distributions, given some samples
def plot_param_distributions(
    samples,
    true,
    init,
    name="",
    burn_in_frac=0.5,
    skip_if_not_trainable=True,
    trainable=True,
    triangle_plot=True,
    box_plot=True,
):
    """
    Plots N_params horizontal box plots for the given N_params x N_samples matrix or a triangle plot of bivariate densities.

    Parameters:
    - samples: N_params x N_samples matrix of parameter samples.
    - true: N_params array of true parameter values.
    - init: N_params array of initial estimates.
    - name: Name of the parameter set.
    - burn_in_frac: Fraction of samples to discard as burn-in.
    - skip_if_not_trainable: If True and trainable is True, skip plotting.
    - trainable: Indicates if the parameter is trainable.
    - triangle_plot: If True, plots a triangle plot with bivariate densities and histograms.
    - box_plot: If True, plots box plots for parameter distributions.

    Returns:
    - A matplotlib figure with N_params horizontal box plots or a triangle plot.
    """
    if skip_if_not_trainable and not trainable:
        return

    if trainable:
        name += " (trainable)"

    # apply burn-in
    burn_in = int(burn_in_frac * samples.shape[1])
    samples = samples[:, burn_in:]

    if triangle_plot:
        import seaborn as sns
        import pandas as pd
        import matplotlib.pyplot as plt

        # Create a DataFrame from the samples
        df = pd.DataFrame(samples.T, columns=["Parameter {}".format(i + 1) for i in range(samples.shape[0])])

        # Plot pairplot with histograms on the diagonal
        g = sns.pairplot(df, kind="kde", diag_kind="hist")
        g.fig.suptitle("{} Triangle Plot with Bivariate Densities".format(name), y=1.02)

        # Add Init and ground truth values to the plot
        for i, param in enumerate(df.columns):
            g.axes[i, i].axvline(true[i], color="red", linestyle="--", label="Ground Truth")
            g.axes[i, i].axvline(init[i], color="magenta", linestyle="--", label="Initial Estimate")
            for j in range(i):
                g.axes[i, j].scatter(true[j], true[i], color="red", marker="x", s=100, zorder=4)
                g.axes[i, j].scatter(init[j], init[i], color="magenta", marker="o", s=100, zorder=3)
                g.axes[j, i].set_visible(False)  # Hide the upper right axes
        handles, labels = g.axes[0, 0].get_legend_handles_labels()
        g.fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1, 0.95))  # Add legend to the bottom-left plot
        plt.show()
    if box_plot:
        N_params = samples.shape[0]
        fig, ax = plt.subplots(figsize=(10, N_params * 2))  # Adjust figure size based on number of parameters

        # Create box plots
        ax.boxplot(samples, vert=False, patch_artist=True)

        # Set the y-axis labels to show parameter indices
        ax.set_yticks(range(1, N_params + 1))
        ax.set_yticklabels(["Parameter {}".format(i + 1) for i in range(N_params)])

        # Plot ground truth and estimates
        ax.scatter(
            init, range(1, N_params + 1), color="magenta", marker="o", s=100, label="Initial Estimate", zorder=3
        )
        ax.scatter(true, range(1, N_params + 1), color="red", marker="x", s=100, label="Ground Truth", zorder=4)

        plt.xlabel("Value")
        plt.ylabel("Parameters")
        plt.title("{} Parameter Distributions".format(name))
        plt.grid(True)
        plt.legend()
        plt.show()

# Plot the posterior distributions of all parameters within a CD-NLGSSM model
def plot_all_cdnlgssm_param_posteriors(
        param_samples,
        param_properties,
        init_params,
        true_params,
        burn_in_frac=0.5,
        skip_if_not_trainable=True,
        triangle_plot=True,
        box_plot=True
    ):
    """
    Plots the posterior distributions of all parameters.
    Burn-in is removed from the samples.
    """
    plot_param_distributions(
        param_samples.initial.mean.params.T,
        true_params.initial.mean.params,
        init_params.initial.mean.params,
        name="Initial mean",
        burn_in_frac=burn_in_frac,
        skip_if_not_trainable=skip_if_not_trainable,
        trainable=param_properties.initial.mean.params.trainable,
        triangle_plot=triangle_plot,
        box_plot=box_plot,
    )
    plot_param_distributions(
        param_samples.initial.cov.params.reshape(param_samples.initial.cov.params.shape[0], -1).T,
        true_params.initial.cov.params.flatten(),
        init_params.initial.cov.params.flatten(),
        name="Initial cov",
        burn_in_frac=burn_in_frac,
        skip_if_not_trainable=skip_if_not_trainable,
        trainable=param_properties.initial.cov.params.trainable,
        triangle_plot=triangle_plot,
        box_plot=box_plot,
    )
    plot_param_distributions(
        param_samples.dynamics.drift.params.reshape(param_samples.dynamics.drift.params.shape[0], -1).T,
        true_params.dynamics.drift.params,
        init_params.dynamics.drift.params,
        name="Dynamics drift parameters",
        burn_in_frac=burn_in_frac,
        skip_if_not_trainable=skip_if_not_trainable,
        trainable=param_properties.dynamics.drift.params.trainable,
        triangle_plot=triangle_plot,
        box_plot=box_plot,
    )
    plot_param_distributions(
        param_samples.dynamics.diffusion_cov.params.reshape(param_samples.dynamics.diffusion_cov.params.shape[0], -1).T,
        true_params.dynamics.diffusion_cov.params.flatten(),
        init_params.dynamics.diffusion_cov.params.flatten(),
        name="Dynamics diffusion cov",
        burn_in_frac=burn_in_frac,
        skip_if_not_trainable=skip_if_not_trainable,
        trainable=param_properties.dynamics.diffusion_cov.params.trainable,
        triangle_plot=triangle_plot,
        box_plot=box_plot,
    )
    plot_param_distributions(
        param_samples.dynamics.diffusion_coefficient.params.reshape(
            param_samples.dynamics.diffusion_coefficient.params.shape[0], -1
        ).T,
        true_params.dynamics.diffusion_coefficient.params.flatten(),
        init_params.dynamics.diffusion_coefficient.params.flatten(),
        name="Dynamics diffusion coefficient",
        burn_in_frac=burn_in_frac,
        skip_if_not_trainable=skip_if_not_trainable,
        trainable=param_properties.dynamics.diffusion_coefficient.params.trainable,
        triangle_plot=triangle_plot,
        box_plot=box_plot,
    )
    plot_param_distributions(
        param_samples.emissions.emission_function.weights.reshape(
            param_samples.emissions.emission_function.weights.shape[0], -1
        ).T,
        true_params.emissions.emission_function.weights.flatten(),
        init_params.emissions.emission_function.weights.flatten(),
        name="Emissions function weights",
        burn_in_frac=burn_in_frac,
        skip_if_not_trainable=skip_if_not_trainable,
        trainable=param_properties.emissions.emission_function.weights.trainable,
        triangle_plot=triangle_plot,
        box_plot=box_plot,
    )
    plot_param_distributions(
        param_samples.emissions.emission_function.bias.reshape(
            param_samples.emissions.emission_function.bias.shape[0], -1
        ).T,
        true_params.emissions.emission_function.bias.flatten(),
        init_params.emissions.emission_function.bias.flatten(),
        name="Emissions function bias",
        burn_in_frac=burn_in_frac,
        skip_if_not_trainable=skip_if_not_trainable,
        trainable=param_properties.emissions.emission_function.bias.trainable,
        triangle_plot=triangle_plot,
        box_plot=box_plot,
    )
    plot_param_distributions(
        param_samples.emissions.emission_cov.params.reshape(param_samples.emissions.emission_cov.params.shape[0], -1).T,
        true_params.emissions.emission_cov.params.flatten(),
        init_params.emissions.emission_cov.params.flatten(),
        name="Emissions cov",
        burn_in_frac=burn_in_frac,
        skip_if_not_trainable=skip_if_not_trainable,
        trainable=param_properties.emissions.emission_cov.params.trainable,
        triangle_plot=triangle_plot,
        box_plot=box_plot,
    )
