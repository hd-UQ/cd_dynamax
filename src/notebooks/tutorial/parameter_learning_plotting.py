# Imports
import numpy as np
from matplotlib import pyplot as plt
import jax
import seaborn as sns
import pandas as pd

## Plotting Utilities


# Plot the marginal log likelihood learning curve
def plot_mll_learning_curve(
    true_model,
    true_params,
    true_emissions,
    t_emissions,
    marginal_lls,
):
    """Note that the true logjoint is computed using default filter hyperparameters in marginal_log_prob."""

    plt.figure()
    plt.xlabel("Iterations")
    true_logjoint = true_model.log_prior(true_params) + true_model.marginal_log_prob(
        true_params, true_emissions, t_emissions
    )
    plt.axhline(
        true_logjoint,
        color="k",
        linestyle=":",
        label="Truth: {}".format(np.round(true_logjoint, 2)),
    )
    plt.plot(
        marginal_lls,
        label="Estimated: {}".format(np.round(marginal_lls[-1], 2)),
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
    samples=None,
    true=None,
    init=None,
    pointwise_estimate=None,
    name="",
    burn_in_frac=0.5,
    trainable=True,
    triangle_plot=True,
    triangle_traj_plot=True,
    box_plot=True,
    sequence_plot=True,
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
    - triangle_traj_plot: If True, plots a triangle plot with parameter trajectories.
    - box_plot: If True, plots box plots for parameter distributions.
    - sequence_plot: If True, plots the parameter values over time/iterations.

    Returns:
    - A matplotlib figure with N_params horizontal box plots or a triangle plot.
    """
    if trainable:
        name += " (trainable)"

    # apply burn-in
    if samples is not None:
        burn_in = int(burn_in_frac * samples.shape[1])
        samples = samples[:, burn_in:]
    else:
        box_plot = True
        triangle_plot = False
        triangle_traj_plot = False

    if samples is None:
        if true is None:
            N_params = true.shape[0]
        else:
            N_params = init.shape[0]
    else:
        N_params = samples.shape[0]

    if triangle_plot:

        # Create a DataFrame from the samples
        df = pd.DataFrame(samples.T, columns=["Parameter {}".format(i + 1) for i in range(samples.shape[0])])

        # Plot pairplot with histograms on the diagonal
        g = sns.pairplot(df, kind="kde", diag_kind="hist")
        g.fig.suptitle("{} Triangle Plot with Bivariate Densities".format(name), y=1.02)

        # Add Init and ground truth values to the plot
        for i, param in enumerate(df.columns):
            if true is not None:
                g.axes[i, i].axvline(true[i], color="red", linestyle="--", label="Ground Truth")
            if init is not None:
                g.axes[i, i].axvline(init[i], color="magenta", linestyle="--", label="Initial Estimate")
            if pointwise_estimate is not None:
                g.axes[i, i].axvline(pointwise_estimate[i], color="orange", linestyle="--", label="Pointwise Estimate")
            for j in range(i):
                if true is not None:
                    g.axes[i, j].scatter(true[j], true[i], color="red", marker="x", s=100, zorder=4)
                if init is not None:
                    g.axes[i, j].scatter(init[j], init[i], color="magenta", marker="o", s=100, zorder=3)
                g.axes[j, i].set_visible(False)  # Hide the upper right axes
        handles, labels = g.axes[0, 0].get_legend_handles_labels()
        g.fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1, 0.95))  # Add legend to the bottom-left plot
        plt.show()

    if triangle_traj_plot:
        # Create a DataFrame from the samples
        df = pd.DataFrame(samples.T, columns=["Parameter {}".format(i + 1) for i in range(samples.shape[0])])

        # Create PairGrid for custom plotting, excluding diagonal and upper right subplots
        g = sns.PairGrid(df, corner=True, diag_sharey=False)

        # Plot scatter plots in the lower triangle subplots with color gradient from magenta to blue
        def scatter_with_gradient(x, y, **kwargs):
            colors = np.linspace(0, 1, len(x))
            cmap = sns.color_palette("cool", as_cmap=True)
            plt.scatter(x, y, c=colors, cmap=cmap, **{k: v for k, v in kwargs.items() if k != "color"})

        g.map_lower(scatter_with_gradient, s=10, zorder=2)

        g.fig.suptitle("{} Trajectory Plot".format(name), y=1.02)

        # Add Init and ground truth values to the plot
        for i, param in enumerate(df.columns):
            for j in range(i):
                # Plot ground truth and initial estimate as points
                if true is not None:
                    g.axes[i, j].scatter(
                        true[j], true[i], color="red", marker="x", s=100, zorder=4, label="Ground Truth"
                    )
                if init is not None:
                    g.axes[i, j].scatter(
                        init[j], init[i], color="magenta", marker="o", s=100, zorder=3, label="Initial Estimate"
                    )
                if pointwise_estimate is not None:
                    g.axes[i, j].scatter(
                        pointwise_estimate[j],
                        pointwise_estimate[i],
                        color="orange",
                        marker="*",
                        s=100,
                        zorder=3,
                        label="Pointwise Estimate",
                    )

        # Remove duplicate legend labels by maintaining a set of seen labels and add legend only once
        handles, labels = [], []
        seen = set()
        for ax in g.axes.flat:
            if ax is not None:
                h, l = ax.get_legend_handles_labels()
                for handle, label in zip(h, l):
                    if label not in seen and label != "":
                        seen.add(label)
                        handles.append(handle)
                        labels.append(label)
        if handles:
            g.fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1, 0.95))  # Add legend to the figure

        plt.show()
    if box_plot:
        fig, ax = plt.subplots(figsize=(10, N_params * 2))  # Adjust figure size based on number of parameters

        # Create box plots
        if samples is not None:
            ax.boxplot(samples, vert=False, patch_artist=True)

        # Plot ground truth and estimates
        if init is not None:
            ax.scatter(
                init, range(1, N_params + 1), color="magenta", marker="o", s=100, label="Initial Estimate", zorder=3
            )
        if true is not None:
            ax.scatter(true, range(1, N_params + 1), color="red", marker="x", s=100, label="Ground Truth", zorder=4)
        if pointwise_estimate is not None:
            ax.scatter(
                pointwise_estimate,
                range(1, N_params + 1),
                color="orange",
                marker="o",
                s=100,
                label="Pointwise Estimate",
                zorder=3,
            )

        # Set the y-axis labels to show parameter indices
        ax.set_yticks(range(1, N_params + 1))
        ax.set_yticklabels(["Parameter {}".format(i + 1) for i in range(N_params)])

        plt.xlabel("Value")
        plt.ylabel("Parameters")
        plt.title("{} Parameter Distributions".format(name))
        plt.grid(True)
        plt.legend()
        plt.show()

    if sequence_plot:
        # Plot the parameter values over time/iterations
        fig, axes = plt.subplots(
            N_params, 1, figsize=(10, N_params * 2), sharex=True
        )  # Create subplots for each parameter
        for i in range(N_params):
            if true is not None:
                axes[i].axhline(true[i], color="k", linestyle="--", label="Ground Truth")
            if pointwise_estimate is not None:
                axes[i].axhline(pointwise_estimate[i], color="C0", linestyle="--", label="Pointwise Estimate")
            if samples is not None:
                axes[i].plot(samples[i], color="C0", label="Parameter {}".format(i + 1))
            if init is not None:
                axes[i].axhline(init[i], color="magenta", linestyle="--", label="Initial Estimate")
            axes[i].set_ylabel("Value")
            axes[i].set_title("Parameter {}".format(i + 1))
            axes[i].grid(True)
            axes[i].legend()
        axes[-1].set_xlabel("Iterations")
        plt.suptitle("{} Parameter Values over Iterations".format(name))
        plt.show()


# Plot the posterior distributions of all parameters within a CD-NLGSSM model
def plot_all_cdnlgssm_param_posteriors(
    param_samples=None,
    param_properties=None,
    init_params=None,
    true_params=None,
    pointwise_estimate=None,
    burn_in_frac=0.5,
    skip_if_not_trainable=True,
    triangle_plot=True,
    box_plot=True,
    triangle_traj_plot=True,
    sequence_plot=True,
):
    """
    Plots the posterior distributions of all parameters.
    Burn-in is removed from the samples.
    """
    if param_properties.initial.mean.params.trainable or not skip_if_not_trainable:
        plot_param_distributions(
            param_samples.initial.mean.params.T if param_samples is not None else None,
            true_params.initial.mean.params if true_params is not None else None,
            init_params.initial.mean.params if init_params is not None else None,
            pointwise_estimate=pointwise_estimate.initial.mean.params if pointwise_estimate is not None else None,
            name="Initial mean",
            trainable=param_properties.initial.mean.params.trainable,
            burn_in_frac=burn_in_frac,
            triangle_plot=triangle_plot,
            box_plot=box_plot,
            triangle_traj_plot=triangle_traj_plot,
            sequence_plot=sequence_plot,
        )

    if param_properties.initial.cov.params.trainable or not skip_if_not_trainable:
        plot_param_distributions(
            (
                param_samples.initial.cov.params.reshape(param_samples.initial.cov.params.shape[0], -1).T
                if param_samples is not None
                else None
            ),
            true_params.initial.cov.params.flatten() if true_params is not None else None,
            init_params.initial.cov.params.flatten() if init_params is not None else None,
            pointwise_estimate=(
                pointwise_estimate.initial.cov.params.flatten() if pointwise_estimate is not None else None
            ),
            name="Initial cov",
            trainable=param_properties.initial.cov.params.trainable,
            burn_in_frac=burn_in_frac,
            triangle_plot=triangle_plot,
            box_plot=box_plot,
            triangle_traj_plot=triangle_traj_plot,
            sequence_plot=sequence_plot,
        )

    if param_properties.dynamics.drift.params.trainable or not skip_if_not_trainable:
        plot_param_distributions(
            (
                param_samples.dynamics.drift.params.reshape(param_samples.dynamics.drift.params.shape[0], -1).T
                if param_samples is not None
                else None
            ),
            true_params.dynamics.drift.params if true_params is not None else None,
            init_params.dynamics.drift.params if init_params is not None else None,
            pointwise_estimate=pointwise_estimate.dynamics.drift.params if pointwise_estimate is not None else None,
            name="Dynamics drift parameters",
            trainable=param_properties.dynamics.drift.params.trainable,
            burn_in_frac=burn_in_frac,
            triangle_plot=triangle_plot,
            box_plot=box_plot,
            triangle_traj_plot=triangle_traj_plot,
            sequence_plot=sequence_plot,
        )

    if param_properties.dynamics.diffusion_cov.params.trainable or not skip_if_not_trainable:
        plot_param_distributions(
            (
                param_samples.dynamics.diffusion_cov.params.reshape(
                    param_samples.dynamics.diffusion_cov.params.shape[0], -1
                ).T
                if param_samples is not None
                else None
            ),
            true_params.dynamics.diffusion_cov.params.flatten() if true_params is not None else None,
            init_params.dynamics.diffusion_cov.params.flatten() if init_params is not None else None,
            pointwise_estimate=(
                pointwise_estimate.dynamics.diffusion_cov.params.flatten() if pointwise_estimate is not None else None
            ),
            name="Dynamics diffusion cov",
            trainable=param_properties.dynamics.diffusion_cov.params.trainable,
            burn_in_frac=burn_in_frac,
            triangle_plot=triangle_plot,
            box_plot=box_plot,
            triangle_traj_plot=triangle_traj_plot,
            sequence_plot=sequence_plot,
        )

    if param_properties.dynamics.diffusion_coefficient.params.trainable or not skip_if_not_trainable:
        plot_param_distributions(
            (
                param_samples.dynamics.diffusion_coefficient.params.reshape(
                    param_samples.dynamics.diffusion_coefficient.params.shape[0], -1
                ).T
                if param_samples is not None
                else None
            ),
            true_params.dynamics.diffusion_coefficient.params.flatten() if true_params is not None else None,
            init_params.dynamics.diffusion_coefficient.params.flatten() if init_params is not None else None,
            pointwise_estimate=(
                pointwise_estimate.dynamics.diffusion_coefficient.params.flatten()
                if pointwise_estimate is not None
                else None
            ),
            name="Dynamics diffusion coefficient",
            trainable=param_properties.dynamics.diffusion_coefficient.params.trainable,
            burn_in_frac=burn_in_frac,
            triangle_plot=triangle_plot,
            box_plot=box_plot,
            triangle_traj_plot=triangle_traj_plot,
            sequence_plot=sequence_plot,
        )

    if param_properties.emissions.emission_function.weights.trainable or not skip_if_not_trainable:
        plot_param_distributions(
            (
                param_samples.emissions.emission_function.weights.reshape(
                    param_samples.emissions.emission_function.weights.shape[0], -1
                ).T
                if param_samples is not None
                else None
            ),
            true_params.emissions.emission_function.weights.flatten() if true_params is not None else None,
            init_params.emissions.emission_function.weights.flatten() if init_params is not None else None,
            pointwise_estimate=(
                pointwise_estimate.emissions.emission_function.weights.flatten()
                if pointwise_estimate is not None
                else None
            ),
            name="Emissions function weights",
            trainable=param_properties.emissions.emission_function.weights.trainable,
            burn_in_frac=burn_in_frac,
            triangle_plot=triangle_plot,
            box_plot=box_plot,
            triangle_traj_plot=triangle_traj_plot,
            sequence_plot=sequence_plot,
        )

    if param_properties.emissions.emission_function.bias.trainable or not skip_if_not_trainable:
        plot_param_distributions(
            (
                param_samples.emissions.emission_function.bias.reshape(
                    param_samples.emissions.emission_function.bias.shape[0], -1
                ).T
                if param_samples is not None
                else None
            ),
            true_params.emissions.emission_function.bias.flatten() if true_params is not None else None,
            init_params.emissions.emission_function.bias.flatten() if init_params is not None else None,
            pointwise_estimate=(
                pointwise_estimate.emissions.emission_function.bias.flatten()
                if pointwise_estimate is not None
                else None
            ),
            name="Emissions function bias",
            trainable=param_properties.emissions.emission_function.bias.trainable,
            burn_in_frac=burn_in_frac,
            triangle_plot=triangle_plot,
            box_plot=box_plot,
            triangle_traj_plot=triangle_traj_plot,
            sequence_plot=sequence_plot,
        )

    if param_properties.emissions.emission_cov.params.trainable or not skip_if_not_trainable:
        plot_param_distributions(
            (
                param_samples.emissions.emission_cov.params.reshape(
                    param_samples.emissions.emission_cov.params.shape[0], -1
                ).T
                if param_samples is not None
                else None
            ),
            true_params.emissions.emission_cov.params.flatten() if true_params is not None else None,
            init_params.emissions.emission_cov.params.flatten() if init_params is not None else None,
            pointwise_estimate=(
                pointwise_estimate.emissions.emission_cov.params.flatten() if pointwise_estimate is not None else None
            ),
            name="Emissions cov",
            trainable=param_properties.emissions.emission_cov.params.trainable,
            burn_in_frac=burn_in_frac,
            triangle_plot=triangle_plot,
            box_plot=box_plot,
            triangle_traj_plot=triangle_traj_plot,
            sequence_plot=sequence_plot,
        )
