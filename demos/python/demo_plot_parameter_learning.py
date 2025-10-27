# Imports
import jax
from jax.tree_util import tree_map
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

## Useful functions for learnable parameter identification and conversion to dataframe
# Figure out which models parameters were trainable -> hence trained and interested in plotting
def learnable_param_or_none(param, prop):
    return param if prop.trainable else None

def learnable_params_to_df(param_tree, prop_tree):
    # Strip out non-trainable parameter leaves,
    # Organized into a dictionary, with keys as concatenated path names and values as the parameter values
    learnable_param_dict = {
            '_'.join([item.name for item in path]): value
            for (path, value) in jax.tree.leaves_with_path(
                tree_map(
                    learnable_param_or_none, param_tree, prop_tree
                )
            )
        }

    # Check if all values in learnable_param_dict are scalars or 1D arrays
    all_scalar = all(
        pd.api.types.is_scalar(v) or (v.ndim == 0) for v in learnable_param_dict.values()
    )

    if all_scalar:
        # All scalars: We MUST provide an index to pandas dataframe
        learnable_param_df = pd.DataFrame(learnable_param_dict, index=[0])
    else:
        # At least one array. Let pandas build the index.
        learnable_param_df = pd.DataFrame(learnable_param_dict)

    return learnable_param_df

## Plotting Utilities
# Plot the marginal log likelihood learning curve
def plot_mll_learning_curve(
    true_model,
    true_params,
    true_emissions,
    t_emissions,
    marginal_lls,
    filter_hyperparams=None,
    plot_save_path=None,
):
    """Note that the true logjoint is computed using default filter hyperparameters in marginal_log_prob."""

    plt.figure()
    plt.xlabel("Iterations")
    mlp_args = [true_params, true_emissions, t_emissions]
    if filter_hyperparams is not None:
        mlp_args.append(filter_hyperparams)
    true_logjoint = true_model.log_prior(true_params) + true_model.marginal_log_prob(*mlp_args)
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
    if plot_save_path is not None:
        plt.savefig(plot_save_path, dpi=300)
        print(f"Marginal log likelihood learning curve saved to {plot_save_path}.")
    else:
        plt.show()
    
    # Close the plot to free memory
    plt.close()


# Plot the parameter distributions, given samples
def plot_param_dist(
    samples=None,
    true=None,
    init=None,
    pointwise_estimate=None,
    name="",
    burn_in_frac=0.0,
    pairwise_plots=False,
    plot_save_path=None,
    ):

    # Figure out number of parameters, from columns in param_history
    N_params = len(samples.columns)

    # Check for dimensions of true param dataframe
    if true is not None:
        assert len(true.columns) == N_params, "True parameter dimensions do not match."
    if init is not None:
        assert len(init.columns) == N_params, "Initial parameter dimensions do not match."
    if pointwise_estimate is not None:
        assert len(pointwise_estimate.columns) == N_params, "Pointwise estimate parameter dimensions do not match."
    
    # Figure out burning-in
    if burn_in_frac < 0.0 or burn_in_frac >= 1.0:
        raise ValueError("burn_in_frac must be in [0.0, 1.0).")
    
    # Apply burn-in to samples
    if burn_in_frac > 0.0:
        burn_in = int(burn_in_frac * samples.shape[0])
        samples = samples.iloc[burn_in:, :]

    # Figure with matplotlib the box plots for each parameter
    fig, ax = plt.subplots(
        figsize=(10, N_params * 2)
    ) 

    # Create box plots
    if samples is not None:
        ax.boxplot(
            samples,
            vert=False,
            patch_artist=True
    )

    # Plot ground truth point
    if true is not None:
        ax.scatter(
            true,
            range(1, N_params + 1),
            color="red",
            marker="x",
            s=100,
            label="Ground Truth",
            zorder=4
        )
        
    # Plot initial estimate point
    if init is not None:
        ax.scatter(
            init,
            range(1, N_params + 1),
            color="magenta",
            marker="o",
            s=100,
            label="Initial Estimate",
            zorder=3
        )
    
    # Plot pointwise estimate point
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

    # Set the y-axis labels to show parameter names
    ax.set_yticks(range(1, N_params + 1))
    ax.set_yticklabels(samples.columns)
    plt.ylabel("Parameters")
    # Set the x-axis
    plt.xlabel("Parameter Value")
    # Set the title
    plt.title("Learned Parameter Distributions")
    # Set the grid and legend
    plt.grid(True)
    plt.legend()
    # Save or show the plot
    if plot_save_path is not None:
        plt.savefig(f"{plot_save_path}_box_plot.png", dpi=300)
        print(f"Box plot saved to {plot_save_path}.")
    else:
        plt.show()
    plt.close()

    if pairwise_plots:
        # Initialize PairGrid
        g = sns.PairGrid(
            samples,
            diag_sharey=False
        )

        # KDE of samples on lower triangle
        g = g.map_lower(
            sns.kdeplot,
            fill=True,
            cmap="Blues"
        )

        # Histogram on diagonals
        g = g.map_diag(
            sns.histplot,
            kde=False,
            color="lightskyblue"
        )

        # Hide the upper triangle
        for i in range(N_params):
            for j in range(i+1, N_params):
                g.axes[i, j].set_visible(False)

        # Apply Reference Lines and Markers Without Setting Axis Limits
        for i, param in enumerate(samples.columns):
            # Diagonal axes
            diag_ax = g.axes[i, i]

            # Add reference lines on diagonal
            if true is not None:
                diag_ax.axvline(
                    true[param].values[0],
                    color="red",
                    linestyle="--",
                    label="Ground Truth"
                )
            if init is not None:
                diag_ax.axvline(
                    init[param].values[0],
                    color="magenta",
                    linestyle="--",
                    label="Initial Estimate"
                )
            if pointwise_estimate is not None:
                diag_ax.axvline(
                    pointwise_estimate[param].values[0],
                    color="orange",
                    linestyle="--",
                    label="Pointwise Estimate"
                )

        # Off-diagonal axes (lower triangle)
        for i, i_param in enumerate(samples.columns):
            for j, j_param in enumerate(samples.columns[:i]):
                ax = g.axes[i, j]

                # Add reference markers
                if true is not None:
                    ax.scatter(
                        true[j_param].values[0],
                        true[i_param].values[0],
                        color="red",
                        marker="x",
                        s=100,
                        zorder=4,
                        label="Ground Truth"
                    )
                if init is not None:
                    ax.scatter(
                        init[j_param].values[0],
                        init[i_param].values[0],
                        color="magenta",
                        marker="o",
                        s=100,
                        zorder=3,
                        label="Initial Estimate"
                    )
                if pointwise_estimate is not None:
                    ax.scatter(
                        pointwise_estimate[j_param].values[0],
                        pointwise_estimate[i_param].values[0],
                        color="orange",
                        marker="*",
                        s=100,
                        zorder=3,
                        label="Pointwise Estimate",
                    )

        # Join the legends from the diagonal and off-diagonal plots
        handles0, labels0 = g.axes[0, 0].get_legend_handles_labels()
        handles1, labels1 = g.axes[i, j].get_legend_handles_labels()
        handles = handles0 + handles1
        labels = labels0 + labels1

        # Add legend to the figure
        if handles:
            g.figure.legend(
                handles,
                labels,
                loc="upper right",
                bbox_to_anchor=(1, 0.95)
            )

        # Title
        g.figure.suptitle(f"{name} Triangle Plot with Bivariate Densities", y=1.02)
        # Adjust layout
        plt.tight_layout()
        # Save or show the plot
        if plot_save_path is not None:
            plt.savefig(f"{plot_save_path}_dist_triangle_plot.png", dpi=300)
            print(f"Triangle plot saved to {plot_save_path}.")
        else:
            plt.show()
        plt.close()


# Plot the parameter sequence over training (MCMC steps or optim iterations)
def plot_param_sequences(
        param_history,
        true=None,
        init=None,
        pointwise_estimate=None,
        burn_in_frac=0.0,
        pairwise_plots=False,
        plot_save_path=None,
    ):
    # Figure out number of parameters, from columns in param_history
    N_params = len(param_history.columns)

    # Check for dimensions of true param dataframe
    if true is not None:
        assert len(true.columns) == N_params, "True parameter dimensions do not match."
    if init is not None:
        assert len(init.columns) == N_params, "Initial parameter dimensions do not match."
    if pointwise_estimate is not None:
        assert len(pointwise_estimate.columns) == N_params, "Pointwise estimate parameter dimensions do not match."
    
    # Figure out burning-in
    if burn_in_frac < 0.0 or burn_in_frac >= 1.0:
        raise ValueError("burn_in_frac must be in [0.0, 1.0).")
    
    # Apply burn-in to param_history
    if burn_in_frac > 0.0:
        burn_in = int(burn_in_frac * param_history.shape[0])
        param_history = param_history.iloc[burn_in:, :]
    
    # Plot the parameter values over time/iterations
    fig, axes = plt.subplots(
        N_params, 1, figsize=(10, N_params * 2), sharex=True
    )

    # Create subplots for each parameter
    for i, param in enumerate(param_history.columns):
        # Plot sequence of parameter values
        axes[i].plot(
            param_history[param],
            color="C0",
            label="Parameter sequence"
        )
        # Plot ground truth line
        if true is not None:
            axes[i].axhline(
                true[param].values[0],
                color="red",
                linestyle="--",
                label="Ground Truth"
            )
        # Plot initial estimate as point at start
        if init is not None:
            axes[i].plot(
                param_history.index[0],
                init[param].values[0],
                color="magenta",
                marker="o",
                label="Initial Estimate"
            )
        # Plot pointwise estimate point at the end
        if pointwise_estimate is not None:
            axes[i].plot(
                param_history.index[-1],
                pointwise_estimate[param].values[0],
                color="orange",
                marker="*",
                label="Pointwise Estimate"
            )
        
        # Set y-label, from parameter name
        axes[i].set_ylabel(param)
        axes[i].grid(True)
    
    # Set x-label for last subplot
    axes[-1].set_xlabel("Iterations")
    
    # Title and legend
    plt.suptitle("Parameter Values over Iterations")
    plt.legend()
    
    # Layout adjustment
    plt.tight_layout()
    # Save or show the plot
    if plot_save_path is not None:
        plt.savefig(f"{plot_save_path}_sequence_plot.png", dpi=300)
        print(f"Sequence plot saved to {plot_save_path}.")
    else:
        plt.show()
    plt.close()

    if pairwise_plots:
        # Using PairGrid, pairwise trajectory plot
        # Custom plot grid, excluding diagonal and upper right subplots
        g = sns.PairGrid(
            param_history,
            diag_sharey=False
        )

        # Plot scatter plots in the lower triangle subplots with color gradient from magenta to blue
        def scatter_with_gradient(x, y=None, **kwargs):
            if y is None:
                y = x
            plt.scatter(
                x,
                y,
                c=np.linspace(0, 1, len(x)),
                cmap=sns.color_palette("cool_r", as_cmap=True),
                **{k: v for k, v in kwargs.items() if k != "color"}
            )

        # Map the scatter plot with gradient to the diagonal for i==j
        g.map_diag(
            scatter_with_gradient,
            s=10,
            zorder=2
        )
        # Map the scatter plot with gradient to the lower triangle for (i,j) with i>j
        g.map_lower(
            scatter_with_gradient,
            s=10,
            zorder=2
        )
        
        # Hide the upper triangle
        for i in range(N_params):
            for j in range(i+1, N_params):
                g.axes[i, j].set_visible(False)

        # Add Init and ground truth values to the plot
        for i, i_param in enumerate(param_history.columns):
            for j, j_param in enumerate(param_history.columns[:i+1]):
                ax = g.axes[i, j]
                # Skip missing axes (PairGrid with corner=True leaves some axes as None)
                if ax is None:
                    continue
                # Plot ground truth as points
                if true is not None:
                    # The ground truth point plotted on (j, i) subplot, as a point
                    ax.scatter(
                        true[j_param].values[0],
                        true[i_param].values[0],
                        color="red",
                        marker="x",
                        s=100,
                        zorder=4,
                        label="Ground Truth"
                    )
                # Plot initial estimate as points
                if init is not None:
                    # The initial estimate point plotted on (j, i) subplot
                    ax.scatter(
                        init[j_param].values[0],
                        init[i_param].values[0],
                        color="magenta",
                        marker="o",
                        s=100,
                        zorder=3,
                        label="Initial Estimate"
                    )
                # Plot pointwise estimate as points
                if pointwise_estimate is not None:
                    # The pointwise estimate point plotted on (j, i) subplot
                    ax.scatter(
                        pointwise_estimate[j_param].values[0],
                        pointwise_estimate[i_param].values[0],
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
        
        # Add legend to the figure
        if handles:
            g.figure.legend(
                handles,
                labels,
                loc="upper right",
                bbox_to_anchor=(1, 0.95)
        )
        # Title
        g.figure.suptitle("Parameter Pairwise Trajectory Plots", y=1.02)
        # Adjust layout
        plt.tight_layout()
        # Save or show the plot
        if plot_save_path is not None:
            plt.savefig(f"{plot_save_path}_sequence_triangle_plot.png", dpi=300)
            print(f"Triangle trajectory plot saved to {plot_save_path}.")
        else:
            plt.show()
        plt.close()