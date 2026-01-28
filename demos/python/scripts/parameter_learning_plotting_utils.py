# Imports
import jax
from jax import vmap
from jax.tree_util import tree_map
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


## Useful functions for learnable parameter identification and conversion to dataframe
# Figure out which models parameters were trainable -> hence trained and interested in plotting
def learnable_param_or_none(param, prop):
    r"""
    Returns the parameter if it is trainable, otherwise returns None.

    Args:
        param: The parameter value.
        prop: The parameter properties, which includes a 'trainable' attribute.

    Returns:
        The parameter if it is trainable, otherwise None.
    """
    return param if prop.trainable else None


# Convert the learnable parameters from a parameter tree to a DataFrame with 1D columns
def learnable_params_to_1d_df(param_tree, prop_tree, has_batch_dim=False):
    """
    Converts the learnable parameters from a parameter tree to a pandas DataFrame,
    where each column corresponds to a 1D parameter (scalar or array flattened into scalars).

    Args:
        param_tree: The parameter pytree (nested structure of parameters).
        prop_tree: The corresponding parameter properties pytree.
        has_batch_dim: Boolean indicating if the parameters have a batch dimension.

    Returns:
        A pandas DataFrame with each learnable parameter as a column.
    """

    # Strip out non-trainable parameter leaves,
    # Organized into a dictionary,
    # with keys as concatenated path names and values as the parameter values
    learnable_param_dict = {
        "_".join([item.name for item in path]): value
        for (path, value) in jax.tree.leaves_with_path(
            tree_map(learnable_param_or_none, param_tree, prop_tree)
        )
    }

    # Initialize the new dictionary for scalar parameters
    learnable_scalar_param_dict = {}

    # Determine the batch size, if needed
    batch_size = None
    if has_batch_dim:
        # Find the batch size from the shape of the first array-like value
        for value in learnable_param_dict.values():
            if hasattr(value, "shape") and value.ndim >= 1:
                batch_size = value.shape[0]
                break

        # If has_batch_dim is True but we couldn't find a batched array,
        # assume single-sample behavior to prevent errors.
        if batch_size is None:
            has_batch_dim = False

    # Process the dictionary of learnable parameters
    for key, value in learnable_param_dict.items():
        # Array-like values
        if hasattr(value, "ndim"):
            if has_batch_dim:
                # print(f'Processing array = {key} in batch mode')
                # Case A: Array in batch mode
                if value.ndim >= 1 and value.shape[0] == batch_size:
                    # Get the shape of the dimensions to iterate over
                    feature_shape = value.shape[1:]

                    # Iterate over all indices in the feature dimensions
                    for idx in np.ndindex(feature_shape):
                        # Construct the new key (e.g., 'dynamics_weights_0_0')
                        suffix = "_".join(map(str, idx))
                        new_key = (
                            f"{key}_{suffix}" if suffix else key
                        )  # 'key' if feature_shape is empty (1D array)

                        # Extracts the 1D array across the batch dimension (length B)
                        learnable_scalar_param_dict[new_key] = value[
                            (slice(None),) + idx
                        ]

                elif value.ndim == 0 and batch_size is not None:
                    # Scalar array (ndim=0) when in batch mode: broadcast its item value
                    learnable_scalar_param_dict[key] = [value.item()] * batch_size

                else:
                    # Array found, but not compatible with assumed batch_size (unlikely if data is clean)
                    # Treating as an error/warning and falling back to single-item value
                    print(
                        f"Warning: Array for '{key}' has incompatible shape {value.shape}. Treating as single item."
                    )
                    learnable_scalar_param_dict[key] = [
                        value
                    ]  # Will likely cause length mismatch error if other arrays are batched

            # Array in single-sample mode
            else:
                # print(f'Processing array = {key} in single-sample mode')
                this_shape = value.shape

                # If it's already a scalar (ndim=0)
                if this_shape == ():
                    learnable_scalar_param_dict[key] = value.item()
                else:
                    # Iterate over all indices and flatten the array into scalar entries
                    for idx in np.ndindex(this_shape):
                        new_key = f"{key}_{'_'.join(map(str, idx))}"
                        learnable_scalar_param_dict[new_key] = value[idx].item()

        # Non-array-like values
        else:
            # print(f'Processing non-array-like value = {key}')
            if has_batch_dim and batch_size is not None:
                # IMPORTANT: Broadcast the scalar value to a list of length batch_size
                learnable_scalar_param_dict[key] = [value] * batch_size
            else:
                # Single-sample mode: use the scalar value directly
                learnable_scalar_param_dict[key] = value

    # Return as DataFrame with each 1d parameter as a column
    if has_batch_dim and batch_size is not None:
        # All columns are 1D arrays of length B. This creates a multi-row DataFrame.
        return pd.DataFrame(learnable_scalar_param_dict)
    else:
        # All values are scalars. This requires an explicit index for a single row.
        return pd.DataFrame(learnable_scalar_param_dict, index=[0])


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
    r"""
        Plots the marginal log likelihood learning curve over iterations,
        comparing the estimated marginal log likelihoods to the true log joint probability.

    Args:
        true_model: The true model cd-dynamax object.
        true_params: The true model parameters.
        true_emissions: The emissions data from the true model --- could have batch dim.
        t_emissions: The emissions data used for training --- could have batch dim.
        marginal_lls: List or array of estimated marginal log likelihoods over iterations.
        filter_hyperparams: Optional filter hyperparameters for the true model.
            If None, the true logjoint is computed using default filter hyperparameters in marginal_log_prob.
        plot_save_path: Optional path to save the plot image. If None, the plot is shown instead.
    Returns:
        None
    """

    # Prepare arguments for true log joint computation
    mlp_args = [true_params, true_emissions, t_emissions]
    if filter_hyperparams is not None:
        mlp_args.append(filter_hyperparams)

    # Compute true log joint, taking care of batches if needed
    if len(true_emissions.shape) > 2 and len(t_emissions.shape) <= 2:
        true_logjoint = vmap(
            true_model.marginal_log_prob,
            in_axes=(None, 0, None, None)
            if filter_hyperparams is not None
            else (None, 0, None),
        )(*mlp_args).sum()
    elif len(true_emissions.shape) <= 2 and len(t_emissions.shape) > 2:
        true_logjoint = vmap(
            true_model.marginal_log_prob,
            in_axes=(None, None, 0, None)
            if filter_hyperparams is not None
            else (None, None, 0),
        )(*mlp_args).sum()
    elif len(true_emissions.shape) > 2 and len(t_emissions.shape) > 2:
        true_logjoint = vmap(
            true_model.marginal_log_prob,
            in_axes=(None, 0, 0, None)
            if filter_hyperparams is not None
            else (None, 0, 0),
        )(*mlp_args).sum()
    else:
        true_logjoint = true_model.marginal_log_prob(*mlp_args)
    # Add prior to likelihood
    true_logjoint += true_model.log_prior(true_params)

    # Plot the learning curve
    plt.figure()
    plt.xlabel("Iterations")
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
    y_min = (
        min(min(marginal_lls), true_logjoint) * 1.1
    )  # 10% lower than the smallest value
    y_max = (
        max(max(marginal_lls), true_logjoint) * 0.9
    )  # 10% higher than the largest value
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
    r"""
    Plots the parameter distributions as box plots and pairwise density plots.

    Args:
        samples: DataFrame of parameter samples (each column is a parameter).
        true: DataFrame of true parameter values (single row).
        init: DataFrame of initial parameter estimates (single row).
        pointwise_estimate: DataFrame of pointwise parameter estimates (single row).
        name: Name for the plot titles.
        burn_in_frac: Fraction of samples to discard as burn-in.
        pairwise_plots: Whether to generate pairwise density plots.
        plot_save_path: Optional path to save the plot images. If None, the plots are shown instead.
    Returns:
        None
    """

    # Figure out number of parameters, from columns in param_history
    N_params = len(samples.columns)

    # Check for dimensions of true param dataframe
    if true is not None:
        assert len(true.columns) == N_params, "True parameter dimensions do not match."
    if init is not None:
        assert len(init.columns) == N_params, (
            "Initial parameter dimensions do not match."
        )
    if pointwise_estimate is not None:
        assert len(pointwise_estimate.columns) == N_params, (
            "Pointwise estimate parameter dimensions do not match."
        )

    # Figure out burning-in
    if burn_in_frac < 0.0 or burn_in_frac >= 1.0:
        raise ValueError("burn_in_frac must be in [0.0, 1.0).")

    # Apply burn-in to samples
    if burn_in_frac > 0.0:
        burn_in = int(burn_in_frac * samples.shape[0])
        samples = samples.iloc[burn_in:, :]

    # Figure with matplotlib the box plots for each parameter
    fig, ax = plt.subplots(figsize=(10, N_params * 2))

    # Create box plots
    if samples is not None:
        ax.boxplot(samples, vert=False, patch_artist=True)

    # Plot ground truth point
    if true is not None:
        ax.scatter(
            true,
            range(1, N_params + 1),
            color="red",
            marker="x",
            s=100,
            label="Ground Truth",
            zorder=4,
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
            zorder=3,
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
        g = sns.PairGrid(samples, diag_sharey=False)

        # KDE of samples on lower triangle
        g = g.map_lower(sns.kdeplot, fill=True, cmap="Blues")

        # Histogram on diagonals
        g = g.map_diag(sns.histplot, kde=False, color="lightskyblue")

        # Hide the upper triangle
        for i in range(N_params):
            for j in range(i + 1, N_params):
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
                    label="Ground Truth",
                )
            if init is not None:
                diag_ax.axvline(
                    init[param].values[0],
                    color="magenta",
                    linestyle="--",
                    label="Initial Estimate",
                )
            if pointwise_estimate is not None:
                diag_ax.axvline(
                    pointwise_estimate[param].values[0],
                    color="orange",
                    linestyle="--",
                    label="Pointwise Estimate",
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
                        label="Ground Truth",
                    )
                if init is not None:
                    ax.scatter(
                        init[j_param].values[0],
                        init[i_param].values[0],
                        color="magenta",
                        marker="o",
                        s=100,
                        zorder=3,
                        label="Initial Estimate",
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
                handles, labels, loc="upper right", bbox_to_anchor=(1, 0.95)
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
    r"""
    Plots the parameter sequences over training iterations,
    including optional ground truth, initial estimates, and pointwise estimates.

    Args:
        param_history: DataFrame of parameter values over iterations (each column is a parameter).
        true: DataFrame of true parameter values (single row).
        init: DataFrame of initial parameter estimates (single row).
        pointwise_estimate: DataFrame of pointwise parameter estimates (single row).
        burn_in_frac: Fraction of initial iterations to discard as burn-in.
        pairwise_plots: Whether to generate pairwise trajectory plots.
        plot_save_path: Optional path to save the plot images. If None, the plots are shown instead.
    Returns:
        None
    """

    # Figure out number of parameters, from columns in param_history
    N_params = len(param_history.columns)

    # Check for dimensions of true param dataframe
    if true is not None:
        assert len(true.columns) == N_params, "True parameter dimensions do not match."
    if init is not None:
        assert len(init.columns) == N_params, (
            "Initial parameter dimensions do not match."
        )
    if pointwise_estimate is not None:
        assert len(pointwise_estimate.columns) == N_params, (
            "Pointwise estimate parameter dimensions do not match."
        )

    # Figure out burning-in
    if burn_in_frac < 0.0 or burn_in_frac >= 1.0:
        raise ValueError("burn_in_frac must be in [0.0, 1.0).")

    # Apply burn-in to param_history
    if burn_in_frac > 0.0:
        burn_in = int(burn_in_frac * param_history.shape[0])
        param_history = param_history.iloc[burn_in:, :]

    # Plot the parameter values over time/iterations
    fig, axes = plt.subplots(N_params, 1, figsize=(10, N_params * 2), sharex=True)

    # Create subplots for each parameter
    for i, param in enumerate(param_history.columns):
        # Plot sequence of parameter values
        axes[i].plot(param_history[param], color="C0", label="Parameter sequence")
        # Plot ground truth line
        if true is not None:
            axes[i].axhline(
                true[param].values[0], color="red", linestyle="--", label="Ground Truth"
            )
        # Plot initial estimate as point at start
        if init is not None:
            axes[i].plot(
                param_history.index[0],
                init[param].values[0],
                color="magenta",
                marker="o",
                label="Initial Estimate",
            )
        # Plot pointwise estimate point at the end
        if pointwise_estimate is not None:
            axes[i].plot(
                param_history.index[-1],
                pointwise_estimate[param].values[0],
                color="orange",
                marker="*",
                label="Pointwise Estimate",
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
        g = sns.PairGrid(param_history, diag_sharey=False)

        # Plot scatter plots in the lower triangle subplots with color gradient from magenta to blue
        def scatter_with_gradient(x, y=None, **kwargs):
            if y is None:
                y = x
            plt.scatter(
                x,
                y,
                c=np.linspace(0, 1, len(x)),
                cmap=sns.color_palette("cool_r", as_cmap=True),
                **{k: v for k, v in kwargs.items() if k != "color"},
            )

        # Map the scatter plot with gradient to the diagonal for i==j
        g.map_diag(scatter_with_gradient, s=10, zorder=2)
        # Map the scatter plot with gradient to the lower triangle for (i,j) with i>j
        g.map_lower(scatter_with_gradient, s=10, zorder=2)

        # Hide the upper triangle
        for i in range(N_params):
            for j in range(i + 1, N_params):
                g.axes[i, j].set_visible(False)

        # Add Init and ground truth values to the plot
        for i, i_param in enumerate(param_history.columns):
            for j, j_param in enumerate(param_history.columns[: i + 1]):
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
                        label="Ground Truth",
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
                        label="Initial Estimate",
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
                handles_list, labels_list = ax.get_legend_handles_labels()
                for handle, label in zip(handles_list, labels_list):
                    if label not in seen and label != "":
                        seen.add(label)
                        handles.append(handle)
                        labels.append(label)

        # Add legend to the figure
        if handles:
            g.figure.legend(
                handles, labels, loc="upper right", bbox_to_anchor=(1, 0.95)
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
