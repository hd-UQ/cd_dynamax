# JAX imports
import jax
import jax.numpy as jnp
from jax import vmap
import jax.random as jr

# Scientific imports
import numpy as np

# Plotting imports
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import transforms
import seaborn as sns

_COLOR_NAMES = [
    "windows blue",
    "red",
    "amber",
    "faded green",
    "dusty purple",
    "orange",
    "clay",
    "pink",
    "greyish",
    "mint",
    "light cyan",
    "steel blue",
    "forest green",
    "pastel purple",
    "salmon",
    "dark brown",
]
COLORS = sns.xkcd_palette(_COLOR_NAMES)


############################
# State and emission filtering utils
def plot_simple(
    time_grid=None,
    true_states=None,
    true_emissions_noisy=None,
    filtered_states=None,
    t_start=None,
    t_end=None,
):
    """
    Wrapper function to plot a simple time series.

    Parameters:
    - time_grid (array-like, optional): The time points corresponding to the true states and emissions.
    - true_states (ndarray, optional): The ground truth states, shape (T, n_states).
    - true_emissions_noisy (ndarray, optional): Noisy emissions from the true model, shape (T, n_emissions).
    - filtered_states (ndarray, optional): Filtered states using the learned model, shape (T, n_states).
    - t_start (float, optional): The start time for plotting.
    - t_end (float, optional): The end time for plotting.

    """

    plot_states_and_emissions(
        time_grid_all=time_grid,
        true_states=true_states,
        true_emissions_noisy=true_emissions_noisy,
        model_filtered_states=filtered_states,
        t_start=t_start,
        t_end=t_end,
    )


def plot_states_and_emissions(
    time_grid_all=None,
    time_grid_filter=None,
    time_grid_forecast=None,
    true_states=None,
    true_filtered_states=None,
    model_filtered_states=None,
    true_filtered_covariances=None,
    model_filtered_covariances=None,
    model_forecast_covariances=None,
    true_forecast_states=None,
    model_forecast_states=None,
    true_emissions=None,
    emission_function=None,
    t_start=None,
    t_end=None,
):
    """
    Plot the true states and emissions, as well as filtered and forecasted states and emissions from true and learned models.

    Parameters:
    - time_grid_all (array-like, optional): The time points corresponding to the true states and emissions.
    - time_grid_filter (array-like, optional): The time points corresponding to the filtered states and emissions.
    - time_grid_forecast (array-like, optional): The time points corresponding to the forecasted states and emissions.
    - true_states (ndarray, optional): The ground truth states, shape (T, n_states).
    - true_filtered_states (ndarray, optional): Filtered states using the true model, shape (T, n_states).
    - model_filtered_states (ndarray, optional): Filtered states using the learned model, shape (T, n_states).
    - true_filtered_covariances (ndarray, optional): Covariance matrices for filtered states using the true model, shape (T, n_states, n_states).
    - model_filtered_covariances (ndarray, optional): Covariance matrices for filtered states using the learned model, shape (T, n_states, n_states).
    - model_forecast_covariances (ndarray, optional): Covariance matrices for forecasted states using the learned model, shape (T, n_states, n_states).
    - true_forecast_states (ndarray, optional): Forecasted states using the true model, shape (T, n_states).
    - model_forecast_states (ndarray, optional): Forecasted states using the learned model, shape (T, n_states).
    - true_emissions (ndarray, optional): Noisy emissions from the true model, shape (T, n_emissions).
    - emission_function (LearnableLinear, optional): A function that computes emissions from states.
    - t_start (float, optional): The start time for plotting.
    - t_end (float, optional): The end time for plotting.
    """
    # Set default time grid if not specified
    plot_divider = True
    if time_grid_forecast is None:
        time_grid_forecast = time_grid_all
        plot_divider = False
    if time_grid_filter is None:
        time_grid_filter = time_grid_all
        plot_divider = False
    if time_grid_all is None:
        raise ValueError(
            "time_grid_all must be specified if time_grid_filter or time_grid_forecast are not provided."
        )

    # Ensure at least one time grid is specified
    if (
        time_grid_all is None
        and time_grid_filter is None
        and time_grid_forecast is None
    ):
        raise ValueError(
            "One of time_grid_all, time_grid_filter, or time_grid_forecast must be specified."
        )

    # squeeze time grids to ensure they are 1D arrays
    time_grid_all = np.squeeze(time_grid_all)
    time_grid_filter = np.squeeze(time_grid_filter)
    time_grid_forecast = np.squeeze(time_grid_forecast)

    # Determine indices based on t_start and t_end for all time grids
    if t_start is not None:
        start_idx_all = np.searchsorted(time_grid_all, t_start)
        start_idx_filter = np.searchsorted(time_grid_filter, t_start)
        start_idx_forecast = np.searchsorted(time_grid_forecast, t_start)
    else:
        start_idx_all = start_idx_filter = start_idx_forecast = 0

    if t_end is not None:
        end_idx_all = np.searchsorted(time_grid_all, t_end, side="right")
        end_idx_filter = np.searchsorted(time_grid_filter, t_end, side="right")
        end_idx_forecast = np.searchsorted(time_grid_forecast, t_end, side="right")
    else:
        end_idx_all = len(time_grid_all)
        end_idx_filter = len(time_grid_filter)
        end_idx_forecast = len(time_grid_forecast)

    # Subset the time grids based on start and end indices
    time_grid_all = time_grid_all[start_idx_all:end_idx_all]
    time_grid_filter = time_grid_filter[start_idx_filter:end_idx_filter]
    time_grid_forecast = time_grid_forecast[start_idx_forecast:end_idx_forecast]

    # Subset the time and state arrays based on start and end indices for each grid
    if true_states is not None:
        true_states = true_states[start_idx_all:end_idx_all, :]
    if true_filtered_states is not None:
        true_filtered_states = true_filtered_states[start_idx_filter:end_idx_filter, :]
    if model_filtered_states is not None:
        model_filtered_states = model_filtered_states[
            start_idx_filter:end_idx_filter, :
        ]
    if true_filtered_covariances is not None:
        true_filtered_covariances = true_filtered_covariances[
            start_idx_filter:end_idx_filter, :, :
        ]
    if model_filtered_covariances is not None:
        model_filtered_covariances = model_filtered_covariances[
            start_idx_filter:end_idx_filter, :, :
        ]
    if true_forecast_states is not None:
        true_forecast_states = true_forecast_states[
            start_idx_forecast:end_idx_forecast, :
        ]
    if model_forecast_states is not None:
        model_forecast_states = model_forecast_states[
            start_idx_forecast:end_idx_forecast, :
        ]
    if model_forecast_covariances is not None:
        model_forecast_covariances = model_forecast_covariances[
            start_idx_forecast:end_idx_forecast, :, :
        ]
    if true_emissions is not None:
        true_emissions = true_emissions[start_idx_all:end_idx_all, :]

    # Determine the number of rows for subplots
    n_states = true_states.shape[1] if true_states is not None else 0
    n_emissions = (
        true_emissions.shape[1]
        if true_emissions is not None
        else (
            emission_function.f(true_states[0]).shape[0]
            if true_states is not None and emission_function is not None
            else 0
        )
    )
    n_rows = max(n_states, n_emissions)

    # Create a canvas with subplots for states and emissions
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=2, figsize=(15, 2 * n_rows), sharex=True
    )

    # Use vmap to apply emission_function across the batch of states
    if emission_function is not None:
        vmap_emission = jax.vmap(emission_function.f, in_axes=(0, None, None))

    # Define color and linestyle profiles for different data types
    true_color = "k"
    true_filtered_color = "gray"
    model_filtered_color = "C0"
    linestyle_true = "-"
    linestyle_filtered = "--"
    linestyle_forecast = ":"
    marker_noisy_emission = "x"

    # Plot the states in the first column
    if true_states is not None:
        for i in range(n_states):
            # Plot true state values
            axes[i, 0].plot(
                time_grid_all,
                true_states[:, i],
                linestyle_true,
                color=true_color,
                alpha=0.5,
                linewidth=2,
                label=f"True State {i}",
            )
            # Plot filtered state values from true model
            if true_filtered_states is not None:
                axes[i, 0].plot(
                    time_grid_filter,
                    true_filtered_states[:, i],
                    color=true_filtered_color,
                    linestyle=linestyle_filtered,
                    alpha=0.7,
                    linewidth=2,
                    label=f"True Filtered State {i}",
                )
                # Plot uncertainty bounds if available
                if (
                    true_filtered_covariances is not None
                    and true_filtered_covariances.shape[0] == time_grid_filter.shape[0]
                ):
                    axes[i, 0].fill_between(
                        time_grid_filter,
                        true_filtered_states[:, i]
                        - np.sqrt(true_filtered_covariances[:, i, i]),
                        true_filtered_states[:, i]
                        + np.sqrt(true_filtered_covariances[:, i, i]),
                        color=true_filtered_color,
                        alpha=0.3,
                    )
            # Plot filtered state values from learned model
            if model_filtered_states is not None:
                axes[i, 0].plot(
                    time_grid_filter,
                    model_filtered_states[:, i],
                    color=model_filtered_color,
                    linestyle=linestyle_filtered,
                    alpha=0.7,
                    linewidth=2,
                    label=f"Model Filtered State {i}",
                )
                # Plot uncertainty bounds if available
                if (
                    model_filtered_covariances is not None
                    and model_filtered_covariances.shape[0] == time_grid_filter.shape[0]
                ):
                    axes[i, 0].fill_between(
                        time_grid_filter,
                        model_filtered_states[:, i]
                        - np.sqrt(model_filtered_covariances[:, i, i]),
                        model_filtered_states[:, i]
                        + np.sqrt(model_filtered_covariances[:, i, i]),
                        color=model_filtered_color,
                        alpha=0.3,
                    )
            # Plot forecast state values from true model
            if true_forecast_states is not None:
                axes[i, 0].plot(
                    time_grid_forecast,
                    true_forecast_states[:, i],
                    color=true_color,
                    linestyle=linestyle_forecast,
                    alpha=0.7,
                    linewidth=2,
                    label=f"True Forecast State {i}",
                )
            # Plot forecast state values from learned model
            if model_forecast_states is not None:
                axes[i, 0].plot(
                    time_grid_forecast,
                    model_forecast_states[:, i],
                    color=model_filtered_color,
                    linestyle=linestyle_forecast,
                    alpha=0.7,
                    linewidth=2,
                    label=f"Model Forecast State {i}",
                )
                # Plot uncertainty bounds if available
                if (
                    model_forecast_covariances is not None
                    and model_forecast_covariances.shape[0]
                    == time_grid_forecast.shape[0]
                ):
                    axes[i, 0].fill_between(
                        time_grid_forecast,
                        model_forecast_states[:, i]
                        - np.sqrt(model_forecast_covariances[:, i, i]),
                        model_forecast_states[:, i]
                        + np.sqrt(model_forecast_covariances[:, i, i]),
                        color=model_filtered_color,
                        alpha=0.3,
                    )

            # Set y-axis label and add legend
            axes[i, 0].set_ylabel(f"State {i}")
            axes[i, 0].legend(loc="lower left")

    # Plot the emissions in the second column
    for i in range(n_emissions):
        # Plot noisy true emissions if available
        if true_emissions is not None:
            axes[i, 1].plot(
                time_grid_all,
                true_emissions[:, i],
                marker_noisy_emission,
                color=true_color,
                alpha=0.5,
                ms=3,
                label=f"Noisy True Emission {i}",
            )

        # Plot filtered emission values from true model
        if emission_function is not None and true_filtered_states is not None:
            true_filtered_emissions = vmap_emission(true_filtered_states, None, None)
            axes[i, 1].plot(
                time_grid_filter,
                true_filtered_emissions[:, i],
                color=true_filtered_color,
                linestyle=linestyle_filtered,
                alpha=0.7,
                linewidth=2,
                label=f"True Filtered Emission {i}",
            )
        # Plot filtered emission values from learned model
        if emission_function is not None and model_filtered_states is not None:
            model_filtered_emissions = vmap_emission(model_filtered_states, None, None)
            axes[i, 1].plot(
                time_grid_filter,
                model_filtered_emissions[:, i],
                color=model_filtered_color,
                linestyle=linestyle_filtered,
                alpha=0.7,
                linewidth=2,
                label=f"Model Filtered Emission {i}",
            )
        # Plot forecast emission values from true model
        if emission_function is not None and true_forecast_states is not None:
            true_forecast_emissions = vmap_emission(true_forecast_states, None, None)
            axes[i, 1].plot(
                time_grid_forecast,
                true_forecast_emissions[:, i],
                color=true_color,
                linestyle=linestyle_forecast,
                alpha=0.7,
                linewidth=2,
                label=f"True Forecast Emission {i}",
            )
        # Plot forecast emission values from learned model
        if emission_function is not None and model_forecast_states is not None:
            model_forecast_emissions = vmap_emission(model_forecast_states, None, None)
            axes[i, 1].plot(
                time_grid_forecast,
                model_forecast_emissions[:, i],
                color=model_filtered_color,
                linestyle=linestyle_forecast,
                alpha=0.7,
                linewidth=2,
                label=f"Model Forecast Emission {i}",
            )

        # Set y-axis label and add legend
        axes[i, 1].set_ylabel(f"Emission {i}")
        axes[i, 1].legend(loc="lower left")

    # Plot a vertical line to indicate the switch between filtered and forecasted states if both are provided
    if plot_divider:
        if len(time_grid_filter) > 0:
            switch_time = time_grid_filter[-1]
        elif len(time_grid_forecast) > 0:
            switch_time = time_grid_forecast[0]
        else:
            switch_time = None

        if switch_time is not None:
            for i in range(n_rows):
                axes[i, 0].axvline(
                    x=switch_time,
                    color="k",
                    linestyle="--",
                    linewidth=1,
                    label="Filter/Forecast Boundary",
                )
                axes[i, 1].axvline(
                    x=switch_time, color="k", linestyle="--", linewidth=1
                )

    # Set x-axis label only on the bottom subplots
    if n_rows > 0:
        for ax in axes[-1, :]:
            ax.set_xlabel("Time $t$")

    # Set a super title for the entire figure
    plt.suptitle(
        "True vs Filtered vs Forecast States and Emissions: True and Learned Models"
    )

    # Adjust layout to prevent overlap and show the plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust space for the title
    plt.show()


# Plot filtered and forecasted states and emissions from true and learned models
def plot_filtered_and_forecasted(
    time_grid_all=None,
    time_grid_filter=None,
    time_grid_forecast=None,
    true_states=None,
    true_filtered_states=None,
    model_filtered_states=None,
    true_forecast_states=None,
    model_forecast_states=None,
    true_emissions_noisy=None,
    emission_function=None,
    t_start=None,
    t_end=None,
    N_samples=None,
    plot_ensemble=True,
):
    # Set default time grid if not specified
    plot_divider = True
    if time_grid_forecast is None:
        time_grid_forecast = time_grid_all
        plot_divider = False
    if time_grid_filter is None:
        time_grid_filter = time_grid_all
        plot_divider = False
    if time_grid_all is None:
        raise ValueError(
            "time_grid_all must be specified if time_grid_filter or time_grid_forecast are not provided."
        )

    # Ensure at least one time grid is specified
    if (
        time_grid_all is None
        and time_grid_filter is None
        and time_grid_forecast is None
    ):
        raise ValueError(
            "One of time_grid_all, time_grid_filter, or time_grid_forecast must be specified."
        )

    # Squeeze time grids to ensure they are 1D arrays
    time_grid_all = np.squeeze(time_grid_all)
    time_grid_filter = np.squeeze(time_grid_filter)
    time_grid_forecast = np.squeeze(time_grid_forecast)

    # Determine indices based on t_start and t_end for all time grids
    start_idx_all = (
        np.searchsorted(time_grid_all, t_start) if t_start is not None else 0
    )
    end_idx_all = (
        np.searchsorted(time_grid_all, t_end, side="right")
        if t_end is not None
        else len(time_grid_all)
    )

    start_idx_filter = (
        np.searchsorted(time_grid_filter, t_start) if t_start is not None else 0
    )
    end_idx_filter = (
        np.searchsorted(time_grid_filter, t_end, side="right")
        if t_end is not None
        else len(time_grid_filter)
    )

    start_idx_forecast = (
        np.searchsorted(time_grid_forecast, t_start) if t_start is not None else 0
    )
    end_idx_forecast = (
        np.searchsorted(time_grid_forecast, t_end, side="right")
        if t_end is not None
        else len(time_grid_forecast)
    )

    # Subset the time grids based on start and end indices
    time_grid_all = time_grid_all[start_idx_all:end_idx_all]
    time_grid_filter = time_grid_filter[start_idx_filter:end_idx_filter]
    time_grid_forecast = time_grid_forecast[start_idx_forecast:end_idx_forecast]

    # Subset the time and state arrays based on start and end indices for each grid
    def subset_data(data, start_idx, end_idx):
        if data is not None:
            return (
                data[..., start_idx:end_idx, :]
                if data.ndim > 2
                else data[start_idx:end_idx, :]
            )
        return None

    true_states = subset_data(true_states, start_idx_all, end_idx_all)
    true_filtered_states = subset_data(
        true_filtered_states, start_idx_filter, end_idx_filter
    )
    model_filtered_states = subset_data(
        model_filtered_states, start_idx_filter, end_idx_filter
    )
    true_forecast_states = subset_data(
        true_forecast_states, start_idx_forecast, end_idx_forecast
    )
    model_forecast_states = subset_data(
        model_forecast_states, start_idx_forecast, end_idx_forecast
    )
    true_emissions_noisy = subset_data(true_emissions_noisy, start_idx_all, end_idx_all)

    # Determine the number of rows for subplots
    n_states = true_states.shape[-1] if true_states is not None else 0
    n_emissions = (
        true_emissions_noisy.shape[-1] if true_emissions_noisy is not None else 0
    )
    n_rows = max(n_states, n_emissions)

    # Create a canvas with subplots for states and emissions
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=2, figsize=(15, 2 * n_rows), sharex=True
    )

    # Function to determine if a batch dimension is present
    def has_batch_dimension(data, N_samples):
        # Warning, this could be a bad check if the data is not structured as expected
        return data is not None and data.shape[0] == N_samples

    # Function to plot confidence intervals if a batch dimension is present
    def plot_with_ci(ax, x, y, label, color, linestyle, alpha=0.7):
        if has_batch_dimension(y, N_samples):  # Check if there's a batch dimension
            mean = jnp.mean(y, axis=0)
            std = jnp.std(y, axis=0)
            ax.plot(x, mean, color=color, linestyle=linestyle, alpha=alpha, label=label)
            ax.fill_between(
                x, mean - 1.96 * std, mean + 1.96 * std, color=color, alpha=0.3
            )
        else:
            ax.plot(x, y, color=color, linestyle=linestyle, alpha=alpha, label=label)

    # Plot the states in the first column
    if true_states is not None:
        for i in range(n_states):
            # Plot true state values
            axes[i, 0].plot(
                time_grid_all,
                true_states[..., i],
                "k-",
                alpha=0.5,
                linewidth=2,
                label=f"True State {i}",
            )
            # Plot filtered state values from true model
            if true_filtered_states is not None:
                plot_with_ci(
                    axes[i, 0],
                    time_grid_filter,
                    true_filtered_states[..., i],
                    f"True Filtered State {i}",
                    "gray",
                    "--",
                )
            # Plot filtered state values from learned model
            if model_filtered_states is not None:
                plot_with_ci(
                    axes[i, 0],
                    time_grid_filter,
                    model_filtered_states[..., i],
                    f"Model Filtered State {i}",
                    "C0",
                    "--",
                )
            # Plot forecast state values from true model
            if true_forecast_states is not None:
                axes[i, 0].plot(
                    time_grid_forecast,
                    true_forecast_states[..., i],
                    "k:",
                    alpha=0.7,
                    linewidth=2,
                    label=f"True Forecast State {i}",
                )
            # Plot forecast state values from learned model
            if model_forecast_states is not None:
                plot_with_ci(
                    axes[i, 0],
                    time_grid_forecast,
                    model_forecast_states[..., i],
                    f"Model Forecast State {i}",
                    "C0",
                    ":",
                )

            # Set y-axis label and add legend
            axes[i, 0].set_ylabel(f"State {i}")
            axes[i, 0].legend(loc="lower left")

    def states_by_emission_fs(e_func, states):
        # if emission_function is a list of functions, apply each one to the states
        if isinstance(e_func, list):

            def apply_nth_function(n, batch):
                # Define a function that applies the nth function to a single vector
                apply_function = lambda vec: jax.lax.switch(n, e_func, vec)
                # Use vmap to apply the function across all vectors in the batch (2000, 3)
                return jax.vmap(apply_function)(batch)

            batched_apply = jax.vmap(apply_nth_function, in_axes=(0, 0))
            output_array = batched_apply(jnp.arange(len(e_func)), states)

        # else, apply the single function to the states across all time steps and samples
        else:
            output_array = jax.vmap(e_func)(states)

        return output_array

    # Plot the emissions in the second column
    if true_emissions_noisy is not None or (
        emission_function is not None and model_filtered_states is not None
    ):
        if emission_function is None:
            raise ValueError(
                "emission_function must be provided to plot emissions from the learned model."
            )
        else:
            if model_filtered_states is not None:
                model_filtered_emissions = states_by_emission_fs(
                    emission_function, model_filtered_states
                )
            else:
                model_filtered_emissions = None

            if model_forecast_states is not None:
                model_forecast_emissions = states_by_emission_fs(
                    emission_function, model_forecast_states
                )
            else:
                model_forecast_emissions = None

        for i in range(n_emissions):
            # Plot noisy true emissions if available
            axes[i, 1].plot(
                time_grid_all,
                true_emissions_noisy[..., i],
                "kx",
                alpha=0.5,
                ms=3,
                label=f"Noisy True Emission {i}",
            )
            # Plot emissions computed from filtered states
            if model_filtered_emissions is not None:
                plot_with_ci(
                    axes[i, 1],
                    time_grid_filter,
                    model_filtered_emissions[..., i],
                    f"Filtered Emission {i}",
                    "C0",
                    "--",
                )
            # Plot emissions computed from forecasted states
            if model_forecast_emissions is not None:
                plot_with_ci(
                    axes[i, 1],
                    time_grid_forecast,
                    model_forecast_emissions[..., i],
                    f"Forecast Emission {i}",
                    "C0",
                    ":",
                )

            # Set y-axis label and add legend
            axes[i, 1].set_ylabel(f"Emission {i}")
            axes[i, 1].legend(loc="lower left")

    # Plot a vertical line to indicate the switch between filtered and forecasted states if both are provided
    if plot_divider:
        if len(time_grid_filter) > 0:
            switch_time = time_grid_filter[-1]
        elif len(time_grid_forecast) > 0:
            switch_time = time_grid_forecast[0]
        else:
            switch_time = None

        if switch_time is not None:
            for i in range(n_rows):
                axes[i, 0].axvline(
                    x=switch_time,
                    color="k",
                    linestyle="--",
                    linewidth=1,
                    label="Filter/Forecast Boundary",
                )
                axes[i, 1].axvline(
                    x=switch_time, color="k", linestyle="--", linewidth=1
                )

    # Set x-axis label only on the bottom subplots
    if n_rows > 0:
        for ax in axes[-1, :]:
            ax.set_xlabel("Time $t$")

    # Set a super title for the entire figure
    plt.suptitle(
        "True vs Filtered vs Forecast States and Emissions: True and Learned Models"
    )

    # Adjust layout to prevent overlap and show the plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust space for the title
    plt.show()

    ## Next, make plots of invariant measures for the true and learned models
    # in the forecatsed states.

    # First make a plot of the emission variables. Each subplot row will correspond to a different emission variable.
    # Use black for the true model and blue for the learned model.
    # We will only use the forecasted states for this plot.
    # There will only be 1 column of subplots, since we are plotting emissions.

    # Create a canvas with subplots for emissions
    fig, axes = plt.subplots(
        nrows=n_emissions, ncols=1, figsize=(10, 2 * n_emissions), sharex=True
    )
    if n_emissions == 1:
        axes = [axes]

    # Plot the emissions in the second column
    if true_emissions_noisy is not None or (
        emission_function is not None and model_filtered_states is not None
    ):
        if emission_function is None:
            raise ValueError(
                "emission_function must be provided to plot emissions from the learned model."
            )
        else:
            if model_filtered_states is not None:
                model_filtered_emissions = states_by_emission_fs(
                    emission_function, model_filtered_states
                )
            else:
                model_filtered_emissions = None

            if model_forecast_states is not None:
                model_forecast_emissions = states_by_emission_fs(
                    emission_function, model_forecast_states
                )
            else:
                model_forecast_emissions = None

        def plot_kde_with_ci(ax, batched_data, color, label, n_grid_points=100):
            # Compute the KDE estimate for each batch of emissions separately via vmap,
            # then plot the mean/CI of the KDE estimates across batches.
            if not has_batch_dimension(batched_data, N_samples):
                if len(batched_data) > 0:
                    sns.kdeplot(batched_data, ax=ax, color=color, label=label)
                    return
                else:
                    return
            else:
                if len(batched_data[0]) == 0:
                    return
                else:
                    pass

            emission_kde = jax.vmap(
                lambda x: jax.scipy.stats.gaussian_kde(x, bw_method="scott"), in_axes=0
            )(batched_data)

            # choose a grid of 1000 x values for the plot
            x = jnp.linspace(
                jnp.min(batched_data), jnp.max(batched_data), n_grid_points
            )
            # compute the KDE estimate for each batch at the x values
            kde_estimates = jax.vmap(lambda kde: kde(x))(emission_kde)
            if plot_ensemble:
                my_label = label + " Ensemble"
                for kde in kde_estimates:
                    ax.plot(x, kde, color=color, alpha=0.1, label=my_label)
                    my_label = None  # only label the first plot
                return
            # compute the mean and std of the KDE estimates across batches
            kde_mean = jnp.mean(kde_estimates, axis=0)
            kde_std = jnp.std(kde_estimates, axis=0)
            # plot the mean KDE estimate with 95% CI
            ax.plot(x, kde_mean, color=color, label=label)
            ax.fill_between(
                x,
                kde_mean - 1.96 * kde_std,
                kde_mean + 1.96 * kde_std,
                color=color,
                alpha=0.3,
            )

            return

        for i in range(n_emissions):
            # plot kde for true_emissions_noisy[..., i] and label it as the i-th true emission
            if true_emissions_noisy is not None and len(true_emissions_noisy) > 0:
                plot_kde_with_ci(
                    axes[i],
                    true_emissions_noisy[..., i],
                    color="black",
                    label="True Emission",
                )

            # plot kde for model_forecast_emissions[..., i] and label it as the i-th learned emission
            if (
                model_forecast_emissions is not None
                and len(model_forecast_emissions) > 0
            ):
                plot_kde_with_ci(
                    axes[i],
                    model_forecast_emissions[..., i],
                    color="blue",
                    label="Learned Emission",
                )
            axes[i].set_ylabel(f"Emission {i}")
            axes[i].legend(loc="upper right")

    # Adjust layout to prevent overlap and show the plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust space for the title

    # Make a title for the figure
    plt.suptitle("True vs Learned Invariant Measures: Emissions")
    plt.show()

    # Now do the same for the states. Each subplot row will correspond to a different state variable.
    # Create a canvas with subplots for emissions
    if (
        true_forecast_states is not None
        or model_forecast_states is not None
        or true_states is not None
    ):
        fig, axes = plt.subplots(
            nrows=n_states, ncols=1, figsize=(10, 2 * n_states), sharex=True
        )
        if n_states == 1:
            axes = [axes]
        # true_forecast_states[..., i] and model_forecast_states[..., i]

        for i in range(n_states):
            # plot kde for true_emissions_noisy[..., i] and label it as the i-th true emission

            if true_forecast_states is not None and len(true_forecast_states) > 0:
                plot_kde_with_ci(
                    axes[i],
                    true_forecast_states[..., i],
                    color="black",
                    label="True State",
                )
            elif true_states is not None and len(true_states) > 0:
                plot_kde_with_ci(
                    axes[i], true_states[..., i], color="black", label="True State"
                )

            if model_forecast_states is not None and len(model_forecast_states) > 0:
                plot_kde_with_ci(
                    axes[i],
                    model_forecast_states[..., i],
                    color="blue",
                    label="Learned State",
                )
            axes[i].set_ylabel(f"State {i}")
            axes[i].legend(loc="upper right")

    # Adjust layout to prevent overlap and show the plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust space for the title

    # Make a title for the figure
    plt.suptitle("True vs Learned Invariant Measures: States")

    plt.show()


################################


def white_to_color_cmap(color, nsteps=256):
    """Return a cmap which ranges from white to the specified color.
    Ported from HIPS-LIB plotting functions [https://github.com/HIPS/hips-lib]
    """
    # Get a red-white-black cmap
    cdict = {
        "red": ((0.0, 1.0, 1.0), (1.0, color[0], color[0])),
        "green": ((0.0, 1.0, 1.0), (1.0, color[1], color[0])),
        "blue": ((0.0, 1.0, 1.0), (1.0, color[2], color[0])),
    }
    cmap = LinearSegmentedColormap("white_color_colormap", cdict, nsteps)
    return cmap


def gradient_cmap(colors, nsteps=256, bounds=None):
    """Return a colormap that interpolates between a set of colors.
    Ported from HIPS-LIB plotting functions [https://github.com/HIPS/hips-lib]
    """
    ncolors = len(colors)
    # assert colors.shape[1] == 3
    if bounds is None:
        bounds = jnp.linspace(0, 1, ncolors)

    reds = []
    greens = []
    blues = []
    alphas = []
    for b, c in zip(bounds, colors):
        reds.append((b, c[0], c[0]))
        greens.append((b, c[1], c[1]))
        blues.append((b, c[2], c[2]))
        alphas.append((b, c[3], c[3]) if len(c) == 4 else (b, 1.0, 1.0))

    cdict = {
        "red": tuple(reds),
        "green": tuple(greens),
        "blue": tuple(blues),
        "alpha": tuple(alphas),
    }

    cmap = LinearSegmentedColormap("grad_colormap", cdict, nsteps)
    return cmap


CMAP = gradient_cmap(COLORS)


# https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
def plot_ellipse(Sigma, mu, ax, n_std=3.0, facecolor="none", edgecolor="k", **kwargs):
    """Plot an ellipse to with centre `mu` and axes defined by `Sigma`."""
    cov = Sigma
    pearson = cov[0, 1] / jnp.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = jnp.sqrt(1 + pearson)
    ell_radius_y = jnp.sqrt(1 - pearson)

    # if facecolor not in kwargs:
    #     kwargs['facecolor'] = 'none'
    # if edgecolor not in kwargs:
    #     kwargs['edgecolor'] = 'k'

    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        edgecolor=edgecolor,
        **kwargs,
    )

    scale_x = jnp.sqrt(cov[0, 0]) * n_std
    mean_x = mu[0]

    scale_y = jnp.sqrt(cov[1, 1]) * n_std
    mean_y = mu[1]

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)


def plot_uncertainty_ellipses(means, Sigmas, ax, n_std=3.0, **kwargs):
    """Loop over means and Sigmas to add ellipses representing uncertainty."""
    for Sigma, mu in zip(Sigmas, means):
        plot_ellipse(Sigma, mu, ax, n_std, **kwargs)


# Some custom params to make prettier plots.
custom_rcparams_base = {
    "font.size": 13.0,
    "font.sans-serif": [
        "Helvetica Neue",
        "Lucida Grande",
        "Verdana",
        "Geneva",
        "Lucid",
        "Arial",
        "Avant Garde",
        "sans-serif",
    ],
    "text.color": "555555",
    "axes.facecolor": "white",  ## axes background color
    "axes.edgecolor": "555555",  ## axes edge color
    "axes.linewidth": 1,  ## edge linewidth
    "axes.titlesize": 14,  ## fontsize of the axes title
    "axes.titlepad": 10.0,  ## pad between axes and title in points
    "axes.labelcolor": "555555",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": plt.cycler(
        "color",
        [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ],
    ),
    "xtick.color": "555555",
    "ytick.color": "555555",
    "grid.color": "eeeeee",  ## grid color
    "legend.frameon": False,  ## if True, draw the legend on a background patch
    "figure.titlesize": 16,  ## size of the figure title (Figure.suptitle())
    "figure.facecolor": "white",  ## figure facecolor
    "figure.frameon": False,  ## enable figure frame
    "figure.subplot.top": 0.91,  ## the top of the subplots of the figure
}

# Some custom params specifically designed for plots in a notebook.
custom_rcparams_notebook = {
    **custom_rcparams_base,
    "figure.figsize": (7.0, 5.0),
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "grid.linewidth": 1,
    "lines.linewidth": 1.75,
    "patch.linewidth": 0.3,
    "lines.markersize": 7,
    "lines.markeredgewidth": 0,
    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.major.pad": 7,
    "ytick.major.pad": 7,
}


def plot_learning_curve(
    marginal_lls,
    true_model,
    true_params,
    test_model,
    test_params,
    emissions,
    t_emissions=None,
):
    plt.figure(figsize=(10, 6))
    plt.xlabel("Iteration")
    plt.ylabel("Marginal Joint Probability")
    plt.plot(marginal_lls, label=f"Estimated ({max(marginal_lls):.2f})")

    # Compute true_marginal_lls and true_logjoint here
    try:
        true_marginal_lls = vmap(
            lambda emissions, t_emissions: true_model.marginal_log_prob(
                true_params, emissions, t_emissions[:, None]
            )
        )(emissions, t_emissions)

        if test_params is not None:
            test_marginal_lls = vmap(
                lambda emissions, t_emissions: test_model.marginal_log_prob(
                    test_params, emissions, t_emissions[:, None]
                )
            )(emissions, t_emissions)
    except Exception:
        true_marginal_lls = vmap(
            lambda emissions, t_emissions: true_model.marginal_log_prob(
                true_params, emissions
            )
        )(emissions, t_emissions)
        if test_params is not None:
            test_marginal_lls = vmap(
                lambda emissions, t_emissions: test_model.marginal_log_prob(
                    test_params, emissions
                )
            )(emissions, t_emissions)

    print("True marginal log probs", true_marginal_lls.sum())
    if test_params is not None:
        print("Test marginal log probs", test_marginal_lls.sum())
    print("True log prior", true_model.log_prior(true_params))
    if test_params is not None:
        print("Test log prior", test_model.log_prior(test_params))

    print("True log prior", true_model.log_prior(true_params))
    true_logjoint = true_model.log_prior(true_params) + true_marginal_lls.sum()
    print("True log joint", true_logjoint)
    plt.axhline(
        true_logjoint, color="k", linestyle=":", label=f"True ({true_logjoint:.2f})"
    )

    y_min, y_max = adjusted_y_limits(marginal_lls, true_logjoint)
    plt.ylim(y_min, y_max)

    # Decide whether to use a linear or symlog scale
    if should_use_linear_scale(y_min, y_max):
        plt.yscale("linear")
        setup_linear_scale(plt.gca())
    else:
        plt.yscale("symlog")
        setup_symlog_scale(plt.gca())

    plt.legend()
    plt.tight_layout()
    plt.show()


def adjusted_y_limits(marginal_lls, true_logjoint):
    y_min = min(min(marginal_lls), true_logjoint)
    y_max = max(max(marginal_lls), true_logjoint)
    return y_min, y_max


def should_use_linear_scale(y_min, y_max):
    # Decide whether to use linear or symlog based on the range of the data
    foo = y_max - y_min < 100
    # foo = np.log10(y_max) - np.log10(np.abs(y_min)) < 2  # Threshold of 3 orders of magnitude
    print("Should use linear scale?", foo)
    return foo


def setup_linear_scale(ax):
    # This function will configure the linear scale ticks and limits
    y_min, y_max = ax.get_ylim()
    # set axis limits to be larger than the data range in linear scale
    y_min = (1 - 0.1 * np.sign(y_min)) * y_min
    y_max = (1 + 0.1 * np.sign(y_max)) * y_max
    ax.set_ylim(y_min, y_max)


def setup_symlog_scale(ax):
    # This function will configure the symlog scale ticks and limits
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.yaxis.set_minor_locator(plt.NullLocator())
    y_min, y_max = ax.get_ylim()
    print("y_min, y_max", y_min, y_max)
    # set axis limits to be larger than the data range in log scale
    y_min = np.sign(y_min) * np.power(
        10, np.floor(np.log10(np.abs(y_min))) - np.sign(y_min)
    )
    y_max = np.sign(y_max) * np.power(
        10, np.ceil(np.log10(np.abs(y_max))) + np.sign(y_max)
    )
    print("y_min, y_max", y_min, y_max)
    ax.set_ylim(y_min, y_max)


def plot_generalization(
    true_model, true_params, test_model, test_params, t_emissions, key, num_samples=1
):
    num_timesteps = t_emissions.shape[0]

    keys = jr.split(key, num_samples)
    t_emissions_arrays = vmap(lambda key: jnp.arange(num_timesteps))(keys)

    # generate a new set of emissions data
    def sample_with_emissions(key, t_emissions):
        try:
            foo = true_model.sample(
                true_params,
                key,
                num_timesteps=num_timesteps,
                t_emissions=t_emissions[:, None],
            )
        except Exception:
            foo = true_model.sample(true_params, key, num_timesteps=num_timesteps)
        return foo

    # Use vmap to sample from lgssm with different t_emissions
    sample_func = vmap(sample_with_emissions)
    true_states, emissions = sample_func(keys, t_emissions_arrays)

    # Plot the true states and emissions
    fig, ax = plt.subplots()
    for n in range(num_samples):
        ax.plot(t_emissions_arrays[n], emissions[n], ls="--", label=f"Trajectory {n}")
    ax.set_title("New Data")
    ax.legend()

    # Run filtering and smoothing on the new emissions data
    plot_filtered_fits(
        true_model,
        true_params,
        test_model,
        test_params,
        emissions,
        t_emissions_arrays,
        num_samples=num_samples,
        true_states=true_states,
    )

    plot_smoothed_fits(
        true_model,
        true_params,
        test_model,
        test_params,
        emissions,
        t_emissions_arrays,
        num_samples=num_samples,
    )


def plot_filtered_fits(
    true_model,
    true_params,
    test_model,
    test_params,
    emissions,
    t_emissions,
    num_samples=1,
    true_states=None,
):
    state_dim = true_model.state_dim
    # run the filter w/ test_params on emissions data
    try:
        filtered_posteriors = vmap(
            lambda y, t: test_model.filter(test_params, y, t[:, None])
        )(emissions, t_emissions)
    except Exception:
        filtered_posteriors = vmap(lambda y, t: test_model.filter(test_params, y))(
            emissions, t_emissions
        )

    # from pdb import set_trace; set_trace()
    # print(filtered_emissions_means.shape)

    # compute the standard deviation of the filtered emissions distribution
    filtered_emissions_std = jnp.sqrt(
        jnp.array(
            [
                filtered_posteriors.filtered_covariances[:, :, i, i]
                for i in range(state_dim)
            ]
        )
    )

    print(filtered_posteriors.filtered_covariances.shape)
    print(filtered_posteriors.filtered_means.shape)

    # t_emissions = t_emissions.squeeze()
    spc = 3
    # make a sub figure with state_dim rows
    plt.figure(figsize=(10, 4))

    for i in range(state_dim):
        # switch to a new subplot
        plt.subplot(state_dim, 1, i + 1)
        plt.ylabel(f"State {i}")
        for n in range(num_samples):
            plt.plot(
                t_emissions[n],
                true_states[n, :, i] + spc * i,
                "--",
                color=f"C{n}",
                label="true",
            )
            ln = plt.plot(
                t_emissions[n],
                filtered_posteriors.filtered_means[n, :, i] + spc * i,
                color=f"C{n}",
                label="filtered",
            )[0]
            plt.fill_between(
                t_emissions[n],
                spc * i
                + filtered_posteriors.filtered_means[n, :, i]
                - 2 * filtered_emissions_std[n, i],
                spc * i
                + filtered_posteriors.filtered_means[n, :, i]
                + 2 * filtered_emissions_std[n, i],
                color=ln.get_color(),
                alpha=0.25,
            )
        plt.legend(loc="upper left")

        # plt.yscale("symlog")
    plt.xlabel("time")
    # plt.xlim(0, t_emissions_arrays[-1])
    plt.suptitle("True vs filtered states")
    plt.show()


def plot_smoothed_fits(
    true_model,
    true_params,
    test_model,
    test_params,
    emissions,
    t_emissions,
    num_samples=1,
):
    emission_dim = true_model.emission_dim
    # run the smoother w/ test_params on emissions data
    try:
        smoothed_emissions, smoothed_emissions_std = vmap(
            lambda y, t: test_model.posterior_predictive(test_params, y, t[:, None])
        )(emissions, t_emissions)
    except Exception:
        smoothed_emissions, smoothed_emissions_std = vmap(
            lambda y, t: test_model.posterior_predictive(test_params, y)
        )(emissions, t_emissions)

    # smoothed_emissions, smoothed_emissions_std = test_model.posterior_predictive(test_params, emissions, t_emissions)

    # t_emissions = t_emissions.squeeze()
    spc = 3
    plt.figure(figsize=(10, 4))
    for n in range(num_samples):
        for i in range(emission_dim):
            plt.plot(
                t_emissions[n],
                emissions[n, :, i] + spc * i,
                "--",
                color=f"C{n}",
                label="observed",
            )
            ln = plt.plot(
                t_emissions[n],
                smoothed_emissions[n, :, i] + spc * i,
                color=f"C{n}",
                label="smoothed",
            )[0]
            plt.fill_between(
                t_emissions[n],
                spc * i
                + smoothed_emissions[n, :, i]
                - 2 * smoothed_emissions_std[n, i],
                spc * i
                + smoothed_emissions[n, :, i]
                + 2 * smoothed_emissions_std[n, i],
                color=ln.get_color(),
                alpha=0.25,
            )
    plt.xlabel("time")
    # plt.xlim(0, t_emissions_arrays[-1])
    # plt.ylabel("true and predicted emissions")
    plt.legend(loc="upper left")
    plt.suptitle("True vs smoothed emissions")
    plt.show()


#################### Parameter plotting
# Useful utils for plotting parameters
def plot_scalar(true_val, test_val, title):
    plt.figure()
    plt.bar(["True", "Test"], [true_val, test_val])
    plt.title(title)
    plt.ylabel("Value")
    plt.show()


def plot_vector(true_val, test_val, title):
    x = jnp.arange(len(true_val))
    width = 0.35

    plt.figure()
    plt.bar(x - width / 2, true_val, width, label="True")
    plt.bar(x + width / 2, test_val, width, label="Test")
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


def plot_matrix(matrix, title):
    plt.figure()
    plt.imshow(matrix, cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.show()


# Compare to sets of parameters
def compare_cdlgssm_parameters(true_params, test_params):
    for level_key, inner_tuple in true_params._asdict().items():
        for param_key, true_value in inner_tuple._asdict().items():
            test_value = getattr(getattr(test_params, level_key), param_key)

            if test_value is None or true_value is None:
                continue

            title = f"{level_key} - {param_key}"
            if jnp.isscalar(true_value):
                plot_scalar(true_value, test_value, title)
            elif true_value.ndim == 1:
                plot_vector(true_value, test_value, title)
            elif true_value.ndim == 2:
                # plot_matrix(true_value, title + " (True Matrix)")
                # plot_matrix(test_value, title + " (Test Matrix)")

                # Vectorize matrices and plot
                true_vectorized = true_value.flatten()
                test_vectorized = test_value.flatten()
                plot_vector(true_vectorized, test_vectorized, title + " (Vectorized)")


# What is this?
def compare_parameters2(true_params, test_params):
    # List to store data for plotting
    plot_data = []
    labels = []

    for level_key, inner_tuple in true_params._asdict().items():
        for param_key, true_value in inner_tuple._asdict().items():
            test_value = getattr(getattr(test_params, level_key), param_key)

            if test_value is None or true_value is None:
                continue

            title = f"{level_key} - {param_key}"

            # Handle scalar, vector, and matrix types
            if np.isscalar(true_value):
                plot_data.append((true_value, test_value))
                labels.append(title)
            elif true_value.ndim == 1:
                for i, (t_val, tst_val) in enumerate(zip(true_value, test_value)):
                    plot_data.append((t_val, tst_val))
                    labels.append(f"{title} [{i}]")
            elif true_value.ndim == 2:
                # Vectorize matrices
                true_vectorized = true_value.flatten()
                test_vectorized = test_value.flatten()
                for i, (t_val, tst_val) in enumerate(
                    zip(true_vectorized, test_vectorized)
                ):
                    plot_data.append((t_val, tst_val))
                    labels.append(f"{title} (Vec) [{i}]")

    # Now plot all data in a single figure with horizontal bars
    true_vals, test_vals = zip(*plot_data)
    indices = np.arange(len(plot_data))
    width = 0.35

    fig, ax = plt.subplots()
    ax.barh(indices - width / 2, true_vals, width, label="True")
    ax.barh(indices + width / 2, test_vals, width, label="Test")

    ax.set_yticks(indices)
    ax.set_yticklabels(labels)
    ax.legend()

    plt.show()


# Plot parameter distributions
def plot_param_distributions(samples, true, init, name="", burn_in_frac=0.5):
    """
    Plots d_params horizontal box plots for the given d_params x N_samples matrix.

    Parameters:
    - samples: N_samples by d_params matrix of parameter samples.
    - true: d_params array of true parameter values.
    - init: d_params array of initial estimates.
    - name: Name of the parameter set.
    - burn_in_frac: Fraction of samples to discard as burn-in.

    Returns:
    - A matplotlib figure with d_params horizontal box plots.
    """
    d_params = samples.shape[1]
    fig, ax = plt.subplots(
        figsize=(10, d_params * 2)
    )  # Adjust figure size based on number of parameters

    # apply burn-in
    burn_in = int(burn_in_frac * samples.shape[0])
    samples = samples[:, burn_in:]

    # Create box plots
    ax.boxplot(samples, vert=False, patch_artist=True)

    # Set the y-axis labels to show parameter indices
    ax.set_yticks(range(1, d_params + 1))
    ax.set_yticklabels(["Parameter {}".format(i + 1) for i in range(d_params)])

    # Plot ground truth and estimates
    ax.scatter(
        init,
        range(1, d_params + 1),
        color="magenta",
        marker="o",
        s=100,
        label="Initial Estimate",
        zorder=3,
    )
    ax.scatter(
        true,
        range(1, d_params + 1),
        color="red",
        marker="x",
        s=100,
        label="Ground Truth",
        zorder=4,
    )

    plt.xlabel("Value")
    plt.ylabel("Parameters")
    plt.title("{} Parameter Distributions".format(name))
    plt.grid(True)
    plt.legend()
    plt.show()


# Plot all parameter samples and compare with truth and init
def plot_all_parameters(param_samples, true_params, init_params, burn_in_frac=0.5):
    """
    Plots the posterior distributions of all parameters.
    Burn-in is removed from the samples.
    """
    plot_param_distributions(
        param_samples.initial.mean,
        true_params.initial.mean,
        init_params.initial.mean,
        name="Initial mean",
        burn_in_frac=burn_in_frac,
    )
    plot_param_distributions(
        param_samples.initial.cov.reshape(param_samples.initial.cov.shape[0], -1),
        true_params.initial.cov.flatten(),
        init_params.initial.cov.flatten(),
        name="Initial cov",
        burn_in_frac=burn_in_frac,
    )
    plot_param_distributions(
        param_samples.dynamics.weights.reshape(
            param_samples.dynamics.weights.shape[0], -1
        ),
        true_params.dynamics.weights,
        init_params.dynamics.weights,
        name="Dynamics weights",
        burn_in_frac=burn_in_frac,
    )
    plot_param_distributions(
        param_samples.dynamics.bias,
        true_params.dynamics.bias,
        init_params.dynamics.bias,
        name="Dynamics bias",
        burn_in_frac=burn_in_frac,
    )
    plot_param_distributions(
        param_samples.dynamics.diffusion_cov.reshape(
            param_samples.dynamics.diffusion_cov.shape[0], -1
        ),
        true_params.dynamics.diffusion_cov.flatten(),
        init_params.dynamics.diffusion_cov.flatten(),
        name="Dynamics diffusion cov",
        burn_in_frac=burn_in_frac,
    )
    plot_param_distributions(
        param_samples.emissions.weights.reshape(
            param_samples.emissions.weights.shape[0], -1
        ),
        true_params.emissions.weights.flatten(),
        init_params.emissions.weights.flatten(),
        name="Emissions function weights",
        burn_in_frac=burn_in_frac,
    )
    plot_param_distributions(
        param_samples.emissions.bias.reshape(param_samples.emissions.bias.shape[0], -1),
        true_params.emissions.bias.flatten(),
        init_params.emissions.bias.flatten(),
        name="Emissions function bias",
        burn_in_frac=burn_in_frac,
    )
    plot_param_distributions(
        param_samples.emissions.cov.reshape(param_samples.emissions.cov.shape[0], -1),
        true_params.emissions.cov.flatten(),
        init_params.emissions.cov.flatten(),
        name="Emissions cov",
        burn_in_frac=burn_in_frac,
    )
