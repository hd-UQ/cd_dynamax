# Imports
import numpy as np
from matplotlib import pyplot as plt
import jax


def plot_lorenz_estimates(
    time_grid_all=None,
    time_grid_filter=None,
    time_grid_forecast=None,
    true_states=None,
    true_filtered_states=None,
    model_filtered_states=None,
    true_filtered_covariances=None,
    model_filtered_covariances=None,
    true_forecast_states=None,
    model_forecast_states=None,
    true_emissions_noisy=None,
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
    - true_forecast_states (ndarray, optional): Forecasted states using the true model, shape (T, n_states).
    - model_forecast_states (ndarray, optional): Forecasted states using the learned model, shape (T, n_states).
    - true_emissions_noisy (ndarray, optional): Noisy emissions from the true model, shape (T, n_emissions).
    - emission_function (LearnableLinear, optional): A function that computes emissions from states.
    - t_start (float, optional): The start time for plotting.
    - t_end (float, optional): The end time for plotting.
    """
    # Ensure at least one time grid is specified
    if time_grid_all is None and time_grid_filter is None and time_grid_forecast is None:
        raise ValueError("One of time_grid_all, time_grid_filter, or time_grid_forecast must be specified.")

    # Use the appropriate time grid
    if time_grid_all is not None:
        time_grid = np.squeeze(time_grid_all)
    elif time_grid_filter is not None:
        time_grid = np.squeeze(time_grid_filter)
    else:
        time_grid = np.squeeze(time_grid_forecast)

    # Determine indices based on t_start and t_end
    if t_start is not None:
        start_idx = np.searchsorted(time_grid, t_start)  # Find the index for the start time
    else:
        start_idx = 0  # Default to the beginning of the time grid

    if t_end is not None:
        end_idx = np.searchsorted(time_grid, t_end, side="right")  # Find the index for the end time
    else:
        end_idx = len(time_grid)  # Default to the end of the time grid

    # Subset the time and state arrays based on start and end indices
    time_grid = time_grid[start_idx:end_idx]
    if true_states is not None:
        true_states = true_states[start_idx:end_idx, :]
    if true_filtered_states is not None:
        true_filtered_states = true_filtered_states[start_idx:end_idx, :]
    if model_filtered_states is not None:
        model_filtered_states = model_filtered_states[start_idx:end_idx, :]
    if true_filtered_covariances is not None:
        true_filtered_covariances = true_filtered_covariances[start_idx:end_idx, :, :]
    if model_filtered_covariances is not None:
        model_filtered_covariances = model_filtered_covariances[start_idx:end_idx, :, :]
    if true_forecast_states is not None:
        true_forecast_states = true_forecast_states[start_idx:end_idx, :]
    if model_forecast_states is not None:
        model_forecast_states = model_forecast_states[start_idx:end_idx, :]
    if true_emissions_noisy is not None:
        true_emissions_noisy = true_emissions_noisy[start_idx:end_idx, :]

    # Determine the number of rows for subplots
    n_states = true_states.shape[1] if true_states is not None else 0
    n_emissions = (
        true_emissions_noisy.shape[1]
        if true_emissions_noisy is not None
        else (
            emission_function.f(true_states[0]).shape[0]
            if true_states is not None and emission_function is not None
            else 0
        )
    )
    n_rows = max(n_states, n_emissions)

    # Create a canvas with subplots for states and emissions
    fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(15, 2 * n_rows), sharex=True)

    # Use vmap to apply emission_function across the batch of states
    if emission_function is not None:
        vmap_emission = jax.vmap(emission_function.f, in_axes=(0, None, None))

    # Plot the states in the first column
    if true_states is not None:
        for i in range(n_states):
            # Plot true state values
            axes[i, 0].plot(
                time_grid, true_states[:, i], "--", color=f"C{i}", alpha=0.5, linewidth=2, label=f"True State {i}"
            )
            # Plot filtered state values from true model
            if true_filtered_states is not None:
                axes[i, 0].plot(
                    time_grid,
                    true_filtered_states[:, i],
                    color=f"C{i}",
                    linestyle="-",
                    alpha=0.7,
                    linewidth=2,
                    label=f"True Filtered State {i}",
                )
                # Plot uncertainty bounds if available
                if true_filtered_covariances is not None:
                    axes[i, 0].fill_between(
                        time_grid,
                        true_filtered_states[:, i] - np.sqrt(true_filtered_covariances[:, i, i]),
                        true_filtered_states[:, i] + np.sqrt(true_filtered_covariances[:, i, i]),
                        color=f"C{i}",
                        alpha=0.3,
                    )
            # Plot filtered state values from learned model
            if model_filtered_states is not None:
                axes[i, 0].plot(
                    time_grid,
                    model_filtered_states[:, i],
                    color=f"C{i}",
                    linestyle="-",
                    alpha=0.7,
                    linewidth=2,
                    label=f"Model Filtered State {i}",
                )
                # Plot uncertainty bounds if available
                if model_filtered_covariances is not None:
                    axes[i, 0].fill_between(
                        time_grid,
                        model_filtered_states[:, i] - np.sqrt(model_filtered_covariances[:, i, i]),
                        model_filtered_states[:, i] + np.sqrt(model_filtered_covariances[:, i, i]),
                        color=f"C{i}",
                        alpha=0.3,
                    )
            # Plot forecast state values from true model
            if true_forecast_states is not None:
                axes[i, 0].plot(
                    time_grid,
                    true_forecast_states[:, i],
                    color=f"C{i}",
                    linestyle="-.",
                    alpha=0.7,
                    linewidth=2,
                    label=f"True Forecast State {i}",
                )
            # Plot forecast state values from learned model
            if model_forecast_states is not None:
                axes[i, 0].plot(
                    time_grid,
                    model_forecast_states[:, i],
                    color=f"C{i}",
                    linestyle=":",
                    alpha=0.7,
                    linewidth=2,
                    label=f"Model Forecast State {i}",
                )

            # Set y-axis label and add legend
            axes[i, 0].set_ylabel(f"State {i}")
            axes[i, 0].legend(loc="upper right")

    # Plot the emissions in the second column
    for i in range(n_emissions):
        # Plot noisy true emissions if available
        if true_emissions_noisy is not None:
            axes[i, 1].plot(
                time_grid,
                true_emissions_noisy[:, i],
                "x",
                color=f"C{i}",
                alpha=0.5,
                ms=3,
                label=f"Noisy True Emission {i}",
            )

        # Plot filtered emission values from true model
        if emission_function is not None and true_filtered_states is not None:
            true_filtered_emissions = vmap_emission(true_filtered_states, None, None)
            axes[i, 1].plot(
                time_grid,
                true_filtered_emissions[:, i],
                color=f"C{i}",
                linestyle="-",
                alpha=0.7,
                linewidth=2,
                label=f"True Filtered Emission {i}",
            )
        # Plot filtered emission values from learned model
        if emission_function is not None and model_filtered_states is not None:
            model_filtered_emissions = vmap_emission(model_filtered_states, None, None)
            axes[i, 1].plot(
                time_grid,
                model_filtered_emissions[:, i],
                color=f"C{i}",
                linestyle="-",
                alpha=0.7,
                linewidth=2,
                label=f"Model Filtered Emission {i}",
            )
        # Plot forecast emission values from true model
        if emission_function is not None and true_forecast_states is not None:
            true_forecast_emissions = vmap_emission(true_forecast_states, None, None)
            axes[i, 1].plot(
                time_grid,
                true_forecast_emissions[:, i],
                color=f"C{i}",
                linestyle="-.",
                alpha=0.7,
                linewidth=2,
                label=f"True Forecast Emission {i}",
            )
        # Plot forecast emission values from learned model
        if emission_function is not None and model_forecast_states is not None:
            model_forecast_emissions = vmap_emission(model_forecast_states, None, None)
            axes[i, 1].plot(
                time_grid,
                model_forecast_emissions[:, i],
                color=f"C{i}",
                linestyle=":",
                alpha=0.7,
                linewidth=2,
                label=f"Model Forecast Emission {i}",
            )

        # Set y-axis label and add legend
        axes[i, 1].set_ylabel(f"Emission {i}")
        axes[i, 1].legend(loc="upper right")

    # Plot a vertical line to indicate the switch between filtered and forecasted states if both are provided
    if time_grid_filter is not None and time_grid_forecast is not None:
        switch_time = time_grid_filter[-1]
        for i in range(n_rows):
            axes[i, 0].axvline(x=switch_time, color="k", linestyle="--", linewidth=1, label="Filter/Forecast Boundary")
            axes[i, 1].axvline(x=switch_time, color="k", linestyle="--", linewidth=1)

    # Set x-axis label only on the bottom subplots
    if n_rows > 0:
        for ax in axes[-1, :]:
            ax.set_xlabel("Time $t$")

    # Set a super title for the entire figure
    plt.suptitle("True vs Filtered vs Forecast States and Emissions: True and Learned Models")

    # Adjust layout to prevent overlap and show the plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust space for the title
    plt.show()