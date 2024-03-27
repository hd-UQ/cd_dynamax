import jax.numpy as jnp
from jax import vmap
import jax.random as jr
import numpy as np
from matplotlib.patches import Ellipse, transforms
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt
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

    cdict = {"red": tuple(reds), "green": tuple(greens), "blue": tuple(blues), "alpha": tuple(alphas)}

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
        (0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, edgecolor=edgecolor, **kwargs
    )

    scale_x = jnp.sqrt(cov[0, 0]) * n_std
    mean_x = mu[0]

    scale_y = jnp.sqrt(cov[1, 1]) * n_std
    mean_y = mu[1]

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

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
        ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"],
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


def plot_learning_curve(marginal_lls, true_model, true_params, test_model, test_params, emissions, t_emissions=None):
    plt.figure(figsize=(10, 6))
    plt.xlabel("Iteration")
    plt.ylabel("Marginal Joint Probability")
    plt.plot(marginal_lls, label=f"Estimated ({max(marginal_lls):.2f})")

    # Compute true_marginal_lls and true_logjoint here
    try:
        true_marginal_lls = vmap(
            lambda emissions, t_emissions: true_model.marginal_log_prob(true_params, emissions, t_emissions[:, None])
        )(emissions, t_emissions)

        test_marginal_lls = vmap(
            lambda emissions, t_emissions: test_model.marginal_log_prob(test_params, emissions, t_emissions[:, None])
        )(emissions, t_emissions)
    except:
        true_marginal_lls = vmap(lambda emissions, t_emissions: true_model.marginal_log_prob(true_params, emissions))(
            emissions, t_emissions
        )
        test_marginal_lls = vmap(lambda emissions, t_emissions: test_model.marginal_log_prob(test_params, emissions))(
            emissions, t_emissions
        )

    print("True marginal log probs", true_marginal_lls.sum())
    print("Test marginal log probs", test_marginal_lls.sum())
    print("True log prior", true_model.log_prior(true_params))
    print("Test log prior", test_model.log_prior(test_params))

    print("True log prior", true_model.log_prior(true_params))
    true_logjoint = true_model.log_prior(true_params) + true_marginal_lls.sum()
    print("True log joint", true_logjoint)
    plt.axhline(true_logjoint, color="k", linestyle=":", label=f"True ({true_logjoint:.2f})")

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
    y_min = np.sign(y_min) * np.power(10, np.floor(np.log10(np.abs(y_min))) - np.sign(y_min))
    y_max = np.sign(y_max) * np.power(10, np.ceil(np.log10(np.abs(y_max))) + np.sign(y_max))
    print("y_min, y_max", y_min, y_max)
    ax.set_ylim(y_min, y_max)


def plot_generalization(true_model, true_params, test_model, test_params, t_emissions, key, num_samples=1):
    num_timesteps = t_emissions.shape[0]

    keys = jr.split(key, num_samples)
    t_emissions_arrays = vmap(lambda key: jnp.arange(num_timesteps))(keys)

    # generate a new set of emissions data
    def sample_with_emissions(key, t_emissions):
        try:
            foo = true_model.sample(true_params, key, num_timesteps=num_timesteps, t_emissions=t_emissions[:, None])
        except:
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
    state_dim = true_model.state_dim
    emission_dim = true_model.emission_dim
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
        true_model, true_params, test_model, test_params, emissions, t_emissions_arrays, num_samples=num_samples
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
    emission_dim = true_model.emission_dim

    # run the filter w/ test_params on emissions data
    try:
        filtered_posteriors = vmap(lambda y, t: test_model.filter(test_params, y, t[:, None]))(emissions, t_emissions)
    except:
        filtered_posteriors = vmap(lambda y, t: test_model.filter(test_params, y))(emissions, t_emissions)

    # from pdb import set_trace; set_trace()
    # print(filtered_emissions_means.shape)

    # compute the standard deviation of the filtered emissions distribution
    filtered_emissions_std = jnp.sqrt(
        jnp.array([filtered_posteriors.filtered_covariances[:, :, i, i] for i in range(state_dim)])
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
                spc * i + filtered_posteriors.filtered_means[n, :, i] - 2 * filtered_emissions_std[n, i],
                spc * i + filtered_posteriors.filtered_means[n, :, i] + 2 * filtered_emissions_std[n, i],
                color=ln.get_color(),
                alpha=0.25,
            )
        plt.legend(loc="upper left")

        # plt.yscale("symlog")
    plt.xlabel("time")
    # plt.xlim(0, t_emissions_arrays[-1])
    plt.suptitle("True vs filtered states")
    plt.show()


def plot_smoothed_fits(true_model, true_params, test_model, test_params, emissions, t_emissions, num_samples=1):
    state_dim = true_model.state_dim
    emission_dim = true_model.emission_dim

    # run the smoother w/ test_params on emissions data
    try:
        smoothed_emissions, smoothed_emissions_std = vmap(
            lambda y, t: test_model.posterior_predictive(test_params, y, t[:, None])
        )(emissions, t_emissions)
    except:
        smoothed_emissions, smoothed_emissions_std = vmap(lambda y, t: test_model.posterior_predictive(test_params, y))(
            emissions, t_emissions
        )

    # smoothed_emissions, smoothed_emissions_std = test_model.posterior_predictive(test_params, emissions, t_emissions)

    # t_emissions = t_emissions.squeeze()
    spc = 3
    plt.figure(figsize=(10, 4))
    for n in range(num_samples):
        for i in range(emission_dim):
            plt.plot(
                t_emissions[n], emissions[n, :, i] + spc * i, "--", color=f"C{n}", label="observed"
            )
            ln = plt.plot(
                t_emissions[n],
                smoothed_emissions[n, :, i] + spc * i,
                color=f"C{n}",
                label="smoothed",
            )[0]
            plt.fill_between(
                t_emissions[n],
                spc * i + smoothed_emissions[n, :, i] - 2 * smoothed_emissions_std[n, i],
                spc * i + smoothed_emissions[n, :, i] + 2 * smoothed_emissions_std[n, i],
                color=ln.get_color(),
                alpha=0.25,
            )
    plt.xlabel("time")
    # plt.xlim(0, t_emissions_arrays[-1])
    # plt.ylabel("true and predicted emissions")
    plt.legend(loc="upper left")
    plt.suptitle("True vs smoothed emissions")
    plt.show()


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


def compare_parameters(true_params, test_params):
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
                for i, (t_val, tst_val) in enumerate(zip(true_vectorized, test_vectorized)):
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
