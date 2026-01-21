# JAX imports
import jax
import jax.numpy as jnp
import jax.random as jr

# Other imports
import itertools

# CD-Dynamax imports
from cd_dynamax import EnKFHyperParams, EKFHyperParams, UKFHyperParams

# Plotting imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import numpyro


# -----------------------
# Data-generation helpers
# -----------------------
def sample_t_emissions(
    start=0.0, stop=10.0, dt=0.1, regular=True, key=None, check=True, verbose=True
):
    """
    Generate emission times between [start, stop) using different sampling styles.

    Args:
        start (float): Start time (inclusive).
        stop (float): Stop time (exclusive).
        dt (float): Average spacing between points.
        regular (bool): If True, use regular spacing. If False, use uniform random sampling.
        key (jax.random.PRNGKey): Required for regular=False (random sampling).

    Returns:
        jnp.ndarray: Times of shape (N, 1).
    """
    if regular:
        t_emissions = jnp.arange(start, stop, dt).reshape(-1, 1)
    else:
        if key is None:
            raise ValueError("`key` must be provided for uniform sampling.")
        N = int(jnp.ceil((stop - start) / dt))
        key, subkey = jax.random.split(key)
        t_emissions = jnp.sort(
            jax.random.uniform(subkey, (N,), minval=start, maxval=stop)
        )
        t_emissions = t_emissions.reshape(-1, 1)

    # ---- Post-checks ----
    if check:
        if t_emissions.size > 1:
            diffs = jnp.diff(t_emissions.squeeze())
            min_step = diffs.min().astype(float)
            is_sorted = jnp.all(diffs >= 0.0)
            has_dupes = jnp.any(diffs == 0.0)

            assert is_sorted, "t_emissions is not sorted."
            assert not has_dupes, "t_emissions contains duplicate points."

            if verbose:
                print(
                    f"[make_t_emissions] Generated {t_emissions.shape[0]} points from {start} to {stop} with avg step ~{dt:.4f}"
                )
            print(f"[make_t_emissions] Smallest time step = {min_step:.6f}")
        else:
            print("[make_t_emissions] Only one or zero points generated.")

    return t_emissions


# ------------------------
# MCMC Plot helpers
# ------------------------
def plot_forest(mcmc_samples, param_name="beta", max_params=50, figsize=(10, 6)):
    """
    Forest plot of posterior means + 95% CIs for up to max_params.
    Works for 1D flattened parameters.
    """
    arr = np.asarray(mcmc_samples[param_name])  # (num_samples, ..., D?)
    flat = arr.reshape(arr.shape[0], -1)  # (num_samples, P)
    P = flat.shape[1]
    idx = np.arange(min(P, max_params))

    means = flat.mean(axis=0)[idx]
    lowers = np.percentile(flat, 2.5, axis=0)[idx]
    uppers = np.percentile(flat, 97.5, axis=0)[idx]

    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(
        means,
        idx,
        xerr=[means - lowers, uppers - means],
        fmt="o",
        color="tab:blue",
        ecolor="gray",
        alpha=0.8,
    )
    ax.set_yticks(idx)
    ax.set_yticklabels([f"{param_name}[{i}]" for i in idx])
    ax.axvline(0, color="k", linestyle="--", lw=1)
    ax.set_title(f"Posterior of {param_name} (first {len(idx)} params)")
    plt.tight_layout()
    return fig


def plot_violin(mcmc_samples, param_name="beta", max_params=50, figsize=(12, 6)):
    """
    Violin plots for marginals of posterior. Good for overview.
    """
    arr = np.asarray(mcmc_samples[param_name])
    flat = arr.reshape(arr.shape[0], -1)
    P = flat.shape[1]
    idx = np.arange(min(P, max_params))

    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(data=flat[:, idx], inner="quartile", orient="v", ax=ax)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{i}" for i in idx], rotation=90)
    ax.set_xlabel(f"{param_name} index")
    ax.set_ylabel("Posterior samples")
    ax.set_title(f"Posterior violin plots (first {len(idx)} params)")
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(
    mcmc_samples, param_name="beta", figsize=(8, 6), max_params=50
):
    """
    Heatmap of posterior correlations between parameters.
    Subsamples if too many parameters.
    """
    arr = np.asarray(mcmc_samples[param_name])
    flat = arr.reshape(arr.shape[0], -1)
    P = flat.shape[1]

    if P > max_params:
        flat = flat[:, :max_params]
        P = max_params

    corr = np.corrcoef(flat.T)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        ax=ax,
        xticklabels=[f"{i}" for i in range(P)],
        yticklabels=[f"{i}" for i in range(P)],
    )
    ax.set_title(f"Posterior correlation heatmap ({param_name}, first {P})")
    plt.tight_layout()
    return fig


def plot_pca_scatter(mcmc_samples, param_name="beta", figsize=(6, 6)):
    """
    PCA projection of posterior samples (first 2 PCs).
    """
    arr = np.asarray(mcmc_samples[param_name])
    flat = arr.reshape(arr.shape[0], -1)

    pca = PCA(n_components=2)
    proj = pca.fit_transform(flat)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(proj[:, 0], proj[:, 1], alpha=0.4, s=10, color="tab:blue")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA of posterior samples ({param_name})")
    plt.tight_layout()
    return fig


# ------------------------
# Plot helpers
# ------------------------
def plot_coeff_heatmaps(W_true, W_learned, exponents=None, relative_error=False):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    vlim = jnp.max(jnp.abs(jnp.concatenate([W_true, W_learned])))
    norm = mcolors.TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)

    im0 = axes[0].imshow(W_true, aspect="auto", cmap="seismic", norm=norm)
    axes[0].set_title("True weights")
    axes[0].set_xlabel("Term index")
    axes[0].set_ylabel("State index")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(W_learned, aspect="auto", cmap="seismic", norm=norm)
    axes[1].set_title("Learned weights (SVI median)")
    axes[1].set_xlabel("Term index")
    axes[1].set_ylabel("State index")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    abs_err = jnp.abs(W_learned - W_true)  # / jnp.maximum(jnp.abs(W_true), 1e-8)
    err_title = "Relative error" if relative_error else "Absolute error"
    if relative_error:
        abs_err /= jnp.maximum(jnp.abs(W_true), 1e-8)
    im2 = axes[2].imshow(abs_err, aspect="auto", cmap="viridis")
    axes[2].set_title(err_title)
    axes[2].set_xlabel("Term index")
    axes[2].set_ylabel("State index")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    return fig


def plot_drift_field(
    f_true,
    f_learned,
    f_learned_sd=None,  # callable or None
    x1_range=(-3.0, 3.0),
    x2_range=(-3.2, 3.2),
    num_points=50,
    return_rmse=False,
    relative_error=False,
):
    """
    Plot true vs learned drift fields (2D state space).
    Optionally include learned uncertainty (stddev).

    Args:
        f_true: callable f(x) -> (2,) array, true drift function
        f_learned: callable f(x) -> (2,) array, learned drift function
        f_learned_sd: optional callable f(x) -> (2,) array (stddev per output dim)
        x1_range: tuple (low, high) for x1 axis
        x2_range: tuple (low, high) for x2 axis
        num_points: number of grid points per axis
    """
    x1 = jnp.linspace(*x1_range, num_points)
    x2 = jnp.linspace(*x2_range, num_points)
    X1, X2 = jnp.meshgrid(x1, x2, indexing="ij")
    grid_points = jnp.stack([X1.ravel(), X2.ravel()], axis=-1)

    # Evaluate true, learned drift and optional stddev
    f_true_vals = jax.vmap(f_true)(grid_points)  # (N, 2)
    f_learned_vals = jax.vmap(f_learned)(grid_points)  # (N, 2)

    if f_learned_sd is not None:
        f_learned_sd_vals = jax.vmap(f_learned_sd)(grid_points)  # .squeeze(1)  # (N, 2)
        if f_learned_sd_vals.ndim == 3 and f_learned_sd_vals.shape[1] == 1:
            f_learned_sd_vals = f_learned_sd_vals.squeeze(1)
        f1_sd = f_learned_sd_vals[:, 0].reshape(num_points, num_points)
        f2_sd = f_learned_sd_vals[:, 1].reshape(num_points, num_points)
    else:
        f1_sd = f2_sd = None

    # Split into components and reshape
    f1_true = f_true_vals[:, 0].reshape(num_points, num_points)
    f2_true = f_true_vals[:, 1].reshape(num_points, num_points)
    f1_learned = f_learned_vals[:, 0].reshape(num_points, num_points)
    f2_learned = f_learned_vals[:, 1].reshape(num_points, num_points)

    f1_err = jnp.abs(f1_learned - f1_true)
    f2_err = jnp.abs(f2_learned - f2_true)

    if relative_error:
        f1_err /= jnp.abs(f1_true) + 1e-6
        f2_err /= jnp.abs(f2_true) + 1e-6

    # Color normalization
    vlim1 = jnp.max(jnp.abs(jnp.concatenate([f1_true.ravel(), f1_learned.ravel()])))
    vlim2 = jnp.max(jnp.abs(jnp.concatenate([f2_true.ravel(), f2_learned.ravel()])))

    # Subplot grid: add uncertainty column if available
    ncols = 4 if f_learned_sd is not None else 3
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 8), constrained_layout=True)

    # f1 row
    im0 = axes[0, 0].imshow(
        f1_true.T,
        origin="lower",
        extent=(*x1_range, *x2_range),
        cmap="seismic",
        vmin=-vlim1,
        vmax=vlim1,
        aspect="auto",
    )
    axes[0, 0].set_title("f1 true")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(
        f1_learned.T,
        origin="lower",
        extent=(*x1_range, *x2_range),
        cmap="seismic",
        vmin=-vlim1,
        vmax=vlim1,
        aspect="auto",
    )
    axes[0, 1].set_title("f1 learned")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[0, 2].imshow(
        f1_err.T,
        origin="lower",
        extent=(*x1_range, *x2_range),
        cmap="viridis",
        aspect="auto",
    )
    axes[0, 2].set_title("f1 error")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    if f1_sd is not None:
        im3 = axes[0, 3].imshow(
            f1_sd.T,
            origin="lower",
            extent=(*x1_range, *x2_range),
            cmap="magma",
            aspect="auto",
        )
        axes[0, 3].set_title("f1 stddev")
        fig.colorbar(im3, ax=axes[0, 3], fraction=0.046, pad=0.04)

    # f2 row
    im4 = axes[1, 0].imshow(
        f2_true.T,
        origin="lower",
        extent=(*x1_range, *x2_range),
        cmap="seismic",
        vmin=-vlim2,
        vmax=vlim2,
        aspect="auto",
    )
    axes[1, 0].set_title("f2 true")
    fig.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im5 = axes[1, 1].imshow(
        f2_learned.T,
        origin="lower",
        extent=(*x1_range, *x2_range),
        cmap="seismic",
        vmin=-vlim2,
        vmax=vlim2,
        aspect="auto",
    )
    axes[1, 1].set_title("f2 learned")
    fig.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im6 = axes[1, 2].imshow(
        f2_err.T,
        origin="lower",
        extent=(*x1_range, *x2_range),
        cmap="viridis",
        aspect="auto",
    )
    axes[1, 2].set_title("f2 error")
    fig.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)

    if f2_sd is not None:
        im7 = axes[1, 3].imshow(
            f2_sd.T,
            origin="lower",
            extent=(*x1_range, *x2_range),
            cmap="magma",
            aspect="auto",
        )
        axes[1, 3].set_title("f2 stddev")
        fig.colorbar(im7, ax=axes[1, 3], fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(False)

    if return_rmse:
        rmse = jnp.sqrt(jnp.mean((f_learned_vals - f_true_vals) ** 2))
        return fig, rmse
    else:
        return fig


def plot_particle_diagnostics(
    t_emissions,  # shape (T,) time vector
    x_ens_filtered,  # shape (T, N, D) ensemble after update
    x_ens_predicted,  # shape (T, N, D) ensemble before update (forecast)
    observations,  # shape (T, D_obs) -- assume D_obs == D for now
    start_idx=0,
    stop_idx=None,
    figsize=(12, 6),
):
    """
    Plot particle forecasts and updates over time, along with observations.
    One subplot per state dimension.

    Args:
        t_emissions: (T,) time vector
        x_ens_filtered: (T, N, D) ensemble after update
        x_ens_predicted: (T, N, D) ensemble before update (forecast)
        observations: (T, D) true observations
        start_idx: int, start timestep
        stop_idx: int, stop timestep (exclusive)
        figsize: tuple, figure size
    """
    T, N, D = x_ens_filtered.shape
    if stop_idx is None:
        stop_idx = T

    t_range = t_emissions[start_idx:stop_idx]

    fig, axes = plt.subplots(D, 1, figsize=figsize, sharex=True)
    if D == 1:
        axes = [axes]

    for d in range(D):
        ax = axes[d]

        # Plot observations
        ax.plot(
            t_range,
            observations[start_idx:stop_idx, d],
            "rx",
            label="obs",
            markersize=20,
        )

        # Plot filtered and forecast ensembles
        for i, t in enumerate(range(start_idx, stop_idx - 1)):
            t0 = t_emissions[t]
            t1 = t_emissions[t + 1]

            xf = x_ens_filtered[t, :, d]
            xp = x_ens_predicted[t + 1, :, d]

            ax.scatter(
                np.full(N, t0),
                xf,
                color="tab:blue",
                alpha=0.5,
                s=15,
                label="filtered" if (d == 0 and i == 0) else "",
            )
            ax.scatter(
                np.full(N, t1),
                xp,
                color="tab:orange",
                alpha=0.5,
                s=15,
                label="forecast" if (d == 0 and i == 0) else "",
            )

            # connect filtered[t] to forecast[t+1] with faint lines
            for n in range(N):
                ax.plot(
                    [t0, t1], [xf[n], xp[n]], color="gray", alpha=0.2, linewidth=0.8
                )

        ax.set_ylabel(f"state {d}")
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("time")
    axes[0].legend(loc="upper right")
    fig.suptitle("Particle diagnostics: forecasts vs updates")

    return fig


def _finite_1d(a):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    return a.ravel()


def _as_2d_TD(x):
    """Assume (T,D) or (T,) -> return (T,D)."""
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim == 2:
        return x
    raise ValueError(f"Expected 1D or 2D, got {x.shape}.")


def _series_colors(series_keys):
    """Deterministic color mapping: one color per series key, consistent across subplots."""
    base = (
        plt.rcParams["axes.prop_cycle"]
        .by_key()
        .get("color", ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"])
    )
    colors = {}
    for i, k in enumerate(series_keys):
        colors[k] = base[i % len(base)]
    return colors


def plot_traj_kde(
    series_dict,
    per_dim=True,
    pool_dims=False,
    dim_names=None,
    gridsize=512,
    bw_method="scott",  # or "silverman" or float
    xlim=None,  # None -> auto from pooled data; or (lo, hi)
    figsize=(9, 5),
    linewidth=2.0,
    alpha=0.85,
    title=None,
):
    """
    KDE comparison for trajectories shaped (T, D) (or (T,) -> (T,1)).

    per_dim=False:
        One axis, one KDE per series (all dims flattened).
    per_dim=True, pool_dims=True:
        One axis, one KDE per series (pool across dims).
    per_dim=True, pool_dims=False:
        Subplot per dimension; overlay series per subplot.

    Legend:
        Shown once, outside the right edge; exactly one entry per series (series_dict keys).
        Colors are consistent for each series across all subplots/modes.
    """
    # Normalize to (T, D)
    series_2d = {k: _as_2d_TD(v) for k, v in series_dict.items()}
    Ds = {arr.shape[1] for arr in series_2d.values()}

    # Fixed, consistent colors per series (respect input order)
    keys = list(series_dict.keys())
    color_map = _series_colors(keys)

    # ---------- Case 1: flatten (single axis) ----------
    if not per_dim:
        data = {k: _finite_1d(v) for k, v in series_2d.items()}
        pooled = (
            np.concatenate([d for d in data.values() if d.size > 0])
            if data
            else np.array([])
        )
        if pooled.size == 0:
            raise ValueError("All inputs empty/NaN.")
        lo, hi = (np.min(pooled), np.max(pooled)) if xlim is None else xlim
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        xs = np.linspace(lo, hi, gridsize)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        for label, d in data.items():
            color = color_map[label]
            if d.size < 2 or np.std(d) == 0:
                ax.plot(
                    [float(d.mean()) if d.size else 0] * 2,
                    [0, 1],
                    color=color,
                    alpha=0.4,
                )
            else:
                kde = gaussian_kde(d, bw_method=bw_method)
                ax.plot(xs, kde(xs), color=color, linewidth=linewidth, alpha=alpha)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        if title:
            ax.set_title(title)

        # Build legend once, outside, using series keys and their colors
        handles = [plt.Line2D([0], [0], color=color_map[k], lw=linewidth) for k in keys]
        fig.subplots_adjust(right=0.80)
        fig.legend(
            handles, keys, loc="center left", bbox_to_anchor=(0.82, 0.5), frameon=False
        )
        return fig

    # ---------- Per-dim modes require same D across series ----------
    if len(Ds) != 1:
        shapes = {k: v.shape for k, v in series_2d.items()}
        raise ValueError(
            f"When per_dim=True, all series must share the same D. Got Ds={Ds}, shapes={shapes}."
        )
    D = Ds.pop()

    # ---------- Case 2: per-dim with dims pooled (single axis) ----------
    if pool_dims:
        pooled_per_series = {k: _finite_1d(v) for k, v in series_2d.items()}
        pooled_all = np.concatenate(
            [v for v in pooled_per_series.values() if v.size > 0]
        )
        lo, hi = (np.min(pooled_all), np.max(pooled_all)) if xlim is None else xlim
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        xs = np.linspace(lo, hi, gridsize)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        for label, d in pooled_per_series.items():
            color = color_map[label]
            if d.size < 2 or np.std(d) == 0:
                ax.plot(
                    [float(d.mean()) if d.size else 0] * 2,
                    [0, 1],
                    color=color,
                    alpha=0.4,
                )
            else:
                kde = gaussian_kde(d, bw_method=bw_method)
                ax.plot(xs, kde(xs), color=color, linewidth=linewidth, alpha=alpha)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        if title:
            ax.set_title(title + " (dims pooled)" if title else "Dims pooled")

        handles = [plt.Line2D([0], [0], color=color_map[k], lw=linewidth) for k in keys]
        fig.subplots_adjust(right=0.80)
        fig.legend(
            handles, keys, loc="center left", bbox_to_anchor=(0.82, 0.5), frameon=False
        )
        return fig

    # ---------- Case 3: per-dim with subplots (what you want) ----------
    # Auto-layout: up to 3 columns
    ncols = min(3, D)
    nrows = int(np.ceil(D / ncols))

    # Scale fig to grid & leave space for legend
    base_w, base_h = figsize
    fig_w = max(base_w, 3.8 * ncols)
    fig_h = max(base_h, 2.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    axes_flat = axes.ravel()

    if dim_names is not None and len(dim_names) != D:
        raise ValueError(f"dim_names length {len(dim_names)} != D {D}.")

    # Precompute per-dim x ranges from pooled dim data
    xs_list = []
    for d in range(D):
        dim_all = np.concatenate([_finite_1d(arr[:, d]) for arr in series_2d.values()])
        lo, hi = (np.min(dim_all), np.max(dim_all)) if xlim is None else xlim
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        xs_list.append(np.linspace(lo, hi, gridsize))

    # Plot each dimension (consistent colors, no labels on lines)
    for d in range(D):
        ax = axes_flat[d]
        xs = xs_list[d]
        for label, arr in series_2d.items():
            color = color_map[label]
            dvals = _finite_1d(arr[:, d])
            if dvals.size < 2 or np.std(dvals) == 0:
                if dvals.size:
                    ax.plot([float(dvals.mean())] * 2, [0, 1], color=color, alpha=0.4)
                continue
            kde = gaussian_kde(dvals, bw_method=bw_method)
            ax.plot(xs, kde(xs), color=color, linewidth=linewidth, alpha=alpha)

        name = f"Dim {d}" if dim_names is None else dim_names[d]
        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.set_title(f"KDE — {name}")

    # Remove unused axes (if D < nrows*ncols)
    for j in range(D, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    if title:
        fig.suptitle(title, y=0.995)

    # One legend outside, with exactly len(series_dict) entries, matching colors
    handles = [plt.Line2D([0], [0], color=color_map[k], lw=linewidth) for k in keys]
    fig.subplots_adjust(right=0.80)  # reserve space
    fig.legend(
        handles, keys, loc="center left", bbox_to_anchor=(0.82, 0.5), frameon=False
    )

    fig.tight_layout(rect=[0, 0, 0.80, 0.96] if title else [0, 0, 0.80, 0.98])
    return fig


def plot_loss_curve(
    loss_values,
    loss_label="ELBO Loss",
    baseline_value=None,
    baseline_label="Baseline",
    figsize=(8, 5),
    title="SVI Loss Curve",
):
    """
    Plot loss values over iterations.

    Args:
        loss_values: (num_iters,) array of loss values
        figsize: tuple, figure size
        title: str, plot title

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(loss_values, color="tab:blue", linewidth=2.0, label=loss_label)
    if baseline_value is not None:
        # dotted horizontal line
        ax.axhline(
            baseline_value,
            color="tab:orange",
            linestyle=":",
            linewidth=2.0,
            label=baseline_label,
        )
        ax.legend()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig


def plot_simulated_data(t=None, states=None, emissions=None):
    """
    Plot simulated states and emissions over time.
    Each state dimension gets its own subplot.

    Args:
        t: (T,) time vector
        states: (T, state_dim) true states
        emissions: (T, emission_dim) observed emissions
            Assume emission_dim <= state_dim and emissions correspond to first N states.

    Returns:
        matplotlib Figure
    """

    # Validate inputs
    if states is None and emissions is None:
        raise ValueError("At least one of states or emissions must be provided.")

    # Make a single figure with subplots for each state and emission overlayed
    # Assume that the N emissions correspond to the the first N states

    n_rows = states.shape[1] if states is not None else emissions.shape[1]
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        if states is not None:
            ax.plot(t, states[:, i], label="State", color="tab:blue")
        if emissions is not None and i < emissions.shape[1]:
            ax.scatter(
                t,
                emissions[:, i],
                label="Emission",
                color="tab:orange",
                s=10,
                alpha=0.6,
            )
        ax.set_ylabel(f"Dim {i}")
        ax.grid(True, linestyle="--", alpha=0.3)
        if i == 0:
            ax.legend()

    axes[-1].set_xlabel("Time")
    fig.suptitle("States and Emissions", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def all_multi_indices(d, max_degree):
    out = []
    for total in range(max_degree + 1):
        for exps in itertools.product(range(total + 1), repeat=d):
            if sum(exps) == total:
                out.append(exps)
    return out


def build_exponents(input_dim, max_degree):
    idx = all_multi_indices(input_dim, max_degree)
    return jnp.array(idx, dtype=jnp.int32)


def eval_monomials(x, exponents):
    return jnp.prod(jnp.where(exponents == 0, 1.0, x[None, :] ** exponents), axis=1)


def poly_drift(x, weights, exponents):
    phi = eval_monomials(x, exponents)
    return weights @ phi


# -----------------------------
# Neural Network Drift Function
# -----------------------------
def make_nn_init_dict(key, state_dim, hidden_dims, eps=1):
    """
    Create an initialization dictionary for NN weights/biases.
    Uses He-style scaling for weights, zeros for biases.
    """
    params = {}
    keys = iter(jr.split(key, len(hidden_dims) * 2 + 2))  # enough rng keys

    in_dim = state_dim
    for layer_idx, out_dim in enumerate(hidden_dims):
        kW = next(keys)
        next(keys)  # Skip bias key
        params[f"W{layer_idx}"] = (
            eps * jnp.sqrt(2.0 / in_dim) * jr.normal(kW, (in_dim, out_dim))
        )
        params[f"b{layer_idx}"] = jnp.zeros((out_dim,))
        in_dim = out_dim

    # Output layer
    kW_out = next(keys)
    next(keys)  # Skip bias key
    params["W_out"] = (
        eps * jnp.sqrt(2.0 / in_dim) * jr.normal(kW_out, (in_dim, state_dim))
    )
    params["b_out"] = jnp.zeros((state_dim,))

    return params


# ------------------------
# Filter hyperparameter helper
# ------------------------
def make_filter_hyperparams(cfg):
    diffeqsolve_settings = {
        # 'solver': eval(cfg.diffeqsolve_settings['solver']),
        # 'stepsize_controller': eval(cfg.diffeqsolve_settings['stepsize_controller']),
        # 'adjoint': eval(cfg.diffeqsolve_settings['adjoint']),
        # 'dt0': cfg.diffeqsolve_settings['dt0'],
        # 'tol_vbt': cfg.diffeqsolve_settings['tol_vbt'],
        "max_steps": cfg.diffeqsolve_max_steps,
    }
    if cfg.filter_type == "EnKF":
        return EnKFHyperParams(
            N_particles=cfg.N_particles,
            state_order=cfg.state_order,
            diffeqsolve_settings=diffeqsolve_settings,
            cov_rescaling=cfg.cov_rescaling,
            inflation_delta=cfg.inflation_delta,
        )
    elif cfg.filter_type == "EKF":
        return EKFHyperParams(
            state_order=cfg.state_order,
            diffeqsolve_settings=diffeqsolve_settings,
        )
    elif cfg.filter_type == "UKF":
        return UKFHyperParams(
            state_order=cfg.state_order,
            diffeqsolve_settings=diffeqsolve_settings,
        )
    else:
        raise ValueError(f"Unknown filter type: {cfg.filter_type}")


# ------------------------
# Inference/model helpers in numpyro
# ------------------------
def get_or_sample(name, dist_obj, value):
    return (
        numpyro.sample(name, dist_obj)
        if value is None
        else numpyro.deterministic(name, value)
    )


# ------------------------
# SVI Plot helpers
# ------------------------
# ------------------------
# Helpers (new plotting)
# ------------------------
### NEW: generic unrolling + scatter plotting by groups
def _flatten_param(x):
    """Return a 1D float array for any leaf (scalar/array)."""
    x = jnp.asarray(x)
    return x.reshape(-1)


def plot_param_recovery(
    true_values,
    groups=None,
    title="Parameter recovery",
    guide=None,
    svi_result=None,
    mcmc=None,
    prng_key=None,
):
    """
    Generic parameter recovery plotter.

    Modes:
    - Variational (SVI): provide `guide` and `svi_result`.
    - MCMC: provide `mcmc`.

    Args:
      true_values : dict of true parameter values
      groups      : list of (group_name, [param_keys...]) to organize subplots
      guide       : AutoGuide (for SVI mode)
      svi_result  : result object from SVI.run (for SVI mode)
      mcmc        : numpyro.infer.MCMC object (for MCMC mode)
      prng_key    : optional jax.random.PRNGKey (used for sampling fallback)
    """
    if (guide is None or svi_result is None) and mcmc is None:
        raise ValueError("Must provide either (guide + svi_result) or mcmc.")

    if groups is None:
        groups = [
            ("weights", ["weights"]),
            ("bias", ["bias"]),
            ("noise", ["emission_sd", "diffusion_coeff"]),
        ]

    xs, ys, names, yerr_low, yerr_high = [], [], [], [], []

    # --------------------------
    # Case 1: SVI (variational)
    # --------------------------
    if guide is not None and svi_result is not None:
        learned = guide.median(svi_result.params)

        # Try quantiles
        q_dict = None
        try:
            q_dict = guide.quantiles(svi_result.params, (0.05, 0.95))
        except Exception:
            q_dict = None

        # Optionally fallback: sample posterior
        sample_dict = None
        if q_dict is None:
            try:
                if prng_key is None:
                    prng_key = jax.random.PRNGKey(0)
                samples = guide.sample_posterior(
                    prng_key, svi_result.params, sample_shape=(200,)
                )
                sample_dict = {
                    k: (v.mean(axis=0), v.std(axis=0)) for k, v in samples.items()
                }
            except Exception:
                sample_dict = None

        def get_estimates(k, p_flat):
            if q_dict is not None and k in q_dict:
                q_low = _flatten_param(q_dict[k][0])
                q_high = _flatten_param(q_dict[k][1])
                return p_flat - q_low, q_high - p_flat
            elif sample_dict is not None and k in sample_dict:
                std_flat = _flatten_param(sample_dict[k][1])
                return std_flat, std_flat
            return None, None

        extractor = lambda k: (learned[k], get_estimates)

    # --------------------------
    # Case 2: MCMC
    # --------------------------
    elif mcmc is not None:
        samples = mcmc.get_samples()

        def get_stats(k):
            vals = samples[k]  # (num_samples, *shape)
            median = jnp.median(vals, axis=0)
            q_low = jnp.quantile(vals, 0.05, axis=0)
            q_high = jnp.quantile(vals, 0.95, axis=0)
            return median, (median - q_low, q_high - median)

        extractor = lambda k: get_stats(k)

    # --------------------------
    # Collect data for plotting
    # --------------------------
    for gname, keys in groups:
        true_vecs, pred_vecs, err_low_vecs, err_high_vecs = [], [], [], []
        for k in keys:
            if k in true_values:
                t_flat = _flatten_param(true_values[k])

                if guide is not None and svi_result is not None:
                    learned_val = extractor(k)[0]
                    p_flat = _flatten_param(learned_val)
                    el, eh = extractor(k)[1](k, p_flat)
                else:  # MCMC
                    median, (el, eh) = extractor(k)
                    p_flat = _flatten_param(median)
                    el, eh = _flatten_param(el), _flatten_param(eh)

                true_vecs.append(t_flat)
                pred_vecs.append(p_flat)
                if el is not None and eh is not None:
                    err_low_vecs.append(el)
                    err_high_vecs.append(eh)

        if true_vecs:
            xs.append(jnp.concatenate(true_vecs, axis=0))
            ys.append(jnp.concatenate(pred_vecs, axis=0))
            names.append(gname)

            if err_low_vecs and err_high_vecs:
                yerr_low.append(jnp.concatenate(err_low_vecs, axis=0))
                yerr_high.append(jnp.concatenate(err_high_vecs, axis=0))
            else:
                yerr_low.append(None)
                yerr_high.append(None)

    # --------------------------
    # Plot
    # --------------------------
    ncols = min(3, len(xs))
    nrows = (len(xs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(-1)

    for i, (xg, yg, name) in enumerate(zip(xs, ys, names)):
        ax = axes[i]
        if yerr_low[i] is not None:
            ax.errorbar(
                xg,
                yg,
                yerr=[yerr_low[i], yerr_high[i]],
                fmt="o",
                ms=4,
                alpha=0.9,
                capsize=2,
            )
        else:
            ax.scatter(xg, yg, s=18, alpha=0.9)

        # Compute limits with some padding (be sure to include error bars)
        all_vals = [xg, yg]
        if yerr_low[i] is not None:
            all_vals.append(yg - yerr_low[i])
        if yerr_high[i] is not None:
            all_vals.append(yg + yerr_high[i])
        lo = float(jnp.min(jnp.concatenate(all_vals)))
        hi = float(jnp.max(jnp.concatenate(all_vals)))
        pad = 0.05 * (hi - lo + 1e-9)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        ax.set_xlim([lo, hi])
        ax.set_ylim([lo, hi])
        ax.set_title(f"{name} (learned vs true)")
        ax.set_xlabel("True")
        ax.set_ylabel("Learned")
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    return fig
