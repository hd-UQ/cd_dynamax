#!/usr/bin/env python3
"""
Simple L–dt sweep using data_reset_dict.

- For each (L, dt) and replicate:
  * Sets regular sampling via data_reset_dict: num_samples, irregular_samples=False, key
  * Calls run_filter_then_forecast with only data_config_file + data_reset_dict
- Optionally makes per-replicate low-level plots
- Makes high-level summary plots:
  (1) For each L: rel-RMSE vs dt   (mean ± 95% CI) across reps, all filters
  (2) For each dt: rel-RMSE vs L   (mean ± 95% CI) across reps, all filters
"""

import os
import argparse
import pickle
import numpy as np
import jax.numpy as jnp

# --- project imports (same package folder) ---
from run_filtering_then_forecast_experiment import (
    run_filter_then_forecast,
    eval_filter_then_forecast_experiment,
    build_results_dir,
)

# ============== utilities ==============
def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def parse_csv_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(',') if x.strip()]

def parse_csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(',') if x.strip()]

def pretty(x: float) -> str:
    return f"{x:g}"  # compact for folder names

def unique_key(base: int | None, iL: int, idt: int, rep: int) -> int:
    base0 = 0 if base is None else int(base)
    return base0 + iL*10_000 + idt*100 + rep

def num_samples_for_dt(t0: float, t1: float, dt: float) -> int:
    n = int(round((t1 - t0) / dt)) + 1
    return max(n, 2)

def rel_rmse(truth: np.ndarray, est: np.ndarray, eps: float = 1e-12) -> float:
    num = np.sqrt(np.mean((est - truth) ** 2))
    den = np.sqrt(np.mean(truth ** 2)) + eps
    return float(num / den)

def load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ============== main ==============
def main():
    ap = argparse.ArgumentParser(description="Simple L–dt sweep using data_reset_dict.")

    # Base configs & I/O
    ap.add_argument('--data_config_file', type=str, default='configs/data/l63_data_x1')
    ap.add_argument('--model_config_file', type=str, default='configs/model/true_l63_mech_x1')
    ap.add_argument('--output_root', type=str, default='results/filter_then_forecast_SWEEP_inflation')
    ap.add_argument('--data_root', type=str, default='data',
                    help='where to save data files; defaults to "data" subfolder of output_root')
    ap.add_argument('--T_filter', type=float, default=0.8)

    # Grid
    ap.add_argument('--Ls', type=str, default='1e-2,1e-1,1,2,5,10')
    ap.add_argument('--dts', type=str, default='1e-3,1e-2,0.1,0.2,0.5')
    ap.add_argument('--num_reps', type=int, default=10)
    ap.add_argument('--irregular_samples', type=int, default=0,
                    help='if 1, use irregular sampling; if 0, use regular sampling')
    # Filters
    ap.add_argument('--filters', type=str, default=','.join([
        # "covInflation10_ekf_StateFirst_EmissionsFirst",
        # "covInflation10_ekf_StateSecond_EmissionsFirst",
        # "covInflation10_ekf_StateZeroth_EmissionsFirst",
        # "covInflation10_enkf_StateFirst",
        # "covInflation10_enkf_StateZero",
        # "covInflation10_ukf_StateFirst",
        # "covInflation10_ukf_StateZeroth",
        # "ekf_StateFirst_EmissionsFirst",
        # "ekf_StateSecond_EmissionsFirst",
        # "ekf_StateZeroth_EmissionsFirst",
        "enkf_StateFirst",
        "enkf_StateZero",
        # "ukf_StateFirst",
        # "ukf_StateZeroth",
    ]))

    # Run / Eval toggles
    ap.add_argument('--do_run', type=int, default=1)
    ap.add_argument('--do_eval', type=int, default=1)
    ap.add_argument('--do_low_level_plots', type=int, default=1)
    ap.add_argument('--enforce_twin_experiment', type=int, default=1)

    # Seeds/keys
    ap.add_argument('--base_data_key', type=int, default=None)
    ap.add_argument('--base_ftf_key', type=int, default=None)

    # Time window for building regular grids via num_samples
    ap.add_argument('--t0', type=float, default=0.0)
    ap.add_argument('--t1', type=float, default=10.0)

    # Optional: override model params too (leave empty for none)
    # e.g., dynamics.diffusion_coefficient.scale
    ap.add_argument('--L_param_path', type=str, default='dynamics.diffusion_coefficient.params')
    ap.add_argument('--apply_L_to_filter', type=int, default=1)

    # Plot scales
    ap.add_argument('--xscale', type=str, default='log', choices=['linear', 'log'])
    ap.add_argument('--yscale', type=str, default='log', choices=['linear', 'log'])
    ap.add_argument('--ci_pct', type=float, default=95.0, help='central CI percentage (e.g., 95)')
    ap.add_argument('--center', type=str, default='median', choices=['median','mean'],
                    help='central tendency plotted (median or mean)')

    args = ap.parse_args()

    Ls = parse_csv_floats(args.Ls)
    dts = parse_csv_floats(args.dts)
    filters = parse_csv_list(args.filters)

    data_dir = ensure_dir(os.path.join(args.output_root, args.data_root))

    # ---------------- RUN ----------------
    if args.do_run:
        print("==> RUN PHASE")
        for iL, L in enumerate(Ls):
            for idt, dt in enumerate(dts):
                for rep in range(args.num_reps):
                    data_key = unique_key(args.base_data_key, iL, idt, rep)
                    ftf_key  = args.base_ftf_key if args.base_ftf_key is not None else data_key + 10

                    grid_root = os.path.join(
                        args.output_root, f"L={pretty(L)}", f"dt={pretty(dt)}", f"rep={rep}"
                    )
                    ensure_dir(grid_root)

                    # === Build data_reset_dict (NO ConfigParser needed) ===
                    n_samples = num_samples_for_dt(args.t0, args.t1, dt)
                    data_reset_dict = {
                        # everything lives under [data_generation] in your config.
                        't0': str(args.t0),
                        't1': str(args.t1),
                        'num_samples': str(n_samples),
                        'irregular_samples': str(args.irregular_samples),
                        'key': str(data_key),
                    }
                    # if args.enforce_twin_experiment:
                    #     data_reset_dict['data_generation.true_model_config_file'] = args.model_config_file

                    # Optional: set L in the *truth* model via param_reset_dict
                    param_reset_truth = {}
                    if args.L_param_path:
                        param_reset_truth[args.L_param_path] = L*jnp.eye(3)

                    # Run each filter (your function handles data generation using data_reset_dict)
                    for name in filters:
                        filter_cfg = f"configs/filter/{name}"
                        # Optionally also apply L to the filter model
                        param_reset_filter = param_reset_truth if args.apply_L_to_filter else {}

                        run_filter_then_forecast(
                            data_config_file=args.data_config_file,
                            model_config_file=args.model_config_file,
                            filter_config_file=filter_cfg,
                            output_dir=grid_root,
                            T_filter=args.T_filter,
                            enforce_twin_experiment=bool(args.enforce_twin_experiment),
                            data_key=data_key,      # used only for naming/keys if your function still supports it
                            ftf_key=ftf_key,
                            data_reset_dict=data_reset_dict,          # <--- new, simple knob
                            param_reset_dict=param_reset_filter,      # optional
                            data_dir=data_dir
                        )

                    # Per-replicate low-level plots (optional)
                    if args.do_low_level_plots:
                        eval_out = ensure_dir(os.path.join(grid_root, 'eval'))
                        result_dirs = [
                            build_results_dir(
                                args.data_config_file,
                                args.model_config_file,
                                f"configs/filter/{name}",
                                grid_root
                            )
                            for name in filters
                        ]
                        eval_filter_then_forecast_experiment(
                            data_config_file=args.data_config_file,
                            list_of_result_dirs=result_dirs,
                            enforce_twin_experiment=bool(args.enforce_twin_experiment),
                            model_config_file=args.model_config_file,
                            eval_output_dir=eval_out,
                            data_key=data_key,
                            ftf_key=ftf_key,
                            # if your evaluator also accepts data_reset_dict, pass it here too:
                            data_reset_dict=data_reset_dict,
                            param_reset_dict=param_reset_truth,
                            data_dir=data_dir
                        )

    # ---------------- SUMMARY ----------------
    if args.do_eval:
        def build_metrics_matrix(
            Ls, dts, num_reps, filters, args,
        ) -> dict[str, np.ndarray]:
            """
            Returns metrics[m] of shape (nL, nDt, nRep), with NaN where missing.
            Each entry is rel-RMSE over the filtering window for that replicate.
            """
            metrics = {m: np.full((len(Ls), len(dts), num_reps), np.nan, dtype=float) for m in filters}

            def data_path_for_key(data_key: int) -> str:
                # Matches _make_data default save pattern in your run function
                return os.path.join(data_dir, f"{os.path.basename(args.model_config_file)}_data_{data_key}.pkl")

            for iL, L in enumerate(Ls):
                for idt, dt in enumerate(dts):
                    for rep in range(num_reps):
                        data_key = unique_key(args.base_data_key, iL, idt, rep)

                        # Load the truth data used for this replicate
                        dp = data_path_for_key(data_key)
                        if not os.path.exists(dp):
                            print(f"[WARN] missing data file: {dp}")
                            continue
                        data = load_pickle(dp)
                        Xtrue = np.asarray(data['states'])

                        for m in filters:
                            res_dir = os.path.join(
                                args.output_root, f"L={pretty(L)}", f"dt={pretty(dt)}", f"rep={rep}",
                                os.path.basename(args.data_config_file),
                                os.path.basename(args.model_config_file),
                                os.path.basename(f"configs/filter/{m}")
                            )
                            rp = os.path.join(res_dir, 'results.pkl')
                            if not os.path.exists(rp):
                                # Missing results for this filter/rep
                                continue

                            res = load_pickle(rp)
                            i0, i1 = int(res['start_idx_filter']), int(res['stop_idx_filter'])
                            if i1 <= i0:
                                continue
                            f_means = np.asarray(res['filtered']['filtered_means'])
                            if f_means.shape[0] < i1:
                                continue

                            metrics[m][iL, idt, rep] = rel_rmse(Xtrue[i0:i1], f_means[i0:i1])

            return metrics

        print("==> SUMMARY PHASE")

        # Build metrics cube first (this was missing)
        metrics = build_metrics_matrix(Ls, dts, args.num_reps, filters, args)

        def compute_empirical_summary(metrics_dict, ci_pct: float, center_mode: str):
            """
            metrics_dict[m] has shape (nL, nDt, nRep) with NaNs for missing reps.
            Returns center[m], lo[m], hi[m], nobs[m] each shape (nL, nDt).
            """
            centers = {}
            lo     = {}
            hi     = {}
            nobs   = {}
            low_q  = (100.0 - ci_pct) / 2.0
            high_q = 100.0 - low_q

            for m, arr in metrics_dict.items():
                n = np.sum(~np.isnan(arr), axis=2)
                nobs[m] = n

                if center_mode == 'median':
                    c = np.nanmedian(arr, axis=2)
                else:  # 'mean'
                    c = np.nanmean(arr, axis=2)

                lo_q = np.nanpercentile(arr, low_q,  axis=2)
                hi_q = np.nanpercentile(arr, high_q, axis=2)

                # If <2 reps -> no band info
                mask = n < 2
                lo_q = lo_q.astype(float); hi_q = hi_q.astype(float)
                lo_q[mask] = np.nan
                hi_q[mask] = np.nan

                centers[m] = c
                lo[m]      = lo_q
                hi[m]      = hi_q

            return centers, lo, hi, nobs

        centers, lo, hi, nobs = compute_empirical_summary(metrics, args.ci_pct, args.center)
        summary_dir = ensure_dir(os.path.join(args.output_root, 'summary'))

    import matplotlib.pyplot as plt

    def series_has_data(y):
        return np.any(np.isfinite(y))

    def prep_log(y, lo=None, hi=None, eps=1e-12):
        y  = np.asarray(y,  dtype=float)
        lo = np.asarray(lo, dtype=float) if lo is not None else None
        hi = np.asarray(hi, dtype=float) if hi is not None else None
        y  = np.clip(y,  eps, None)
        if lo is not None:
            lo = np.clip(lo, eps, None)
            hi = np.clip(hi, eps, None)
        return y, lo, hi

    # Plot 1: For each L, y vs dt
    for iL, L in enumerate(Ls):
        xs = np.array(dts); order = np.argsort(xs); xs = xs[order]
        plt.figure(figsize=(7.2, 4.4))
        plotted=False
        for m in filters:
            y  = centers[m][iL][order]
            l  = lo[m][iL][order]
            h  = hi[m][iL][order]
            if not series_has_data(y):
                continue
            plotted=True
            if args.yscale == 'log':
                y_plot, l_plot, h_plot = prep_log(y, l, h)
                plt.plot(xs, y_plot, label=m)
                if series_has_data(l_plot) and series_has_data(h_plot):
                    plt.fill_between(xs, l_plot, h_plot, alpha=0.15)
            else:
                plt.plot(xs, y, label=m)
                if series_has_data(l) and series_has_data(h):
                    plt.fill_between(xs, l, h, alpha=0.15)

        if not plotted:
            plt.close(); print(f"[WARN] No data for L={L:g}; skipping."); continue

        plt.xscale(args.xscale)
        if args.yscale == 'log' and np.nanmax([np.nanmax(centers[m][iL]) for m in filters])>0:
            plt.yscale('log')

        plt.xlabel(r'$\Delta t$ (sample interval)')
        plt.ylabel(f'Filtered {args.center} relative RMSE')
        plt.title(f'{args.center.title()} & central {args.ci_pct:.0f}% band vs Δt  (L = {pretty(L)})')
        plt.legend(fontsize=8)
        out = os.path.join(summary_dir, f"summary_vs_dt_L={pretty(L)}.png")
        plt.tight_layout(); plt.savefig(out, dpi=240); plt.close(); print("Saved:", out)

    # Plot 2: For each dt, y vs L
    for idt, dt in enumerate(dts):
        xs = np.array(Ls); order = np.argsort(xs); xs = xs[order]
        plt.figure(figsize=(7.2, 4.4))
        plotted=False
        for m in filters:
            y  = centers[m][:, idt][order]
            l  = lo[m][:, idt][order]
            h  = hi[m][:, idt][order]
            if not series_has_data(y):
                continue
            plotted=True
            if args.yscale == 'log':
                y_plot, l_plot, h_plot = prep_log(y, l, h)
                plt.plot(xs, y_plot, label=m)
                if series_has_data(l_plot) and series_has_data(h_plot):
                    plt.fill_between(xs, l_plot, h_plot, alpha=0.15)
            else:
                plt.plot(xs, y, label=m)
                if series_has_data(l) and series_has_data(h):
                    plt.fill_between(xs, l, h, alpha=0.15)

        if not plotted:
            plt.close(); print(f"[WARN] No data for dt={dt:g}; skipping."); continue

        plt.xscale(args.xscale)
        if args.yscale == 'log' and np.nanmax([np.nanmax(centers[m][:, idt]) for m in filters])>0:
            plt.yscale('log')

        plt.xlabel(r'$L$ (diffusion coefficient)')
        plt.ylabel('Filtered mean relative RMSE')
        plt.title(f'Rel-RMSE vs L  (Δt = {pretty(dt)})')
        plt.legend(fontsize=8)
        out = os.path.join(summary_dir, f"summary_vs_L_dt={pretty(dt)}.png")
        plt.tight_layout(); plt.savefig(out, dpi=240); plt.close(); print("Saved:", out)

if __name__ == "__main__":
    main()
