"""Dense Random Optimizer Comparison.

Baseline dense random phase retrieval comparison across GD, Accelerated GD, Adam, NCG, and L-BFGS.
"""
from report_experiments_split.common import *
from report_experiments_split.experiment_configs import BASELINE_ALPHA, BASELINE_CONFIGS


def run():
    """Run this experiment and return its main runs plus summary table when available."""
    all_runs = []

    dense_alpha = BASELINE_ALPHA
    dense_use_plot_floor = True  # True: clip curves at PLOT_FLOOR; False: do not clip, only avoid log(0).
    dense_method_configs = BASELINE_CONFIGS

    dense_runs = [
        run_case(method, "deepinv_random", "intensity", "spectral", dense_alpha, 0.0, config)
        for method, config in dense_method_configs.items()
    ]
    all_runs += dense_runs
    dense_df = save_summary(dense_runs, "dense_summary.csv")
    plot_loss_error_side_by_side(
        dense_runs,
        "dense_loss_and_error_vs_iteration",
        "Dense random phase retrieval optimizer comparison",
        alpha=dense_alpha,
        use_plot_floor=dense_use_plot_floor,
    )
    plot_error_vs_time(dense_runs, "dense_error_vs_time", "Dense random phase retrieval: error vs time", alpha=dense_alpha, use_plot_floor=dense_use_plot_floor)

    dense_fit_ranges = {
        "GD": (0.20, 0.80),
        "Accelerated GD": (0.10, 0.3),
        "Adam": (0.15, 0.30),
        "NCG": (0.05, 0.50),
        "L-BFGS": (0.05, 0.30),
    }

    fit_dense_convergence_rate_fit = plot_convergence_rate_fits(
        dense_runs,
        "dense_convergence_rate_fit",
        "Dense random phase retrieval",
        alpha=dense_alpha,
        fit_ranges=dense_fit_ranges,
        use_plot_floor=dense_use_plot_floor,
    )

    psnr_dense_runs_df = plot_reconstructions_with_psnr(
        dense_runs,
        "dense_reconstructions_psnr",
        "Dense random phase retrieval reconstructions",
        label_keys=('method',),
    )
    fit_dense_convergence_rate_fit


    def plot_local_convergence_ratios_cut(runs, methods=("NCG", "L-BFGS"), eps=1e-12):
        fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
        rows = []

        for run in runs:
            method = run["method"]
            if method not in methods:
                continue

            err = np.asarray(run["recon_error_hist"], dtype=float) / n_pixels

            valid = np.isfinite(err) & (err > eps)
            err = err[valid]

            if len(err) < 5:
                continue

            ratio = err[1:] / err[:-1]

            # remove plateau part where error almost stops changing
            active = ratio < 0.999
            ratio = ratio[active]

            if len(ratio) < 5:
                continue

            k = np.arange(1, len(ratio) + 1)

            ax.plot(k, ratio, marker="o", markersize=3, label=method)

            tail = ratio[-5:]

            rows.append({
                "method": method,
                "first_ratio": ratio[0],
                "median_ratio": np.median(ratio),
                "last_active_ratio_mean": np.mean(tail),
                "last_active_ratio_min": np.min(tail),
                "last_active_ratio_max": np.max(tail),
            })

        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"Local ratio $e_{k+1}/e_k$")
        ax.set_title("Local convergence ratio before plateau")
        ax.grid(True, which="both")
        ax.legend(frameon=False)

        return pd.DataFrame(rows)

    ratio_dense_cut_df = plot_local_convergence_ratios_cut(dense_runs)
    ratio_dense_cut_df
    return {
        "runs": locals().get("dense_runs"),
        "summary": locals().get("dense_df"),
        "all_runs": all_runs,
    }


if __name__ == "__main__":
    outputs = run()
    summary = outputs.get("summary")
    if summary is not None:
        print(summary)
