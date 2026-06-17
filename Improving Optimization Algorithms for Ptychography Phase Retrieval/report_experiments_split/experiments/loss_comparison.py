"""Amplitude Loss Experiment.

Runs the non-baseline amplitude-loss experiment for the selected optimizers.
"""
from report_experiments_split.common import *
from report_experiments_split.experiment_configs import AMPLITUDE_LOSS_CONFIGS, BASELINE_ALPHA


def run():
    """Run this experiment and return its main runs plus summary table when available."""
    all_runs = []

    loss_alpha = BASELINE_ALPHA
    loss_use_plot_floor = False  # True: clip curves at PLOT_FLOOR; False: do not clip, only avoid log(0).
    loss_type = "amplitude"
    loss_method_configs = AMPLITUDE_LOSS_CONFIGS

    loss_runs = []
    for method, config in loss_method_configs.items():
        loss_runs.append(run_case(method, "deepinv_random", loss_type, "spectral", loss_alpha, 0.0, config))

    all_runs += loss_runs
    loss_df = save_summary(loss_runs, "amplitude_loss_summary.csv")
    plot_loss_error_side_by_side(
        loss_runs,
        "amplitude_loss_and_error_vs_iteration",
        "Amplitude loss experiment",
        alpha=loss_alpha,
        use_plot_floor=loss_use_plot_floor,
    )

    fit_loss_comparison_convergence_rate_fit = plot_convergence_rate_fits(
        loss_runs,
        "amplitude_loss_convergence_rate_fit",
        "Amplitude loss experiment",
        alpha=loss_alpha,
        use_plot_floor=loss_use_plot_floor,
    )

    psnr_loss_runs_df = plot_reconstructions_with_psnr(
        loss_runs,
        "amplitude_loss_reconstructions_psnr",
        "Amplitude loss reconstructions",
    )
    fit_loss_comparison_convergence_rate_fit

    return {
        "runs": locals().get("loss_runs"),
        "summary": locals().get("loss_df"),
        "all_runs": all_runs,
    }


if __name__ == "__main__":
    outputs = run()
    summary = outputs.get("summary")
    if summary is not None:
        print(summary)
