"""Noise Robustness.

Runs the reported noisy-measurement experiment.
"""
from report_experiments_split.common import *
from report_experiments_split.experiment_configs import BASELINE_ALPHA, NOISY_MEASUREMENT_LEVEL, NOISY_MEASUREMENTS_CONFIGS


def run():
    """Run this experiment and return its main runs plus summary table when available."""
    all_runs = []

    noise_alpha = BASELINE_ALPHA
    noise_use_plot_floor = False  # True: clip curves at PLOT_FLOOR; False: do not clip, only avoid log(0).
    noise_levels = [NOISY_MEASUREMENT_LEVEL]
    noise_method_configs = NOISY_MEASUREMENTS_CONFIGS

    noise_runs = []
    for noise_level in noise_levels:
        for method, config in noise_method_configs.items():
            noise_runs.append(run_case(method, "deepinv_random", "intensity", "spectral", noise_alpha, noise_level, config))

    all_runs += noise_runs
    noise_df = save_summary(noise_runs, "noise_summary.csv")
    plot_loss_error_side_by_side(
        noise_runs,
        "noise_loss_and_error_vs_iteration",
        f"Noise robustness (noise = {NOISY_MEASUREMENT_LEVEL:g})",
        alpha=noise_alpha,
        use_plot_floor=noise_use_plot_floor,
    )
    plot_error_vs_time(
        noise_runs,
        "noise_error_vs_time",
        f"Noise robustness: error vs time (noise = {NOISY_MEASUREMENT_LEVEL:g})",
        alpha=noise_alpha,
        use_plot_floor=noise_use_plot_floor,
    )

    psnr_noise_runs_df = plot_reconstructions_with_psnr(
        noise_runs,
        "noise_reconstructions_psnr",
        "Noise robustness reconstructions",
        label_keys=('method', 'noise_level'),
    )

    return {
        "runs": locals().get("noise_runs"),
        "summary": locals().get("noise_df"),
        "all_runs": all_runs,
    }


if __name__ == "__main__":
    outputs = run()
    summary = outputs.get("summary")
    if summary is not None:
        print(summary)
