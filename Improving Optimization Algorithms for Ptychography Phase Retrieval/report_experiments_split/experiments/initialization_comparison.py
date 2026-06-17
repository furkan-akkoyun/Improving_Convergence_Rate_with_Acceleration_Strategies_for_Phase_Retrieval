"""Random Initialization Experiment.

Runs the non-baseline random-initialization experiment for the selected optimizers.
"""
from report_experiments_split.common import *
from report_experiments_split.experiment_configs import BASELINE_ALPHA, RANDOM_INITIALIZATION_CONFIGS


def run():
    """Run this experiment and return its main runs plus summary table when available."""
    all_runs = []

    initialization_alpha = BASELINE_ALPHA
    initialization_use_plot_floor = False  # True: clip curves at PLOT_FLOOR; False: do not clip, only avoid log(0).
    initialization = "random"
    initialization_method_configs = RANDOM_INITIALIZATION_CONFIGS

    initialization_runs = []
    for method, config in initialization_method_configs.items():
        initialization_runs.append(run_case(method, "deepinv_random", "intensity", initialization, initialization_alpha, 0.0, config))

    all_runs += initialization_runs
    initialization_df = save_summary(initialization_runs, "random_initialization_summary.csv")
    plot_loss_error_side_by_side(
        initialization_runs,
        "random_initialization_loss_and_error_vs_iteration",
        "Random initialization experiment",
        alpha=initialization_alpha,
        use_plot_floor=initialization_use_plot_floor,
    )

    barplot_df(
        initialization_df,
        "initialization",
        "final_reconstruction_error",
        "method",
        "random_initialization_final_error",
        "Normalized final reconstruction error",
        f"Random initialization: final error (alpha = {initialization_alpha:g})",
        normalize_error=True,
        use_plot_floor=initialization_use_plot_floor,
    )
    psnr_initialization_runs_df = plot_reconstructions_with_psnr(
        initialization_runs,
        "random_initialization_reconstructions_psnr",
        "Random initialization reconstructions",
    )

    return {
        "runs": locals().get("initialization_runs"),
        "summary": locals().get("initialization_df"),
        "all_runs": all_runs,
    }


if __name__ == "__main__":
    outputs = run()
    summary = outputs.get("summary")
    if summary is not None:
        print(summary)
