"""Oversampling Ratio Experiment.

Runs the non-baseline oversampling-ratio experiments.
"""
from report_experiments_split.common import *
from report_experiments_split.experiment_configs import OVERSAMPLING_ALPHAS, OVERSAMPLING_CONFIGS_BY_ALPHA


OVERSAMPLING_PLOT_FLOORS = {
    2.0: False,
    10.0: True,
}


def run():
    """Run this experiment and return its main runs plus summary table when available."""
    all_runs = []

    oversampling_alphas = OVERSAMPLING_ALPHAS

    oversampling_runs = []
    for alpha in oversampling_alphas:
        alpha = float(alpha)
        oversampling_method_configs = OVERSAMPLING_CONFIGS_BY_ALPHA[alpha]
        for method, config in oversampling_method_configs.items():
            oversampling_runs.append(run_case(method, "deepinv_random", "intensity", "spectral", alpha, 0.0, config))

    all_runs += oversampling_runs
    oversampling_df = save_summary(oversampling_runs, "oversampling_summary.csv")
    for alpha in oversampling_alphas:
        alpha_runs = [run for run in oversampling_runs if run["alpha"] == float(alpha)]
        alpha_stem = f"oversampling_alpha_{alpha:g}".replace(".", "p")
        alpha_use_plot_floor = OVERSAMPLING_PLOT_FLOORS.get(float(alpha), True)
        plot_loss_error_side_by_side(
            alpha_runs,
            f"{alpha_stem}_loss_and_error_vs_iteration",
            f"Oversampling ratio alpha = {alpha:g}",
            alpha=float(alpha),
            use_plot_floor=alpha_use_plot_floor,
        )
        plot_error_vs_time(
            alpha_runs,
            f"{alpha_stem}_error_vs_time",
            f"Oversampling ratio alpha = {alpha:g}: error vs time",
            alpha=float(alpha),
            use_plot_floor=alpha_use_plot_floor,
        )

    return {
        "runs": locals().get("oversampling_runs"),
        "summary": locals().get("oversampling_df"),
        "all_runs": all_runs,
    }


if __name__ == "__main__":
    outputs = run()
    summary = outputs.get("summary")
    if summary is not None:
        print(summary)
