"""Structured Random Forward Model Comparison.

Structured random forward model comparison using the report method configuration.
"""
from report_experiments_split.common import *
from report_experiments_split.experiment_configs import BASELINE_ALPHA, STRUCTURED_MODEL_CONFIGS


def run():
    """Run this experiment and return its main runs plus summary table when available."""
    all_runs = []

    structured_alpha = BASELINE_ALPHA
    structured_use_plot_floor = True  # True: clip curves at PLOT_FLOOR; False: do not clip, only avoid log(0).
    structured_method_configs = STRUCTURED_MODEL_CONFIGS

    structured_runs = [
        run_case(method, "structured", "intensity", "spectral", structured_alpha, 0.0, config)
        for method, config in structured_method_configs.items()
    ]
    all_runs += structured_runs
    structured_df = save_summary(structured_runs, "structured_summary.csv")
    plot_loss_error_side_by_side(
        structured_runs,
        "structured_loss_and_error_vs_iteration",
        "Structured random forward model optimizer comparison",
        alpha=structured_alpha,
        use_plot_floor=structured_use_plot_floor,
    )
    plot_error_vs_time(structured_runs, "structured_error_vs_time", "Structured random forward model: error vs time", alpha=structured_alpha, use_plot_floor=structured_use_plot_floor)

    structured_fit_ranges = {
        "GD": (0.20, 0.80),
        "Accelerated GD": (0.05, 0.35),
        "Adam": (0.15, 0.38),  
        "NCG": (0.05, 0.50),
        "L-BFGS": (0.05, 0.50),
    }

    fit_structured_convergence_rate_fit = plot_convergence_rate_fits(
        structured_runs,
        "structured_convergence_rate_fit",
        "Structured random forward model",
        alpha=structured_alpha,
        fit_ranges=structured_fit_ranges,
        use_plot_floor=structured_use_plot_floor,
    )

    psnr_structured_runs_df = plot_reconstructions_with_psnr(
        structured_runs,
        "structured_reconstructions_psnr",
        "Structured random forward model reconstructions",
        label_keys=('method',),
    )
    fit_structured_convergence_rate_fit

    return {
        "runs": locals().get("structured_runs"),
        "summary": locals().get("structured_df"),
        "all_runs": all_runs,
    }


if __name__ == "__main__":
    outputs = run()
    summary = outputs.get("summary")
    if summary is not None:
        print(summary)
