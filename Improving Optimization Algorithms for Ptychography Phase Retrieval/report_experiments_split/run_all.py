"""Run all split report experiments."""
from report_experiments_split.common import save_summary
from report_experiments_split.experiments import (
    dense_random_optimizer_comparison,
    structured_random_forward_model,
    oversampling_ratio,
    loss_comparison,
    initialization_comparison,
    noise_robustness,
)


def main():
    all_runs = []
    for module in [
        dense_random_optimizer_comparison,
        structured_random_forward_model,
        oversampling_ratio,
        loss_comparison,
        initialization_comparison,
        noise_robustness,
    ]:
        print(f"Running {module.__name__.split('.')[-1]}...")
        outputs = module.run()
        all_runs.extend(outputs.get("all_runs") or [])
    all_df = save_summary(all_runs, "all_experiment_summary.csv")
    print("Saved combined rows:", len(all_df))
    return all_df


if __name__ == "__main__":
    main()
