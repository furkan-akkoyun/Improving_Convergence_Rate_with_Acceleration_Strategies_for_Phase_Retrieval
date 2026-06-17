# Split Report Experiments

This folder is a readable split of `report_experiments_src_algorithms_SEPARATE copy.ipynb`.

- `common.py` is a backward-compatible public namespace that re-exports the split helpers.
- `config.py` contains output paths, seeds, device selection, plot style, and method colors.
- `experiment_configs.py` contains the report hyperparameters from Appendix A.
- `problem.py` contains the phase retrieval problem setup, physics wrappers, measurements, and initialization.
- `runners.py` contains optimizer dispatch and `run_case`.
- `metrics.py` contains PSNR, summary rows, plot-floor helpers, and convergence fitting.
- `plotting.py` contains figure/table export and plotting utilities.
- `deepinv_baseline.py` keeps the original direct `dinv.physics.RandomPhaseRetrieval` baseline setup separate.
- `experiments/` contains one Python file per report experiment.
- `notebooks/` contains one lightweight notebook per experiment.
- `run_all.py` runs the six main experiments and writes the combined summary.

Run one experiment from the project root, for example:

```powershell
.\.venv\Scripts\python.exe -m report_experiments_split.experiments.dense_random_optimizer_comparison
```

Run all experiments:

```powershell
.\.venv\Scripts\python.exe -m report_experiments_split.run_all
```

These scripts use the same dependencies as the original notebook, including `deepinv`.
