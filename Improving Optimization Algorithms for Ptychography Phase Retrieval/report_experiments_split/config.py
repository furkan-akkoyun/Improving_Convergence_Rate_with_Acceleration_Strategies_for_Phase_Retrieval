"""Shared configuration and plotting style for report experiments."""
from pathlib import Path

import matplotlib as mpl
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "results" / "report_experiments_src_separate"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

PLOT_FLOOR = 1e-6
USE_PLOT_FLOOR = True
LOG_EPS = 1e-12
SUCCESS_THRESHOLD = 1e-2

mpl.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.24,
    "grid.linewidth": 0.7,
    "lines.linewidth": 2.0,
})

METHOD_COLORS = {
    "GD": "#1f77b4",
    "Accelerated GD": "#ff7f0e",
    "Adam": "#2ca02c",
    "NCG": "#d62728",
    "L-BFGS": "#6f4aa8",
}
