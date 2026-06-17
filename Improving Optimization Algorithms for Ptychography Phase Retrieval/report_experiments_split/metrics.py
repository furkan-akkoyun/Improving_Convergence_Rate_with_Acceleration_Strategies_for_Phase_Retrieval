"""Metrics, summaries, and convergence helpers."""
import deepinv as dinv
import numpy as np
import torch

from .config import LOG_EPS, PLOT_FLOOR, USE_PLOT_FLOOR
from .problem import correct_global_phase, ensure_signal_shape, n_pixels, x_true, x_true_vis


def reconstruction_image_from_final(x_final):
    x_corr = correct_global_phase(ensure_signal_shape(x_final.detach().clone(), "x_final"), x_true)
    return torch.angle(x_corr) / torch.pi + 0.5


def compute_reconstruction_psnr(x_final):
    recon = reconstruction_image_from_final(x_final)
    return dinv.metric.PSNR()(x_true_vis.detach(), recon.detach()).item()


def summary_row(run):
    return {
        "method": run["method"],
        "model": run["model"],
        "loss": run["loss"],
        "initialization": run["initialization"],
        "alpha": run["alpha"],
        "noise_level": run["noise_level"],
        "final_loss": run["final_loss"],
        "final_reconstruction_error": run["final_reconstruction_error"],
        "psnr_db": run.get("psnr_db", compute_reconstruction_psnr(run["x_final"]) if "x_final" in run else np.nan),
        "runtime_seconds": run["runtime_seconds"],
        "iterations": run["iterations"],
        "success": run["success"],
    }


def resolve_plot_floor(use_plot_floor=None):
    return USE_PLOT_FLOOR if use_plot_floor is None else use_plot_floor


def apply_plot_floor(values, logy=True, use_plot_floor=None):
    values = np.asarray(values, dtype=float)
    if not logy:
        return values
    floor = PLOT_FLOOR if resolve_plot_floor(use_plot_floor) else LOG_EPS
    return np.maximum(values, floor)


def maybe_set_floor_ylim(ax, use_plot_floor=None):
    if resolve_plot_floor(use_plot_floor):
        ax.set_ylim(bottom=PLOT_FLOOR)


def normalized_error(values, use_plot_floor=None):
    return apply_plot_floor(np.asarray(values, dtype=float) / n_pixels, logy=True, use_plot_floor=use_plot_floor)


def objective_values(values, use_plot_floor=None):
    return apply_plot_floor(values, logy=True, use_plot_floor=use_plot_floor)


DEFAULT_FIT_RANGES = {
    "GD": (0.20, 0.80),
    "Accelerated GD": (0.05, 0.35),
    "Adam": (0.20, 0.70),
    "NCG": (0.05, 0.50),
    "L-BFGS": (0.05, 0.50),
}


def fit_semilog_convergence(error_hist, start_ratio=0.20, end_ratio=0.80, use_plot_floor=None):
    err = np.asarray(error_hist, dtype=float) / n_pixels
    err = apply_plot_floor(err, logy=True, use_plot_floor=use_plot_floor)
    n = len(err)
    k = np.arange(1, n + 1)
    start = max(0, int(start_ratio * n))
    end = min(n, int(end_ratio * n))
    if end <= start + 2:
        start, end = 0, n

    k_fit = k[start:end]
    e_fit = err[start:end]
    valid = np.isfinite(e_fit) & (e_fit > 0)
    if valid.sum() < 3:
        return None

    k_fit = k_fit[valid]
    e_fit = e_fit[valid]
    slope, intercept = np.polyfit(k_fit, np.log(e_fit), 1)
    rho = float(np.exp(slope))
    fitted = np.exp(intercept) * (rho ** k_fit)
    return {
        "k": k,
        "err": err,
        "k_fit": k_fit,
        "fitted": fitted,
        "rho": rho,
        "slope": float(slope),
        "start": int(start),
        "end": int(end),
    }
