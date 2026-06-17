"""Plotting and export helpers for report experiments."""
import deepinv as dinv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    from IPython.display import display
except ImportError:  # pragma: no cover - only used outside notebooks.
    def display(value):
        print(value)

from .config import METHOD_COLORS, OUT_DIR
from .metrics import (
    DEFAULT_FIT_RANGES,
    apply_plot_floor,
    fit_semilog_convergence,
    maybe_set_floor_ylim,
    normalized_error,
    objective_values,
    reconstruction_image_from_final,
    summary_row,
)
from .problem import correct_global_phase, n_pixels, x_true, x_true_vis


def save_fig(fig, stem):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.show()
    plt.close(fig)


def save_summary(runs, filename):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([summary_row(run) for run in runs])
    df.to_csv(OUT_DIR / filename, index=False)
    return df


def plot_loss_error_side_by_side(runs, stem, title, alpha=None, label_key="method", use_plot_floor=None):
    alpha_text = f" (alpha = {alpha:g})" if alpha is not None else ""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)

    for run in runs:
        label = run[label_key] if label_key in run else run["method"]
        color = METHOD_COLORS.get(run["method"])
        k_loss = np.arange(1, len(run["loss_hist"]) + 1)
        k_err = np.arange(1, len(run["recon_error_hist"]) + 1)

        axes[0].plot(k_loss, objective_values(run["loss_hist"], use_plot_floor=use_plot_floor), label=label, color=color)
        axes[1].plot(k_err, normalized_error(run["recon_error_hist"], use_plot_floor=use_plot_floor), label=label, color=color)

    axes[0].set_yscale("log")
    maybe_set_floor_ylim(axes[0], use_plot_floor=use_plot_floor)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Objective value")
    axes[0].set_title(f"Loss vs iteration{alpha_text}")

    axes[1].set_yscale("log")
    maybe_set_floor_ylim(axes[1], use_plot_floor=use_plot_floor)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Normalized reconstruction error")
    axes[1].set_title(f"Reconstruction error vs iteration{alpha_text}")

    for ax in axes:
        ax.legend(frameon=False)
        ax.grid(True, which="both")

    fig.suptitle(title, fontsize=12)
    save_fig(fig, stem)


def plot_error_vs_time(runs, stem, title, alpha=None, use_plot_floor=None):
    alpha_text = f" (alpha = {alpha:g})" if alpha is not None else ""
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    for run in runs:
        t = np.cumsum(np.asarray(run["time_hist"], dtype=float))
        ax.plot(t, normalized_error(run["recon_error_hist"], use_plot_floor=use_plot_floor), label=run["method"], color=METHOD_COLORS.get(run["method"]))
    ax.set_yscale("log")
    maybe_set_floor_ylim(ax, use_plot_floor=use_plot_floor)
    ax.set_xlabel("Runtime (seconds)")
    ax.set_ylabel("Normalized reconstruction error")
    ax.set_title(f"{title}{alpha_text}")
    ax.legend(frameon=False)
    ax.grid(True, which="both")
    save_fig(fig, stem)


def lineplot_df(df, x, y, hue, stem, ylabel, title, logy=True, normalize_error=False, use_plot_floor=None):
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    for name, group in df.groupby(hue):
        group = group.sort_values(x)
        vals = group[y].to_numpy(dtype=float)
        if normalize_error:
            vals = vals / n_pixels
        if logy:
            vals = apply_plot_floor(vals, logy=logy, use_plot_floor=use_plot_floor)
        ax.plot(group[x], vals, marker="o", label=str(name), color=METHOD_COLORS.get(str(name)))
    if logy:
        ax.set_yscale("log")
        maybe_set_floor_ylim(ax, use_plot_floor=use_plot_floor)
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, which="both")
    save_fig(fig, stem)


def barplot_df(df, x, y, hue, stem, ylabel, title, logy=True, normalize_error=False, use_plot_floor=None):
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    labels = list(df[x].drop_duplicates())
    methods = list(df[hue].drop_duplicates())
    xs = np.arange(len(labels))
    width = 0.75 / len(methods)
    for i, method in enumerate(methods):
        vals = []
        for label in labels:
            match = df[(df[x] == label) & (df[hue] == method)]
            vals.append(match[y].iloc[0] if len(match) else np.nan)
        vals = np.asarray(vals, dtype=float)
        if normalize_error:
            vals = vals / n_pixels
        if logy:
            vals = apply_plot_floor(vals, logy=logy, use_plot_floor=use_plot_floor)
        ax.bar(xs + (i - (len(methods) - 1) / 2) * width, vals, width, label=method, color=METHOD_COLORS.get(method))
    if logy:
        ax.set_yscale("log")
        maybe_set_floor_ylim(ax, use_plot_floor=use_plot_floor)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, axis="y", which="both")
    save_fig(fig, stem)


def plot_convergence_rate_fits(
    runs,
    stem,
    title,
    alpha=None,
    label_key="method",
    fit_ranges=None,
    use_plot_floor=None,
):
    fit_ranges = fit_ranges or DEFAULT_FIT_RANGES
    alpha_text = f" (alpha = {alpha:g})" if alpha is not None else ""
    fig, ax = plt.subplots(figsize=(7.6, 5.0), constrained_layout=True)

    fit_rows = []
    for run in runs:
        method = run["method"]
        label_base = run[label_key] if label_key in run else method
        start_ratio, end_ratio = fit_ranges.get(method, (0.20, 0.80))
        fit = fit_semilog_convergence(
            run["recon_error_hist"],
            start_ratio=start_ratio,
            end_ratio=end_ratio,
            use_plot_floor=use_plot_floor,
        )
        if fit is None:
            continue

        color = METHOD_COLORS.get(method)
        ax.plot(fit["k"], fit["err"], color=color, alpha=0.25)
        ax.plot(
            fit["k_fit"],
            fit["fitted"],
            "--",
            color=color,
            label=f"{label_base} (rho={fit['rho']:.4f})",
        )

        fit_rows.append({
            "method": method,
            "label": label_base,
            "rho": fit["rho"],
            "slope": fit["slope"],
            "fit_start_index": fit["start"],
            "fit_end_index": fit["end"],
            "alpha": run.get("alpha", np.nan),
            "model": run.get("model", ""),
            "loss": run.get("loss", ""),
            "initialization": run.get("initialization", ""),
            "noise_level": run.get("noise_level", np.nan),
        })

    ax.set_yscale("log")
    maybe_set_floor_ylim(ax, use_plot_floor=use_plot_floor)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Normalized reconstruction error")
    ax.set_title(f"{title}: semilog convergence fit{alpha_text}")
    ax.legend(frameon=False)
    ax.grid(True, which="both")
    save_fig(fig, stem)

    fit_df = pd.DataFrame(fit_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fit_df.to_csv(OUT_DIR / f"{stem}.csv", index=False)
    return fit_df


def plot_qualitative(runs):
    methods = ["GD", "Accelerated GD", "Adam", "NCG", "L-BFGS"]
    run_by_method = {run["method"]: run for run in runs}
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.4), constrained_layout=True)
    axes = axes.ravel()
    axes[0].imshow(x_true_vis.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Ground truth")
    axes[0].axis("off")
    for ax, method in zip(axes[1:], methods):
        run = run_by_method[method]
        x_final = run["x_final"]
        x_corr = correct_global_phase(x_final.detach().clone(), x_true)
        recon = torch.angle(x_corr) / torch.pi + 0.5
        psnr_db = run.get("psnr_db", dinv.metric.PSNR()(x_true_vis.detach(), recon.detach()).item())
        final_error = run.get("final_reconstruction_error", np.nan)
        normalized_final_error = final_error / n_pixels if np.isfinite(final_error) else np.nan
        ax.imshow(recon.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{method}\nPSNR={psnr_db:.2f} dB | Err={final_error:.2e}\nErr/n={normalized_final_error:.2e}")
        ax.axis("off")
    save_fig(fig, "qualitative_reconstructions")

    error_maps = {}
    vmax = 0.0
    for method in methods:
        x_final = run_by_method[method]["x_final"]
        x_corr = correct_global_phase(x_final.detach().clone(), x_true)
        emap = torch.abs(x_corr - x_true).detach().cpu().squeeze().numpy()
        error_maps[method] = emap
        vmax = max(vmax, float(np.quantile(emap, 0.98)))

    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.4), constrained_layout=True)
    axes = axes.ravel()
    axes[0].imshow(np.zeros_like(error_maps[methods[0]]), cmap="magma", vmin=0, vmax=max(vmax, 1e-8))
    axes[0].set_title("Ground truth")
    axes[0].axis("off")
    for ax, method in zip(axes[1:], methods):
        ax.imshow(error_maps[method], cmap="magma", vmin=0, vmax=max(vmax, 1e-8))
        ax.set_title(method)
        ax.axis("off")
    save_fig(fig, "qualitative_error_maps")


def format_run_label(run, label_keys=None):
    if label_keys is None:
        if "plot_label" in run:
            return str(run["plot_label"])
        label_keys = ("method",)
    parts = []
    for key in label_keys:
        if key not in run:
            continue
        value = run[key]
        if isinstance(value, float):
            value = f"{value:g}"
        if key == "method":
            parts.append(str(value))
        else:
            parts.append(f"{key}={value}")
    return "\n".join(parts) if parts else str(run.get("method", "run"))


def plot_reconstructions_with_psnr(runs, stem, title, label_keys=None, max_cols=4):
    if not runs:
        return pd.DataFrame(columns=["label", "method", "psnr_db"])

    rows = []
    total = len(runs) + 1
    cols = min(max_cols, total)
    grid_rows = int(np.ceil(total / cols))
    fig, axes = plt.subplots(
        grid_rows,
        cols,
        figsize=(2.35 * cols, 2.65 * grid_rows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    axes[0].imshow(x_true_vis.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Ground truth")
    axes[0].axis("off")

    for ax, run in zip(axes[1:], runs):
        recon = reconstruction_image_from_final(run["x_final"])
        psnr_db = run.get("psnr_db", dinv.metric.PSNR()(x_true_vis.detach(), recon.detach()).item())
        final_error = run.get("final_reconstruction_error", np.nan)
        normalized_final_error = final_error / n_pixels if np.isfinite(final_error) else np.nan
        label = format_run_label(run, label_keys=label_keys)
        rows.append({
            "label": label.replace("\n", " / "),
            "method": run.get("method", ""),
            "alpha": run.get("alpha", np.nan),
            "model": run.get("model", ""),
            "loss": run.get("loss", ""),
            "initialization": run.get("initialization", ""),
            "noise_level": run.get("noise_level", np.nan),
            "final_reconstruction_error": final_error,
            "normalized_final_reconstruction_error": normalized_final_error,
            "psnr_db": psnr_db,
        })
        ax.imshow(recon.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(
            f"{label}\n"
            f"PSNR={psnr_db:.2f} dB | Err={final_error:.2e}\n"
            f"Err/n={normalized_final_error:.2e}"
        )
        ax.axis("off")

    for ax in axes[total:]:
        ax.axis("off")

    fig.suptitle(title, y=1.02)
    save_fig(fig, stem)

    psnr_df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    psnr_df.to_csv(OUT_DIR / f"{stem}.csv", index=False)
    display(psnr_df[[
        "label",
        "method",
        "psnr_db",
        "final_reconstruction_error",
        "normalized_final_reconstruction_error",
    ]])
    return psnr_df
