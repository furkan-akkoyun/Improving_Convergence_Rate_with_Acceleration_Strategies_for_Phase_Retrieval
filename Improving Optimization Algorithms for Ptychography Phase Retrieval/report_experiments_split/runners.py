"""Optimizer dispatch and experiment execution helpers."""
import time

import numpy as np
import torch
from src.algorithms import (
    accelerated_gradient_descent,
    adam,
    gradient_descent_phase_retrieval,
    nonlinear_conjugate_gradient,
)
from src.losses import amplitude_loss, intensity_loss

from .config import SEED, SUCCESS_THRESHOLD
from .metrics import compute_reconstruction_psnr
from .problem import (
    correct_global_phase,
    ensure_signal_shape,
    make_initialization,
    make_measurements,
    make_physics,
    n_pixels,
    sync_if_needed,
    x_true,
)


def quasi_newton_lbfgs(x_init, y, physics, x_true, num_iter=80, loss_type="intensity", lr=1.0, history_size=10, verbose=False):
    x_var = torch.nn.Parameter(x_init.clone().detach())
    optimizer = torch.optim.LBFGS(
        [x_var],
        lr=lr,
        max_iter=1,
        history_size=history_size,
        line_search_fn="strong_wolfe",
    )

    loss_hist, recon_error_hist, time_hist = [], [], []

    def compute_loss():
        if loss_type == "intensity":
            return intensity_loss(x_var, y, physics)
        if loss_type == "amplitude":
            return amplitude_loss(x_var, y, physics)
        raise ValueError(loss_type)

    sync_if_needed()
    start_time = time.perf_counter()

    for i in range(num_iter):
        iter_start = time.perf_counter()

        def closure():
            optimizer.zero_grad()
            loss = compute_loss()
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            current_loss = compute_loss().item()
            x_corrected = correct_global_phase(x_var.detach().clone(), x_true)
            recon_error = torch.sum(torch.abs(x_corrected - x_true) ** 2).item()

        loss_hist.append(current_loss)
        recon_error_hist.append(recon_error)
        sync_if_needed()
        time_hist.append(time.perf_counter() - iter_start)

        if verbose and i % 20 == 0:
            print(f"[{i}] {loss_type}_loss={current_loss:.6e} recon_error={recon_error:.6e}")

    sync_if_needed()
    total_time = time.perf_counter() - start_time
    return {
        "x_final": x_var.detach(),
        "loss_hist": loss_hist,
        "recon_error_hist": recon_error_hist,
        "time_hist": time_hist,
        "total_time": total_time,
        "avg_time": total_time / max(1, len(loss_hist)),
    }


DEFAULT_METHOD_CONFIGS = {
    "GD": {"num_iter": 300, "step_mode": "constant", "step_size": 1e-1},
    "Accelerated GD": {"num_iter": 300, "step_mode": "exp_decay", "initial_lr": 5e-2, "decay_rate": 0.99, "beta": 0.9},
    "Adam": {"num_iter": 300, "lr": 1e-2, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8},
    "NCG": {"num_iter": 120, "beta_type": "PR+", "alpha0": 1.0, "rho": 0.5, "c": 1e-4, "min_alpha": 1e-8, "max_ls_steps": 20},
    "L-BFGS": {"num_iter": 80, "lr": 1.0, "history_size": 10},
}


def method_config(method, overrides=None):
    config = DEFAULT_METHOD_CONFIGS[method].copy()
    if overrides:
        config.update(overrides)
    return config


def run_method(method, x_init, y, physics, loss_type, config=None):
    config = method_config(method, config)
    if method == "GD":
        return gradient_descent_phase_retrieval(
            x_init=x_init,
            y=y,
            physics=physics,
            x_true=x_true,
            num_iter=config["num_iter"],
            step_mode=config["step_mode"],
            step_size=config["step_size"],
            loss_type=loss_type,
            verbose=False,
        )
    if method == "Accelerated GD":
        return accelerated_gradient_descent(
            x_init=x_init,
            y=y,
            physics=physics,
            x_true=x_true,
            num_iter=config["num_iter"],
            step_mode=config["step_mode"],
            initial_lr=config["initial_lr"],
            decay_rate=config["decay_rate"],
            beta=config["beta"],
            loss_type=loss_type,
            verbose=False,
        )
    if method == "Adam":
        return adam(
            x_init=x_init,
            y=y,
            physics=physics,
            x_true=x_true,
            num_iter=config["num_iter"],
            lr=config["lr"],
            beta1=config["beta1"],
            beta2=config["beta2"],
            eps=config["eps"],
            loss_type=loss_type,
            verbose=False,
        )
    if method == "NCG":
        return nonlinear_conjugate_gradient(
            x_init=x_init,
            y=y,
            physics=physics,
            x_true=x_true,
            num_iter=config["num_iter"],
            loss_type=loss_type,
            beta_type=config["beta_type"],
            alpha0=config["alpha0"],
            rho=config["rho"],
            c=config["c"],
            min_alpha=config["min_alpha"],
            max_ls_steps=config["max_ls_steps"],
            verbose=False,
        )
    if method == "L-BFGS":
        return quasi_newton_lbfgs(
            x_init=x_init,
            y=y,
            physics=physics,
            x_true=x_true,
            num_iter=config["num_iter"],
            loss_type=loss_type,
            lr=config["lr"],
            history_size=config["history_size"],
            verbose=False,
        )
    raise ValueError(method)


def run_case(method, model, loss, initialization, alpha, noise_level, method_config=None):
    print(f"run: method={method}, model={model}, loss={loss}, init={initialization}, alpha={alpha}, noise={noise_level}")
    torch.manual_seed(SEED)
    physics = make_physics(model, alpha)
    y = make_measurements(physics, noise_level=noise_level)
    x_init = ensure_signal_shape(make_initialization(physics, y, initialization), "x_init")
    print(f"    shapes: x_true={tuple(x_true.shape)}, x_init={tuple(x_init.shape)}, y={tuple(y.shape)}")
    out = run_method(method, x_init, y, physics, loss, config=method_config)
    out.update({
        "method": method,
        "model": model,
        "loss": loss,
        "initialization": initialization,
        "alpha": alpha,
        "noise_level": noise_level,
    })
    out["final_loss"] = out["loss_hist"][-1]
    out["final_reconstruction_error"] = out["recon_error_hist"][-1]
    out["runtime_seconds"] = out.get("total_time", float(np.sum(out["time_hist"])))
    out["iterations"] = len(out["loss_hist"])
    out["success"] = (out["final_reconstruction_error"] / n_pixels) < SUCCESS_THRESHOLD
    out["psnr_db"] = compute_reconstruction_psnr(out["x_final"])
    return out
