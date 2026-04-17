import torch
from deepinv.optim.prior import Prior
import numpy as np
import torch
import matplotlib.pyplot as plt
from deepinv.utils.plotting import plot
from deepinv.optim.phase_retrieval import (
    correct_global_phase,
)
from .losses import * 

def gradient_descent_phase_retrieval(
    x_init,          
    y,
    physics,
    x_true,
    num_iter=1000,
    step_size=1e-2,
    step_mode="constant",  # "constant", "exp_decay", "inverse_decay"
    initial_lr=1e-2,
    decay_rate=0.99,
    loss_type="intensity",  
    verbose=True
):
    import time
    import torch

    x = x_init.clone().detach().requires_grad_(True)

    loss_hist = []
    recon_error_hist = []
    time_hist = []

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    for i in range(num_iter):
        iter_start = time.perf_counter()

        if step_mode == "constant":
            lr = step_size
        elif step_mode == "exp_decay":
            lr = initial_lr * (decay_rate ** i)
        elif step_mode == "inverse_decay":
            lr = initial_lr / (1 + decay_rate * i)
        else:
            raise ValueError("Unknown step_mode")

        if loss_type == "intensity":
            loss = intensity_loss(x, y, physics)
        elif loss_type == "amplitude":
            loss = amplitude_loss(x, y, physics)
        else:
            raise ValueError("loss_type must be 'intensity' or 'amplitude'")

        if x.grad is not None:
            x.grad.zero_()

        loss.backward()

        with torch.no_grad():
            x -= lr * x.grad

            # reconstruction error
            x_corrected = correct_global_phase(x.clone(), x_true)
            recon_error = torch.sum(torch.abs(x_corrected - x_true) ** 2).item()
            recon_error_hist.append(recon_error)

        x = x.detach().requires_grad_(True)
        loss_hist.append(loss.item())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        iter_end = time.perf_counter()
        time_hist.append(iter_end - iter_start)

        if verbose and i % 100 == 0:
            print(f"[{i}] loss={loss.item():.4f}  ||x_k-x*||^2={recon_error:.4f}  lr={lr:.6f}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time

    return {
        "x_final": x.detach(),
        "loss_hist": loss_hist,
        "recon_error_hist": recon_error_hist,
        "time_hist": time_hist,
        "total_time": total_time,
        "avg_time": total_time / num_iter
    }

def stochastic_gradient_descent(
    x_init,
    y,
    physics,
    x_true,
    num_iter=1000,
    step_size=1e-3,
    batch_ratio=0.1,
    step_mode="constant",   # "constant", "exp_decay", "inverse_decay"
    initial_lr=1e-3,
    decay_rate=0.99,
    loss_type="intensity",
    verbose=True
):
    import time
    import torch

    x = x_init.clone().detach().requires_grad_(True)

    loss_hist = []
    recon_error_hist = []
    time_hist = []

    y_flat = y.flatten()
    n_measurements = y_flat.shape[0]
    batch_size = max(1, int(n_measurements * batch_ratio))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    for i in range(num_iter):
        iter_start = time.perf_counter()

        if step_mode == "constant":
            lr = step_size
        elif step_mode == "exp_decay":
            lr = initial_lr * (decay_rate ** i)
        elif step_mode == "inverse_decay":
            lr = initial_lr / (1 + decay_rate * i)
        else:
            raise ValueError("Unknown step_mode")

        idx = torch.randperm(n_measurements, device=y.device)[:batch_size]

        Bx_full = physics.B.A(x).flatten()
        Bx_batch = Bx_full[idx]
        y_batch = y_flat[idx]

        if loss_type == "intensity":
            loss = torch.sum((Bx_batch.abs().square() - y_batch) ** 2)
        elif loss_type == "amplitude":
            loss = torch.sum((Bx_batch.abs() - torch.sqrt(y_batch + 1e-12)) ** 2)
        else:
            raise ValueError("loss_type must be 'intensity' or 'amplitude'")

        if x.grad is not None:
            x.grad.zero_()

        loss.backward()

        with torch.no_grad():
            x -= lr * x.grad

            x_corrected = correct_global_phase(x.clone(), x_true)
            recon_error = torch.sum(torch.abs(x_corrected - x_true) ** 2).item()
            recon_error_hist.append(recon_error)

        x = x.detach().requires_grad_(True)
        loss_hist.append(loss.item())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        iter_end = time.perf_counter()
        time_hist.append(iter_end - iter_start)

        if verbose and i % 100 == 0:
            print(f"[{i}] loss={loss.item():.4f}  ||x_k-x*||^2={recon_error:.4f}  lr={lr:.6f}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time

    return {
        "x_final": x.detach(),
        "loss_hist": loss_hist,
        "recon_error_hist": recon_error_hist,
        "time_hist": time_hist,
        "total_time": total_time,
        "avg_time": total_time / num_iter
    }

def stochastic_gradient_descent_torch(
    x_init,
    y,
    physics,
    x_true,
    num_iter=1000,
    step_size=1e-3,
    batch_ratio=0.1,
    step_mode="constant",   # "constant", "exp_decay", "inverse_decay"
    initial_lr=1e-3,
    decay_rate=0.99,
    loss_type="intensity",
    momentum=0.0,
    optimizer_type="sgd",   # "sgd" or "adam"
    verbose=True
):
    import time
    import torch

    device = x_init.device

    x = torch.nn.Parameter(x_init.clone().detach().to(device))
    y = y.to(device)
    x_true = x_true.to(device)

    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD([x], lr=step_size, momentum=momentum)
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam([x], lr=step_size)
    else:
        raise ValueError("optimizer_type must be 'sgd' or 'adam'")

    loss_hist = []
    recon_error_hist = []
    time_hist = []

    y_flat = y.flatten()
    n_measurements = y_flat.shape[0]
    batch_size = max(1, int(n_measurements * batch_ratio))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    for i in range(num_iter):
        iter_start = time.perf_counter()

        # learning rate schedule
        if step_mode == "constant":
            lr = step_size
        elif step_mode == "exp_decay":
            lr = initial_lr * (decay_rate ** i)
        elif step_mode == "inverse_decay":
            lr = initial_lr / (1 + decay_rate * i)
        else:
            raise ValueError("Unknown step_mode")

        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad()

        idx = torch.randperm(n_measurements, device=y.device)[:batch_size]

        Bx_full = physics.B.A(x).flatten()
        Bx_batch = Bx_full[idx]
        y_batch = y_flat[idx]

        if loss_type == "intensity":
            loss = torch.sum((Bx_batch.abs().square() - y_batch) ** 2)
        elif loss_type == "amplitude":
            loss = torch.sum((Bx_batch.abs() - torch.sqrt(y_batch + 1e-12)) ** 2)
        else:
            raise ValueError("loss_type must be 'intensity' or 'amplitude'")

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            x_corrected = correct_global_phase(x.detach().clone(), x_true)
            recon_error = torch.sum(torch.abs(x_corrected - x_true) ** 2).item()

        loss_hist.append(loss.item())
        recon_error_hist.append(recon_error)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        iter_end = time.perf_counter()
        time_hist.append(iter_end - iter_start)

        if verbose and i % 100 == 0:
            print(f"[{i}] loss={loss.item():.4f}  ||x_k-x*||^2={recon_error:.4f}  lr={lr:.6f}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time

    return {
        "x_final": x.detach(),
        "loss_hist": loss_hist,
        "recon_error_hist": recon_error_hist,
        "time_hist": time_hist,
        "total_time": total_time,
        "avg_time": total_time / num_iter
    }

def accelerated_gradient_descent(
    x_init,
    y,
    physics,
    x_true,
    num_iter=1000,
    step_size=1e-3,
    step_mode="constant",   # "constant", "exp_decay", "inverse_decay"
    initial_lr=1e-3,
    decay_rate=0.99,
    beta=0.9,
    loss_type="intensity",
    verbose=True
):
    import time
    import torch

    x = x_init.clone().detach()
    v = torch.zeros_like(x)

    loss_hist = []
    recon_error_hist = []
    time_hist = []

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    for i in range(num_iter):
        iter_start = time.perf_counter()

        if step_mode == "constant":
            lr = step_size
        elif step_mode == "exp_decay":
            lr = initial_lr * (decay_rate ** i)
        elif step_mode == "inverse_decay":
            lr = initial_lr / (1 + decay_rate * i)
        else:
            raise ValueError("Unknown step_mode")

        x_lookahead = (x + beta * v).detach().requires_grad_(True)

        if loss_type == "intensity":
            loss = intensity_loss(x_lookahead, y, physics)
        elif loss_type == "amplitude":
            loss = amplitude_loss(x_lookahead, y, physics)
        else:
            raise ValueError("loss_type must be 'intensity' or 'amplitude'")

        if x_lookahead.grad is not None:
            x_lookahead.grad.zero_()

        loss.backward()

        with torch.no_grad():
            grad = x_lookahead.grad.clone()

            v = beta * v - lr * grad
            x = x + v

            x_corrected = correct_global_phase(x.clone(), x_true)
            recon_error = torch.sum(torch.abs(x_corrected - x_true) ** 2).item()
            recon_error_hist.append(recon_error)

        loss_hist.append(loss.item())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        iter_end = time.perf_counter()
        time_hist.append(iter_end - iter_start)

        if verbose and i % 100 == 0:
            print(
                f"[{i}] {loss_type}_loss={loss.item():.4f}  "
                f"||x_k-x*||^2={recon_error:.4f}  "
                f"lr={lr:.6e}"
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time

    return {
        "x_final": x.detach(),
        "loss_hist": loss_hist,
        "recon_error_hist": recon_error_hist,
        "time_hist": time_hist,
        "total_time": total_time,
        "avg_time": total_time / num_iter
    }

def adam(
    x_init,
    y,
    physics,
    x_true,
    num_iter=1000,
    lr=1e-2,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    loss_type="intensity",
    verbose=True
):
    import time
    import torch

    x = x_init.clone().detach().requires_grad_(True)

    loss_hist = []
    recon_error_hist = []
    time_hist = []

    m = torch.zeros_like(x)
    v = torch.zeros_like(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    for i in range(num_iter):
        iter_start = time.perf_counter()

        if loss_type == "intensity":
            loss = intensity_loss(x, y, physics)
        elif loss_type == "amplitude":
            loss = amplitude_loss(x, y, physics)
        else:
            raise ValueError("loss_type must be 'intensity' or 'amplitude'")

        if x.grad is not None:
            x.grad.zero_()

        loss.backward()

        with torch.no_grad():
            g = x.grad.clone()

            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * (g.abs() ** 2)

            t = i + 1
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            x -= lr * m_hat / (torch.sqrt(v_hat) + eps)

            x_corrected = correct_global_phase(x.clone(), x_true)
            recon_error = torch.sum(torch.abs(x_corrected - x_true) ** 2).item()
            recon_error_hist.append(recon_error)

        x = x.detach().requires_grad_(True)
        loss_hist.append(loss.item())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        iter_end = time.perf_counter()
        time_hist.append(iter_end - iter_start)

        if verbose and i % 100 == 0:
            print(
                f"[{i}] {loss_type}_loss={loss.item():.4f}  "
                f"||x_k-x*||^2={recon_error:.4f}"
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time

    return {
        "x_final": x.detach(),
        "loss_hist": loss_hist,
        "recon_error_hist": recon_error_hist,
        "time_hist": time_hist,
        "total_time": total_time,
        "avg_time": total_time / num_iter
    }

def nonlinear_conjugate_gradient(
    x_init,
    y,
    physics,
    x_true,
    num_iter=300,
    loss_type="intensity",
    beta_type="PR+",
    alpha0=1.0,
    rho=0.5,
    c=1e-4,
    min_alpha=1e-8,
    max_ls_steps=20,
    verbose=True
):
    import time
    import torch

    x = x_init.clone().detach().requires_grad_(True)

    loss_hist = []
    recon_error_hist = []
    time_hist = []
    alpha_hist = []
    beta_hist = []

    prev_grad = None
    prev_dir = None

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    for i in range(num_iter):
        iter_start = time.perf_counter()

        if loss_type == "intensity":
            loss = intensity_loss(x, y, physics)
        elif loss_type == "amplitude":
            loss = amplitude_loss(x, y, physics)
        else:
            raise ValueError("loss_type must be 'intensity' or 'amplitude'")

        if x.grad is not None:
            x.grad.zero_()

        loss.backward()

        with torch.no_grad():
            grad = x.grad.clone()
            gk = grad.reshape(-1)

            if i == 0:
                beta = torch.tensor(0.0, device=grad.device, dtype=torch.float32)
                direction = -grad
            else:
                gprev = prev_grad.reshape(-1)
                dprev = prev_dir.reshape(-1)
                yk = gk - gprev

                if beta_type == "FR":
                    numerator = torch.real(torch.sum(torch.conj(gk) * gk))
                    denominator = torch.real(torch.sum(torch.conj(gprev) * gprev)) + 1e-12
                    beta = numerator / denominator

                elif beta_type == "PR":
                    numerator = torch.real(torch.sum(torch.conj(gk) * yk))
                    denominator = torch.real(torch.sum(torch.conj(gprev) * gprev)) + 1e-12
                    beta = numerator / denominator

                elif beta_type == "PR+":
                    numerator = torch.real(torch.sum(torch.conj(gk) * yk))
                    denominator = torch.real(torch.sum(torch.conj(gprev) * gprev)) + 1e-12
                    beta = torch.clamp(numerator / denominator, min=0.0)

                elif beta_type == "HS":
                    numerator = torch.real(torch.sum(torch.conj(gk) * yk))
                    denominator = torch.real(torch.sum(torch.conj(dprev) * yk)) + 1e-12
                    beta = numerator / denominator

                elif beta_type == "DY":
                    numerator = torch.real(torch.sum(torch.conj(gk) * gk))
                    denominator = torch.real(torch.sum(torch.conj(dprev) * yk)) + 1e-12
                    beta = numerator / denominator

                elif beta_type == "CD":
                    numerator = -torch.real(torch.sum(torch.conj(gk) * gk))
                    denominator = torch.real(torch.sum(torch.conj(dprev) * gprev)) + 1e-12
                    beta = numerator / denominator

                else:
                    raise ValueError("beta_type must be one of: 'FR', 'PR', 'PR+', 'HS', 'DY', 'CD'")

                direction = -grad + beta * prev_dir

                dir_inner = torch.real(torch.sum(torch.conj(gk) * direction.reshape(-1)))
                if dir_inner >= 0:
                    beta = torch.tensor(0.0, device=grad.device, dtype=torch.float32)
                    direction = -grad

            current_loss = loss.item()
            grad_dir = torch.real(torch.sum(torch.conj(gk) * direction.reshape(-1))).item()

            alpha = alpha0
            ls_step = 0

            if grad_dir >= 0:
                direction = -grad
                grad_dir = -torch.real(torch.sum(torch.conj(gk) * gk)).item()
                beta = torch.tensor(0.0, device=grad.device, dtype=torch.float32)

            while ls_step < max_ls_steps:
                x_trial = x.detach() + alpha * direction

                if loss_type == "intensity":
                    trial_loss = intensity_loss(x_trial, y, physics).item()
                else:
                    trial_loss = amplitude_loss(x_trial, y, physics).item()

                if trial_loss <= current_loss + c * alpha * grad_dir:
                    break

                alpha *= rho
                ls_step += 1

                if alpha < min_alpha:
                    alpha = min_alpha
                    break

            x = (x.detach() + alpha * direction).requires_grad_(True)

            x_corrected = correct_global_phase(x.detach().clone(), x_true)
            recon_error = torch.sum(torch.abs(x_corrected - x_true) ** 2).item()

            loss_hist.append(current_loss)
            recon_error_hist.append(recon_error)
            alpha_hist.append(alpha)
            beta_hist.append(beta.item())

            prev_grad = grad.clone()
            prev_dir = direction.clone()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        iter_end = time.perf_counter()
        time_hist.append(iter_end - iter_start)

        if verbose and i % 20 == 0:
            print(
                f"[{i}] {loss_type}_loss={loss_hist[-1]:.6f}  "
                f"beta={beta_hist[-1]:.6f}  "
                f"alpha={alpha_hist[-1]:.3e}  "
                f"||x_k-x*||^2={recon_error_hist[-1]:.6f}"
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time

    return {
        "x_final": x.detach(),
        "loss_hist": loss_hist,
        "recon_error_hist": recon_error_hist,
        "time_hist": time_hist,
        "alpha_hist": alpha_hist,
        "beta_hist": beta_hist,
        "total_time": total_time,
        "avg_time": total_time / num_iter
    }