import torch
from deepinv.optim.prior import Prior
import numpy as np
import torch
import matplotlib.pyplot as plt
from deepinv.utils.plotting import plot
from deepinv.optim.phase_retrieval import (
    correct_global_phase,
)

class ZeroPrior(Prior):
    """g(x)=0  => grad=0, prox=identity."""
    def __init__(self):
        super().__init__()

    def grad(self, x, *args, **kwargs):
        return torch.zeros_like(x)

    def prox(self, x, gamma=None, *args, **kwargs):
        return 
    
def plot_metrics(loss_hist, recon_error_hist, time_hist, loss_name="Loss"):
    cumulative_time = np.cumsum(time_hist)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].plot(loss_hist)
    axes[0].set_yscale("log")
    axes[0].set_title(loss_name)
    axes[0].set_xlabel("Iteration")

    axes[1].plot(recon_error_hist)
    axes[1].set_yscale("log")
    axes[1].set_title(r"$\|x_k - x^*\|^2$")
    axes[1].set_xlabel("Iteration")

    axes[2].plot(cumulative_time, recon_error_hist)
    axes[2].set_yscale("log")
    axes[2].set_title("Reconstruction Error vs Time")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel(r"$\|x_k - x^*\|^2$")

    plt.tight_layout()
    plt.show()

def plot_reconstruction(x_true_vis, x_recon_vis, title_recon="Reconstruction"):
    plot(
        [x_true_vis, x_recon_vis],
        titles=["Original Image", title_recon],
        rescale_mode="clip"
    )

def intensity_loss(x, y, physics):
    Ax = physics.A(x)
    return torch.sum((Ax - y)**2)

def print_final_metrics(x_recon, x_true, loss_hist, method_name="GD"):
    final_recon_error = torch.sum(torch.abs(x_recon.detach() - x_true) ** 2).item()

    print(f"{method_name} Final Loss:          {loss_hist[-1]:.6f}")
    print(f"{method_name} Final ||x_k-x*||^2:  {final_recon_error:.6f}")

    return final_recon_error

def print_run_summary(loss_hist, total_time, avg_time, method_name="GD"):
    print(f"{method_name} initial loss: {loss_hist[0]:.6f}")
    print(f"{method_name} final loss:   {loss_hist[-1]:.6f}")
    print(f"{method_name} total time:   {total_time:.4f} sec")
    print(f"{method_name} avg/iter:     {avg_time:.6f} sec")

# def gradient_descent_phase_retrieval(
#     x_init,          
#     y,
#     physics,
#     x_true,
#     num_iter=1000,
#     step_size=1e-2,
#     step_mode="constant",  # "constant", "exp_decay", "inverse_decay"
#     initial_lr=1e-2,
#     decay_rate=0.99,
#     loss_type="intensity",  
#     verbose=True
# ):
#     import time
#     import torch

#     x = x_init.clone().detach().requires_grad_(True)

#     loss_hist = []
#     recon_error_hist = []
#     time_hist = []

#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     start_time = time.perf_counter()

#     for i in range(num_iter):
#         iter_start = time.perf_counter()

#         if step_mode == "constant":
#             lr = step_size
#         elif step_mode == "exp_decay":
#             lr = initial_lr * (decay_rate ** i)
#         elif step_mode == "inverse_decay":
#             lr = initial_lr / (1 + decay_rate * i)
#         else:
#             raise ValueError("Unknown step_mode")

#         if loss_type == "intensity":
#             loss = intensity_loss(x, y, physics)
#         elif loss_type == "amplitude":
#             loss = amplitude_loss(x, y, physics)
#         else:
#             raise ValueError("loss_type must be 'intensity' or 'amplitude'")

#         if x.grad is not None:
#             x.grad.zero_()

#         loss.backward()

#         with torch.no_grad():
#             x -= lr * x.grad

#             # reconstruction error
#             x_corrected = correct_global_phase(x.clone(), x_true)
#             recon_error = torch.sum(torch.abs(x_corrected - x_true) ** 2).item()
#             recon_error_hist.append(recon_error)

#         x = x.detach().requires_grad_(True)
#         loss_hist.append(loss.item())

#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#         iter_end = time.perf_counter()
#         time_hist.append(iter_end - iter_start)

#         if verbose and i % 100 == 0:
#             print(f"[{i}] loss={loss.item():.4f}  ||x_k-x*||^2={recon_error:.4f}  lr={lr:.6f}")

#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     total_time = time.perf_counter() - start_time

#     return {
#         "x_final": x.detach(),
#         "loss_hist": loss_hist,
#         "recon_error_hist": recon_error_hist,
#         "time_hist": time_hist,
#         "total_time": total_time,
#         "avg_time": total_time / num_iter
#     }

# def estimate_convergence(
#     loss_hist,
#     start_ratio=0.3,
#     end_idx=None,
#     eps=1e-12,
#     method_name="Method",
#     plot=True
# ):

#     loss_hist = np.asarray(loss_hist, dtype=np.float64)
#     N = len(loss_hist)

#     if N < 2:
#         raise ValueError("loss_hist en az 2 eleman içermeli.")

#     start_idx = int(start_ratio * N)

#     if end_idx is None:
#         end_idx = N

#     if not (0 <= start_idx < end_idx <= N):
#         raise ValueError("Geçersiz start/end index.")

#     x = np.arange(start_idx, end_idx)
#     y = np.maximum(loss_hist[start_idx:end_idx], eps)

#     log_y = np.log(y)

#     # log(L_n) = a*n + b
#     a, b = np.polyfit(x, log_y, 1)

#     k = np.exp(a)      # çünkü log(k) = a
#     rate = -a          # çünkü k = exp(-rate)
#     fitted_y = np.exp(a * x + b)

#     print(f"{method_name} convergence estimate")
#     print(f"Fit range        : [{start_idx}, {end_idx})")
#     print(f"slope            : {a:.6e}")
#     print(f"estimated k      : {k:.6f}")
#     print(f"estimated rate   : {rate:.6e}")

#     if plot:
#         plt.figure(figsize=(6, 4))
#         plt.plot(np.arange(N), loss_hist, label="Loss")
#         plt.plot(x, fitted_y, "--", label=f"Fit on tail (k={k:.4f})")
#         plt.yscale("log")
#         plt.xlabel("Iteration")
#         plt.ylabel("Loss")
#         plt.title(f"{method_name} Convergence Estimate")
#         plt.legend()
#         plt.tight_layout()
#         plt.show()

#     return {
#         "k": k,
#         "rate": rate,
#         "slope": a,
#         "intercept": b,
#         "start_idx": start_idx,
#         "end_idx": end_idx,
#         "x_fit": x,
#         "y_fit": fitted_y
#     }

def estimate_sublinear_rate(
    loss_hist,
    start_ratio=0.3,
    end_idx=None,
    eps=1e-12,
    method_name="Method",
    plot=True
):
    loss_hist = np.asarray(loss_hist, dtype=np.float64)
    N = len(loss_hist)

    start_idx = max(1, int(start_ratio * N))  # en az 1

    if end_idx is None:
        end_idx = N
    end_idx = min(end_idx, N)  # taşma koruması

    # 1-based iteration numaraları
    k_vals = np.arange(start_idx + 1, end_idx + 1, dtype=np.float64)
    y = loss_hist[start_idx:end_idx]

    # Sadece eps'ten büyük değerleri fit'e al
    valid = y > eps
    if valid.sum() < 2:
        raise ValueError("Fit için yeterli geçerli nokta yok.")

    k_fit = k_vals[valid]
    y_fit = y[valid]

    log_k = np.log(k_fit)
    log_y = np.log(y_fit)

    slope, intercept = np.polyfit(log_k, log_y, 1)

    p = -slope
    C = np.exp(intercept)
    fitted_y = C / (k_fit ** p)

    print(f"{method_name} sublinear convergence estimate")
    print(f"Fit range      : [{start_idx+1}, {end_idx}]  ({valid.sum()} pts)")
    print(f"slope          : {slope:.6e}")
    print(f"estimated p    : {p:.6f}")
    print(f"estimated C    : {C:.6e}")
    print(f"model          : L_k ≈ C / k^{p:.4f}")

    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(np.arange(1, N + 1), loss_hist, label="Loss")  # 1-based
        plt.plot(k_fit, fitted_y, "--", label=f"Fit: C/k^{p:.3f}")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Iteration k")
        plt.ylabel("Loss")
        plt.title(f"{method_name} Sublinear Rate Estimate")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "p": p,
        "C": C,
        "slope": slope,
        "intercept": intercept,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "k_fit": k_fit,
        "y_fit": fitted_y
    }

import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt


def estimate_semilog_rate(
    error_hist,
    start_ratio=0.3,
    end_ratio=1.0,
    eps=1e-12,
    method_name="Method",
    plot=True
):
    """
    Semi-log fit:
        log(e_k) ≈ a k + b

    Model:
        e_k ≈ C * rho^k

    Returns:
        rho            : exp(slope)
        slope          : semi-log slope
        ratio_mean     : mean of e_{k+1} / e_k
        ratio_median   : median of e_{k+1} / e_k
        ratio_std      : std of e_{k+1} / e_k
        r2             : goodness of fit
    """

    error_hist = np.asarray(error_hist, dtype=np.float64)
    N = len(error_hist)

    if N < 3:
        raise ValueError("error_hist en az 3 eleman içermeli.")

    start_idx = int(start_ratio * N)
    end_idx = int(end_ratio * N)

    if start_idx < 0:
        start_idx = 0
    if end_idx > N:
        end_idx = N

    if not (0 <= start_idx < end_idx <= N):
        raise ValueError("Geçersiz start_ratio / end_ratio seçimi.")

    # Seçilen aralık
    y_raw = error_hist[start_idx:end_idx]

    # log için güvenli maske
    valid = np.isfinite(y_raw) & (y_raw > eps)
    if np.sum(valid) < 2:
        raise ValueError("Fit için yeterli geçerli nokta yok.")

    # 1-based iteration index
    k_all = np.arange(start_idx + 1, end_idx + 1, dtype=np.float64)
    k_vals = k_all[valid]
    y = y_raw[valid]

    log_y = np.log(y)

    # log(e_k) = a k + b
    slope, intercept = np.polyfit(k_vals, log_y, 1)

    rho = np.exp(slope)
    C = np.exp(intercept)
    fitted_y = np.exp(slope * k_vals + intercept)

    # R^2 on log-scale
    log_y_pred = slope * k_vals + intercept
    ss_res = np.sum((log_y - log_y_pred) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r2 = 1.0 if ss_tot < 1e-15 else 1.0 - ss_res / ss_tot

    # successive ratios
    ratios = y[1:] / y[:-1]
    valid_ratios = ratios[np.isfinite(ratios)]

    ratio_mean = np.mean(valid_ratios) if len(valid_ratios) > 0 else np.nan
    ratio_median = np.median(valid_ratios) if len(valid_ratios) > 0 else np.nan
    ratio_std = np.std(valid_ratios) if len(valid_ratios) > 0 else np.nan

    print(f"{method_name} semi-log convergence estimate")
    print(f"Fit range         : [{start_idx+1}, {end_idx}]")
    print(f"slope             : {slope:.6e}")
    print(f"rho = exp(slope)  : {rho:.6f}")
    print(f"C                 : {C:.6e}")
    print(f"R^2               : {r2:.6f}")
    print(f"ratio mean        : {ratio_mean:.6f}")
    print(f"ratio median      : {ratio_median:.6f}")
    print(f"ratio std         : {ratio_std:.6f}")
    print("model             : e_k ≈ C * rho^k")

    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(np.arange(1, N + 1), error_hist, label="Error")
        plt.plot(k_vals, fitted_y, "--", label=f"Fit: rho={rho:.4f}")
        plt.yscale("log")
        plt.xlabel("Iteration k")
        plt.ylabel("Error")
        plt.title(f"{method_name} Semi-log Rate Estimate")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "rho": rho,
        "C": C,
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "ratio_mean": ratio_mean,
        "ratio_median": ratio_median,
        "ratio_std": ratio_std,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "k_fit": k_vals,
        "y_fit": fitted_y,
        "ratios": valid_ratios
    }


# def stochastic_gradient_descent(
#     x_init,
#     y,
#     physics,
#     x_true,
#     num_iter=1000,
#     step_size=1e-3,
#     batch_ratio=0.1,
#     step_mode="constant",   # "constant", "exp_decay", "inverse_decay"
#     initial_lr=1e-3,
#     decay_rate=0.99,
#     loss_type="intensity",
#     verbose=True
# ):
#     import time
#     import torch

#     x = x_init.clone().detach().requires_grad_(True)

#     loss_hist = []
#     recon_error_hist = []
#     time_hist = []

#     y_flat = y.flatten()
#     n_measurements = y_flat.shape[0]
#     batch_size = max(1, int(n_measurements * batch_ratio))

#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     start_time = time.perf_counter()

#     for i in range(num_iter):
#         iter_start = time.perf_counter()

#         if step_mode == "constant":
#             lr = step_size
#         elif step_mode == "exp_decay":
#             lr = initial_lr * (decay_rate ** i)
#         elif step_mode == "inverse_decay":
#             lr = initial_lr / (1 + decay_rate * i)
#         else:
#             raise ValueError("Unknown step_mode")

#         idx = torch.randperm(n_measurements, device=y.device)[:batch_size]

#         Bx_full = physics.B.A(x).flatten()
#         Bx_batch = Bx_full[idx]
#         y_batch = y_flat[idx]

#         if loss_type == "intensity":
#             loss = torch.sum((Bx_batch.abs().square() - y_batch) ** 2)
#         elif loss_type == "amplitude":
#             loss = torch.sum((Bx_batch.abs() - torch.sqrt(y_batch + 1e-12)) ** 2)
#         else:
#             raise ValueError("loss_type must be 'intensity' or 'amplitude'")

#         if x.grad is not None:
#             x.grad.zero_()

#         loss.backward()

#         with torch.no_grad():
#             x -= lr * x.grad

#             x_corrected = correct_global_phase(x.clone(), x_true)
#             recon_error = torch.sum(torch.abs(x_corrected - x_true) ** 2).item()
#             recon_error_hist.append(recon_error)

#         x = x.detach().requires_grad_(True)
#         loss_hist.append(loss.item())

#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#         iter_end = time.perf_counter()
#         time_hist.append(iter_end - iter_start)

#         if verbose and i % 100 == 0:
#             print(f"[{i}] loss={loss.item():.4f}  ||x_k-x*||^2={recon_error:.4f}  lr={lr:.6f}")

#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     total_time = time.perf_counter() - start_time

#     return {
#         "x_final": x.detach(),
#         "loss_hist": loss_hist,
#         "recon_error_hist": recon_error_hist,
#         "time_hist": time_hist,
#         "total_time": total_time,
#         "avg_time": total_time / num_iter
#     }

# def accelerated_gradient_descent(
#     x_init,
#     y,
#     physics,
#     x_true,
#     num_iter=1000,
#     step_size=1e-3,
#     step_mode="constant",   # "constant", "exp_decay", "inverse_decay"
#     initial_lr=1e-3,
#     decay_rate=0.99,
#     beta=0.9,
#     loss_type="intensity",
#     verbose=True
# ):
#     import time
#     import torch

#     x = x_init.clone().detach()
#     v = torch.zeros_like(x)

#     loss_hist = []
#     recon_error_hist = []
#     time_hist = []

#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     start_time = time.perf_counter()

#     for i in range(num_iter):
#         iter_start = time.perf_counter()

#         if step_mode == "constant":
#             lr = step_size
#         elif step_mode == "exp_decay":
#             lr = initial_lr * (decay_rate ** i)
#         elif step_mode == "inverse_decay":
#             lr = initial_lr / (1 + decay_rate * i)
#         else:
#             raise ValueError("Unknown step_mode")

#         x_lookahead = (x + beta * v).detach().requires_grad_(True)

#         if loss_type == "intensity":
#             loss = intensity_loss(x_lookahead, y, physics)
#         elif loss_type == "amplitude":
#             loss = amplitude_loss(x_lookahead, y, physics)
#         else:
#             raise ValueError("loss_type must be 'intensity' or 'amplitude'")

#         if x_lookahead.grad is not None:
#             x_lookahead.grad.zero_()

#         loss.backward()

#         with torch.no_grad():
#             grad = x_lookahead.grad.clone()

#             v = beta * v - lr * grad
#             x = x + v

#             x_corrected = correct_global_phase(x.clone(), x_true)
#             recon_error = torch.sum(torch.abs(x_corrected - x_true) ** 2).item()
#             recon_error_hist.append(recon_error)

#         loss_hist.append(loss.item())

#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#         iter_end = time.perf_counter()
#         time_hist.append(iter_end - iter_start)

#         if verbose and i % 100 == 0:
#             print(
#                 f"[{i}] {loss_type}_loss={loss.item():.4f}  "
#                 f"||x_k-x*||^2={recon_error:.4f}  "
#                 f"lr={lr:.6e}"
#             )

#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     total_time = time.perf_counter() - start_time

#     return {
#         "x_final": x.detach(),
#         "loss_hist": loss_hist,
#         "recon_error_hist": recon_error_hist,
#         "time_hist": time_hist,
#         "total_time": total_time,
#         "avg_time": total_time / num_iter
#     }

# def adam(
#     x_init,
#     y,
#     physics,
#     x_true,
#     num_iter=1000,
#     lr=1e-2,
#     beta1=0.9,
#     beta2=0.999,
#     eps=1e-8,
#     loss_type="intensity",
#     verbose=True
# ):
#     import time
#     import torch

#     x = x_init.clone().detach().requires_grad_(True)

#     loss_hist = []
#     recon_error_hist = []
#     time_hist = []

#     m = torch.zeros_like(x)
#     v = torch.zeros_like(x)

#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     start_time = time.perf_counter()

#     for i in range(num_iter):
#         iter_start = time.perf_counter()

#         if loss_type == "intensity":
#             loss = intensity_loss(x, y, physics)
#         elif loss_type == "amplitude":
#             loss = amplitude_loss(x, y, physics)
#         else:
#             raise ValueError("loss_type must be 'intensity' or 'amplitude'")

#         if x.grad is not None:
#             x.grad.zero_()

#         loss.backward()

#         with torch.no_grad():
#             g = x.grad.clone()

#             m = beta1 * m + (1 - beta1) * g
#             v = beta2 * v + (1 - beta2) * (g.abs() ** 2)

#             t = i + 1
#             m_hat = m / (1 - beta1 ** t)
#             v_hat = v / (1 - beta2 ** t)

#             x -= lr * m_hat / (torch.sqrt(v_hat) + eps)

#             x_corrected = correct_global_phase(x.clone(), x_true)
#             recon_error = torch.sum(torch.abs(x_corrected - x_true) ** 2).item()
#             recon_error_hist.append(recon_error)

#         x = x.detach().requires_grad_(True)
#         loss_hist.append(loss.item())

#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#         iter_end = time.perf_counter()
#         time_hist.append(iter_end - iter_start)

#         if verbose and i % 100 == 0:
#             print(
#                 f"[{i}] {loss_type}_loss={loss.item():.4f}  "
#                 f"||x_k-x*||^2={recon_error:.4f}"
#             )

#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     total_time = time.perf_counter() - start_time

#     return {
#         "x_final": x.detach(),
#         "loss_hist": loss_hist,
#         "recon_error_hist": recon_error_hist,
#         "time_hist": time_hist,
#         "total_time": total_time,
#         "avg_time": total_time / num_iter
#     }

# def nonlinear_conjugate_gradient(
#     x_init,
#     y,
#     physics,
#     x_true,
#     num_iter=300,
#     loss_type="intensity",
#     beta_type="PR+",
#     alpha0=1.0,
#     rho=0.5,
#     c=1e-4,
#     min_alpha=1e-8,
#     max_ls_steps=20,
#     verbose=True
# ):
#     import time
#     import torch

#     x = x_init.clone().detach().requires_grad_(True)

#     loss_hist = []
#     recon_error_hist = []
#     time_hist = []
#     alpha_hist = []
#     beta_hist = []

#     prev_grad = None
#     prev_dir = None

#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     start_time = time.perf_counter()

#     for i in range(num_iter):
#         iter_start = time.perf_counter()

#         if loss_type == "intensity":
#             loss = intensity_loss(x, y, physics)
#         elif loss_type == "amplitude":
#             loss = amplitude_loss(x, y, physics)
#         else:
#             raise ValueError("loss_type must be 'intensity' or 'amplitude'")

#         if x.grad is not None:
#             x.grad.zero_()

#         loss.backward()

#         with torch.no_grad():
#             grad = x.grad.clone()
#             gk = grad.reshape(-1)

#             if i == 0:
#                 beta = torch.tensor(0.0, device=grad.device, dtype=torch.float32)
#                 direction = -grad
#             else:
#                 gprev = prev_grad.reshape(-1)
#                 dprev = prev_dir.reshape(-1)
#                 yk = gk - gprev

#                 if beta_type == "FR":
#                     numerator = torch.real(torch.sum(torch.conj(gk) * gk))
#                     denominator = torch.real(torch.sum(torch.conj(gprev) * gprev)) + 1e-12
#                     beta = numerator / denominator

#                 elif beta_type == "PR":
#                     numerator = torch.real(torch.sum(torch.conj(gk) * yk))
#                     denominator = torch.real(torch.sum(torch.conj(gprev) * gprev)) + 1e-12
#                     beta = numerator / denominator

#                 elif beta_type == "PR+":
#                     numerator = torch.real(torch.sum(torch.conj(gk) * yk))
#                     denominator = torch.real(torch.sum(torch.conj(gprev) * gprev)) + 1e-12
#                     beta = torch.clamp(numerator / denominator, min=0.0)

#                 elif beta_type == "HS":
#                     numerator = torch.real(torch.sum(torch.conj(gk) * yk))
#                     denominator = torch.real(torch.sum(torch.conj(dprev) * yk)) + 1e-12
#                     beta = numerator / denominator

#                 elif beta_type == "DY":
#                     numerator = torch.real(torch.sum(torch.conj(gk) * gk))
#                     denominator = torch.real(torch.sum(torch.conj(dprev) * yk)) + 1e-12
#                     beta = numerator / denominator

#                 elif beta_type == "CD":
#                     numerator = -torch.real(torch.sum(torch.conj(gk) * gk))
#                     denominator = torch.real(torch.sum(torch.conj(dprev) * gprev)) + 1e-12
#                     beta = numerator / denominator

#                 else:
#                     raise ValueError("beta_type must be one of: 'FR', 'PR', 'PR+', 'HS', 'DY', 'CD'")

#                 direction = -grad + beta * prev_dir

#                 dir_inner = torch.real(torch.sum(torch.conj(gk) * direction.reshape(-1)))
#                 if dir_inner >= 0:
#                     beta = torch.tensor(0.0, device=grad.device, dtype=torch.float32)
#                     direction = -grad

#             current_loss = loss.item()
#             grad_dir = torch.real(torch.sum(torch.conj(gk) * direction.reshape(-1))).item()

#             alpha = alpha0
#             ls_step = 0

#             if grad_dir >= 0:
#                 direction = -grad
#                 grad_dir = -torch.real(torch.sum(torch.conj(gk) * gk)).item()
#                 beta = torch.tensor(0.0, device=grad.device, dtype=torch.float32)

#             while ls_step < max_ls_steps:
#                 x_trial = x.detach() + alpha * direction

#                 if loss_type == "intensity":
#                     trial_loss = intensity_loss(x_trial, y, physics).item()
#                 else:
#                     trial_loss = amplitude_loss(x_trial, y, physics).item()

#                 if trial_loss <= current_loss + c * alpha * grad_dir:
#                     break

#                 alpha *= rho
#                 ls_step += 1

#                 if alpha < min_alpha:
#                     alpha = min_alpha
#                     break

#             x = (x.detach() + alpha * direction).requires_grad_(True)

#             x_corrected = correct_global_phase(x.detach().clone(), x_true)
#             recon_error = torch.sum(torch.abs(x_corrected - x_true) ** 2).item()

#             loss_hist.append(current_loss)
#             recon_error_hist.append(recon_error)
#             alpha_hist.append(alpha)
#             beta_hist.append(beta.item())

#             prev_grad = grad.clone()
#             prev_dir = direction.clone()

#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#         iter_end = time.perf_counter()
#         time_hist.append(iter_end - iter_start)

#         if verbose and i % 20 == 0:
#             print(
#                 f"[{i}] {loss_type}_loss={loss_hist[-1]:.6f}  "
#                 f"beta={beta_hist[-1]:.6f}  "
#                 f"alpha={alpha_hist[-1]:.3e}  "
#                 f"||x_k-x*||^2={recon_error_hist[-1]:.6f}"
#             )

#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     total_time = time.perf_counter() - start_time

#     return {
#         "x_final": x.detach(),
#         "loss_hist": loss_hist,
#         "recon_error_hist": recon_error_hist,
#         "time_hist": time_hist,
#         "alpha_hist": alpha_hist,
#         "beta_hist": beta_hist,
#         "total_time": total_time,
#         "avg_time": total_time / num_iter
#     }

# def amplitude_loss(x, y, physics):
#     Ax = physics.B.A(x)
#     return 0.5 * torch.norm(torch.abs(Ax) - torch.sqrt(y)) ** 2

# def stochastic_gradient_descent_torch(
#     x_init,
#     y,
#     physics,
#     x_true,
#     num_iter=1000,
#     step_size=1e-3,
#     batch_ratio=0.1,
#     step_mode="constant",   # "constant", "exp_decay", "inverse_decay"
#     initial_lr=1e-3,
#     decay_rate=0.99,
#     loss_type="intensity",
#     momentum=0.0,
#     optimizer_type="sgd",   # "sgd" or "adam"
#     verbose=True
# ):
#     import time
#     import torch

#     device = x_init.device

#     x = torch.nn.Parameter(x_init.clone().detach().to(device))
#     y = y.to(device)
#     x_true = x_true.to(device)

#     if optimizer_type == "sgd":
#         optimizer = torch.optim.SGD([x], lr=step_size, momentum=momentum)
#     elif optimizer_type == "adam":
#         optimizer = torch.optim.Adam([x], lr=step_size)
#     else:
#         raise ValueError("optimizer_type must be 'sgd' or 'adam'")

#     loss_hist = []
#     recon_error_hist = []
#     time_hist = []

#     y_flat = y.flatten()
#     n_measurements = y_flat.shape[0]
#     batch_size = max(1, int(n_measurements * batch_ratio))

#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     start_time = time.perf_counter()

#     for i in range(num_iter):
#         iter_start = time.perf_counter()

#         # learning rate schedule
#         if step_mode == "constant":
#             lr = step_size
#         elif step_mode == "exp_decay":
#             lr = initial_lr * (decay_rate ** i)
#         elif step_mode == "inverse_decay":
#             lr = initial_lr / (1 + decay_rate * i)
#         else:
#             raise ValueError("Unknown step_mode")

#         for g in optimizer.param_groups:
#             g["lr"] = lr

#         optimizer.zero_grad()

#         idx = torch.randperm(n_measurements, device=y.device)[:batch_size]

#         Bx_full = physics.B.A(x).flatten()
#         Bx_batch = Bx_full[idx]
#         y_batch = y_flat[idx]

#         if loss_type == "intensity":
#             loss = torch.sum((Bx_batch.abs().square() - y_batch) ** 2)
#         elif loss_type == "amplitude":
#             loss = torch.sum((Bx_batch.abs() - torch.sqrt(y_batch + 1e-12)) ** 2)
#         else:
#             raise ValueError("loss_type must be 'intensity' or 'amplitude'")

#         loss.backward()
#         optimizer.step()

#         with torch.no_grad():
#             x_corrected = correct_global_phase(x.detach().clone(), x_true)
#             recon_error = torch.sum(torch.abs(x_corrected - x_true) ** 2).item()

#         loss_hist.append(loss.item())
#         recon_error_hist.append(recon_error)

#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#         iter_end = time.perf_counter()
#         time_hist.append(iter_end - iter_start)

#         if verbose and i % 100 == 0:
#             print(f"[{i}] loss={loss.item():.4f}  ||x_k-x*||^2={recon_error:.4f}  lr={lr:.6f}")

#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     total_time = time.perf_counter() - start_time

#     return {
#         "x_final": x.detach(),
#         "loss_hist": loss_hist,
#         "recon_error_hist": recon_error_hist,
#         "time_hist": time_hist,
#         "total_time": total_time,
#         "avg_time": total_time / num_iter
#     }