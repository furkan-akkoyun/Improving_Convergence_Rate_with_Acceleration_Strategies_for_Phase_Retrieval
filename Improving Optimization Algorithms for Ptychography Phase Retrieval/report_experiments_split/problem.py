"""Problem setup and physics wrappers for report experiments."""
import deepinv as dinv
import numpy as np
import torch
from deepinv.utils import load_example

import src.algorithms as alg_impl

from .config import SEED, device


class DenseLinearOperator:
    def __init__(self, matrix, signal_shape):
        self.matrix = matrix
        self.signal_shape = signal_shape

    def A(self, x):
        flat = x.reshape(-1)
        return self.matrix @ flat

    def A_adjoint(self, z):
        return (self.matrix.conj().T @ z.reshape(-1)).reshape(self.signal_shape)


class DenseRandomPhaseRetrieval:
    """Dense random sensing model with the same interface used by src.losses."""

    def __init__(self, m, img_size, signal_shape, device):
        self.m = int(m)
        self.img_size = tuple(img_size)
        self.signal_shape = tuple(signal_shape)
        n = int(np.prod(self.signal_shape))
        real = torch.randn(self.m, n, device=device)
        imag = torch.randn(self.m, n, device=device)
        matrix = (real + 1j * imag) / torch.sqrt(torch.tensor(2 * n, device=device))
        self.B = DenseLinearOperator(matrix, self.signal_shape)

    def A(self, x):
        return torch.abs(self.B.A(x)) ** 2

    def __call__(self, x):
        return self.A(x)

    def A_dagger(self, y, n_iter=80):
        n = int(np.prod(self.signal_shape))
        x = torch.randn(n, device=y.device) + 1j * torch.randn(n, device=y.device)
        x = x / torch.linalg.norm(x)
        y_flat = y.reshape(-1)
        y_clip = torch.clamp(y_flat, max=torch.quantile(y_flat.float(), 0.95))
        for _ in range(n_iter):
            x = self.B.A_adjoint(y_clip * self.B.A(x.reshape(self.signal_shape))).reshape(-1)
            x = x / (torch.linalg.norm(x) + 1e-12)
        scale = torch.sqrt(torch.mean(y_flat.float()) / (torch.mean(torch.abs(self.B.A(x.reshape(self.signal_shape))) ** 2).float() + 1e-12))
        return (scale * x).reshape(self.signal_shape)


img_size = 32
alpha_default = 4.0

x = load_example(
    "SheppLogan.png",
    img_size=img_size,
    grayscale=True,
    resize_mode="resize",
    device=device,
)
x_phase = torch.exp(1j * x * torch.pi - 0.5j * torch.pi)
x_true = x_phase.clone().detach()
x_true_vis = torch.angle(x_true) / torch.pi + 0.5
phase_img_size = x.shape[1:]
signal_shape = x_true.shape
n_pixels = int(torch.prod(torch.tensor(signal_shape)))

print("image shape:", tuple(x.shape), "phase img size:", tuple(phase_img_size), "signal shape:", tuple(signal_shape), "n:", n_pixels)


def sync_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def safe_correct_global_phase(x_recon, x, threshold=1e-5, verbose=False):
    """Shape-tolerant global phase correction used by src.algorithms in this notebook."""
    if x_recon.shape != x.shape:
        if x_recon.numel() == x.numel():
            x_recon = x_recon.reshape_as(x)
        else:
            raise ValueError(f"x_recon shape {tuple(x_recon.shape)} is incompatible with x shape {tuple(x.shape)}")

    x_flat = x.reshape(x.shape[0], -1)
    r_flat = x_recon.reshape(x_recon.shape[0], -1)
    phase = torch.sum(torch.conj(r_flat) * x_flat, dim=1)
    phase = phase / (torch.abs(phase) + 1e-12)
    view_shape = (x_recon.shape[0],) + (1,) * (x_recon.ndim - 1)
    return x_recon * phase.reshape(view_shape)


# Patch the module global used inside functions imported from src.algorithms.
alg_impl.correct_global_phase = safe_correct_global_phase
correct_global_phase = safe_correct_global_phase


def structured_output_size(alpha):
    # DeepInv StructuredRandomPhaseRetrieval controls oversampling through output_size.
    # For alpha=4 and img_size=(1, 32, 32), this gives output_size=(1, 64, 64).
    channels, height, width = phase_img_size
    scale = float(alpha) ** 0.5
    out_h = max(height, int(np.ceil(height * scale)))
    out_w = max(width, int(np.ceil(width * scale)))
    return (channels, out_h, out_w)


def make_physics(model, alpha):
    m = int(alpha * n_pixels)
    if model == "dense":
        return DenseRandomPhaseRetrieval(m=m, img_size=phase_img_size, signal_shape=signal_shape, device=device)
    if model == "structured":
        return dinv.physics.StructuredRandomPhaseRetrieval(
            img_size=phase_img_size,
            output_size=structured_output_size(alpha),
            n_layers=2,
            transform="fft",
            diagonal_mode="uniform_phase",
            device=device,
        )
    if model in ["deepinv_random", "original_random", "random_phase_retrieval"]:
        return dinv.physics.RandomPhaseRetrieval(m=m, img_size=phase_img_size, device=device)
    raise ValueError(model)


def make_measurements(physics, noise_level=0.0):
    y_clean = physics(x_true)
    if noise_level <= 0:
        return y_clean
    noise = torch.randn_like(y_clean)
    noise = noise / (torch.linalg.norm(noise) + 1e-12) * noise_level * torch.linalg.norm(y_clean)
    return torch.clamp(y_clean + noise, min=0.0)


def ensure_signal_shape(z, name="tensor"):
    """Keep src.algorithms compatible with deepinv.correct_global_phase."""
    z = z.clone().detach()
    if z.shape == x_true.shape:
        return z
    if z.numel() == x_true.numel():
        return z.reshape_as(x_true)
    raise ValueError(f"{name} has shape {tuple(z.shape)}, expected {tuple(x_true.shape)}")


def make_initialization(physics, y, initialization):
    if initialization == "spectral":
        return ensure_signal_shape(physics.A_dagger(y, n_iter=300), "spectral initialization")
    if initialization == "random":
        return ensure_signal_shape(torch.exp(1j * torch.rand_like(x_true.real) * 2 * torch.pi), "random initialization")
    raise ValueError(initialization)
