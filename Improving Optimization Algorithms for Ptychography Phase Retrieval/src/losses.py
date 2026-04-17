import torch
from deepinv.optim.prior import Prior
import numpy as np
import torch
import matplotlib.pyplot as plt
from deepinv.utils.plotting import plot
from deepinv.optim.phase_retrieval import (
    correct_global_phase,
)

def amplitude_loss(x, y, physics):
    Ax = physics.B.A(x)
    return 0.5 * torch.norm(torch.abs(Ax) - torch.sqrt(y)) ** 2

def intensity_loss(x, y, physics):
    Ax = physics.A(x)
    return torch.sum((Ax - y)**2)