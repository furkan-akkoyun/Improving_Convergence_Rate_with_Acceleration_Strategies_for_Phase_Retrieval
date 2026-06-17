"""Report hyperparameters from Appendix A.

The values in this file mirror the optimizer hyperparameter tables in the
final report. Shared optimizer defaults that are fixed in the report are:
accelerated GD decay_rate=0.99 and beta=0.9; Adam beta1=0.9, beta2=0.999,
eps=1e-8; L-BFGS uses strong Wolfe line search in runners.py.
"""

BASELINE_CONFIGS = {
    "GD": {"num_iter": 300, "step_size": 1e-1},
    "Accelerated GD": {"num_iter": 300, "initial_lr": 1e-1, "decay_rate": 0.99, "beta": 0.9},
    "Adam": {"num_iter": 300, "lr": 1e-1},
    "NCG": {"num_iter": 120, "alpha0": 1.0, "max_ls_steps": 20},
    "L-BFGS": {"num_iter": 80, "lr": 1.0, "history_size": 10},
}

AMPLITUDE_LOSS_CONFIGS = {
    "GD": {"num_iter": 300, "step_size": 1e-1},
    "Accelerated GD": {"num_iter": 300, "initial_lr": 1e-1, "decay_rate": 0.99, "beta": 0.9},
    "Adam": {"num_iter": 300, "lr": 1e-1},
    "NCG": {"num_iter": 300, "alpha0": 1.0, "max_ls_steps": 20},
    "L-BFGS": {"num_iter": 80, "lr": 1.0, "history_size": 10},
}

RANDOM_INITIALIZATION_CONFIGS = {
    "GD": {"num_iter": 1000, "step_size": 4e-4},
    "Accelerated GD": {"num_iter": 1000, "initial_lr": 9e-5, "decay_rate": 0.99, "beta": 0.9},
    "Adam": {"num_iter": 1000, "lr": 4e-4},
    "NCG": {"num_iter": 1000, "alpha0": 5e-4, "max_ls_steps": 20},
    "L-BFGS": {"num_iter": 300, "lr": 1e-6, "history_size": 10},
}

STRUCTURED_MODEL_CONFIGS = {
    "GD": {"num_iter": 300, "step_size": 2e-1},
    "Accelerated GD": {"num_iter": 300, "initial_lr": 1e-1, "decay_rate": 0.99, "beta": 0.9},
    "Adam": {"num_iter": 300, "lr": 1e-1},
    "NCG": {"num_iter": 120, "alpha0": 1.0, "max_ls_steps": 20},
    "L-BFGS": {"num_iter": 80, "lr": 1.0, "history_size": 10},
}

OVERSAMPLING_CONFIGS = {
    "GD": {"num_iter": 300, "step_size": 1e-4},
    "Accelerated GD": {"num_iter": 300, "initial_lr": 1e-4, "decay_rate": 0.99, "beta": 0.9},
    "Adam": {"num_iter": 300, "lr": 3e-3},
    "NCG": {"num_iter": 300, "alpha0": 5e-4, "max_ls_steps": 200},
    "L-BFGS": {"num_iter": 80, "lr": 3e-7, "history_size": 100},
}

OVERSAMPLING_CONFIGS_BY_ALPHA = {
    2.0: OVERSAMPLING_CONFIGS,
    10.0: {
        "GD": {"num_iter": 300, "step_size": 1.0},
        "Accelerated GD": {"num_iter": 300, "initial_lr": 0.6, "decay_rate": 0.99, "beta": 0.9},
        "Adam": {"num_iter": 300, "lr": 0.7},
        "NCG": {"num_iter": 120, "alpha0": 1.0, "max_ls_steps": 20},
        "L-BFGS": {"num_iter": 80, "lr": 10.0, "history_size": 10},
    },
}

NOISY_MEASUREMENTS_CONFIGS = {
    "GD": {"num_iter": 300, "step_size": 9e-2},
    "Accelerated GD": {"num_iter": 300, "initial_lr": 1e-1, "decay_rate": 0.99, "beta": 0.9},
    "Adam": {"num_iter": 300, "lr": 1e-1},
    "NCG": {"num_iter": 300, "alpha0": 1e-1, "max_ls_steps": 200},
    "L-BFGS": {"num_iter": 80, "lr": 1.0, "history_size": 10},
}

BASELINE_ALPHA = 4.0
OVERSAMPLING_ALPHAS = [2.0, 10.0]
NOISY_MEASUREMENT_LEVEL = 0.3
