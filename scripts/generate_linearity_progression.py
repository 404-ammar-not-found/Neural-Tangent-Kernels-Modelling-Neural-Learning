"""
Visualise gradual linearisation of neural network error planes
as network width increases.

Generates 3 plots for widths:
10 → 1000 → 10000
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src import models
from src.utils import set_random_seeds


def generate_data(n_samples=50, noise_std=0.1):
    """Generate regression data."""
    x = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    y = np.sin(x.flatten()) + noise_std * np.random.randn(n_samples)
    return x.flatten(), y


# -------------------------
# Parameter helpers
# -------------------------

def flatten_params(params):
    """Convert parameter list to flat vector."""
    flat = []
    shapes = []

    for W, b in params:
        shapes.append(W.shape)
        flat.append(W.flatten())

        shapes.append(b.shape)
        flat.append(b.flatten())

    return np.concatenate(flat), shapes


def unflatten_params(theta, params_template):
    """Convert flat vector back to parameter list."""
    new_params = []
    idx = 0

    for W, b in params_template:

        W_size = W.size
        b_size = b.size

        new_W = theta[idx:idx + W_size].reshape(W.shape)
        idx += W_size

        new_b = theta[idx:idx + b_size].reshape(b.shape)
        idx += b_size

        new_params.append((new_W, new_b))

    return new_params


# -------------------------
# Loss landscape computation
# -------------------------

def compute_error_plane(width, x_data, y_data, grid=25):

    params = models.initialize_weights(width=width, n_layers=3)

    theta, _ = flatten_params(params)

    # random directions
    d1 = np.random.randn(theta.size)
    d2 = np.random.randn(theta.size)

    d1 /= np.linalg.norm(d1)
    d2 /= np.linalg.norm(d2)

    # scale perturbation with width
    param_range = 1 / np.sqrt(width)

    alphas = np.linspace(-param_range, param_range, grid)
    betas = np.linspace(-param_range, param_range, grid)

    X, Y = np.meshgrid(alphas, betas)
    Z = np.zeros_like(X)

    def compute_loss(p):
        preds = models.forward_batch(x_data, p)
        return 0.5 * np.mean((preds - y_data) ** 2)

    for i in range(grid):
        for j in range(grid):

            theta_new = theta + X[i, j] * d1 + Y[i, j] * d2
            params_new = unflatten_params(theta_new, params)

            Z[i, j] = compute_loss(params_new)

    # Fit linear plane
    A = np.column_stack([X.flatten(), Y.flatten(), np.ones(X.size)])
    coeffs, _, _, _ = np.linalg.lstsq(A, Z.flatten(), rcond=None)

    linear_fit = coeffs[0]*X + coeffs[1]*Y + coeffs[2]

    ss_res = np.sum((Z - linear_fit) ** 2)
    ss_tot = np.sum((Z - np.mean(Z)) ** 2)

    r2 = 1 - ss_res / ss_tot

    return X, Y, Z, linear_fit, r2


# -------------------------
# Plotting
# -------------------------

def plot_linearity():

    widths = [10, 1000, 10000]

    x_data, y_data = generate_data()

    fig = plt.figure(figsize=(18, 6))

    colors = ['#050515', '#0a0a1a', '#94c544']
    cmap = LinearSegmentedColormap.from_list("ntk", colors, N=256)

    for i, width in enumerate(widths):

        print(f"Computing width {width}")

        X, Y, Z, linear_fit, r2 = compute_error_plane(width, x_data, y_data)

        ax = fig.add_subplot(1, 3, i+1, projection='3d')

        ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.85, edgecolor='none')

        ax.plot_surface(X, Y, linear_fit, color='red', alpha=0.35)

        ax.set_title(
            f"Width = {width}\nR² = {r2:.4f}",
            fontsize=13,
            fontweight='bold'
        )

        ax.set_xlabel("Direction 1")
        ax.set_ylabel("Direction 2")
        ax.set_zlabel("Loss")

    fig.suptitle(
        "Gradual Linearisation of Neural Network Error Planes",
        fontsize=16,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig("figures/error_plane_linearisation.png", dpi=300)
    plt.show()


# -------------------------
# Main
# -------------------------

def main():

    set_random_seeds()

    plot_linearity()


if __name__ == "__main__":
    main()