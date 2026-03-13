"""
Generate Neural Tangent Kernel (NTK) visualization figures.
This script creates visualizations demonstrating the key concepts of NTK theory.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.neural_network import MLPRegressor
from scipy.ndimage import gaussian_filter1d

from src import config
from src import data
from src import models
from src import visualization
from src.utils import set_random_seeds, compute_drift


def create_ntk_demonstration(save_path='ntk_demonstration.png'):
    """
    Create NTK demonstration figure showing NN fitting vs NTK prediction.
    
    Args:
        save_path: Output file path
    
    Returns:
        Path to saved figure
    """
    print("Creating NTK demonstration figure...")
    
    # Set random seeds
    np.random.seed(config.RANDOM_SEED)
    
    # Generate sin(x) data with noise
    x = np.linspace(0, 2*np.pi, 100).reshape(-1, 1)
    y_true = np.sin(x).flatten()
    y_noisy = y_true + np.random.normal(0, 0.3, size=y_true.shape)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Graph 1: Noisy data points
    ax = axes[0]
    ax.set_facecolor(config.COLORS['panel'])
    visualization.style_axes(ax)
    
    ax.scatter(x, y_noisy, color=config.COLORS['primary'], s=50, label='Data', alpha=0.7)
    ax.set_xlabel('x', fontsize=12, color=config.COLORS['white'])
    ax.set_ylabel('y', fontsize=12, color=config.COLORS['white'])
    ax.set_title('Graph 1: Noisy sin(x) Data', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, alpha=0.3)
    
    # Set custom x-axis labels
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    
    # Graph 2: Neural Network fit
    ax = axes[1]
    ax.set_facecolor(config.COLORS['panel'])
    visualization.style_axes(ax)
    
    # Create and train neural network
    nn_model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=config.RANDOM_SEED)
    nn_model.fit(x, y_noisy)
    y_nn = nn_model.predict(x)
    
    ax.scatter(x, y_noisy, color=config.COLORS['primary'], s=50, alpha=0.7)
    ax.plot(x, y_nn, color=config.COLORS['primary'], linewidth=2, label='NN Fit')
    ax.set_xlabel('x', fontsize=12, color=config.COLORS['white'])
    ax.set_ylabel('y', fontsize=12, color=config.COLORS['white'])
    ax.set_title('Graph 2: Neural Network Fit', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    
    # Graph 3: NTK Prediction (smooth line overlapping NN)
    ax = axes[2]
    ax.set_facecolor(config.COLORS['panel'])
    visualization.style_axes(ax)
    
    # Use smoothed version representing NTK
    y_ntk = gaussian_filter1d(y_nn, sigma=3)
    
    ax.scatter(x, y_noisy, color=config.COLORS['primary'], s=50, alpha=0.7, label='Data')
    ax.plot(x, y_nn, color=config.COLORS['primary'], linewidth=2, label='NN Fit', alpha=0.5)
    ax.plot(x, y_ntk, color=config.COLORS['danger'], linewidth=3, label='NTK Prediction')
    ax.set_xlabel('x', fontsize=12, color=config.COLORS['white'])
    ax.set_ylabel('y', fontsize=12, color=config.COLORS['white'])
    ax.set_title('Graph 3: NTK Prediction', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    ax.legend(loc='upper right')
    
    # Add main title and caption
    fig.suptitle('Demonstration: Fitting sin(x) - Neural Networks vs Neural Tangent Kernels', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    figure_path = visualization.save_figure(fig, save_path)
    
    return figure_path


def create_frozen_kernel_comparison(save_path='frozen_kernel_comparison.png'):
    """
    Create figure comparing finite vs infinite width kernel drift.
    
    Args:
        save_path: Output file path
    
    Returns:
        Path to saved figure
    """
    print("Creating frozen kernel comparison figure...")
    
    # Set random seeds
    np.random.seed(config.RANDOM_SEED)
    
    # Define custom diverging colormap
    cmap_colors = [
        (0.05, 0.10, 0.35),  # Deep blue
        (0.08, 0.08, 0.15),  # Near black
        (0.94, 0.75, 0.22),  # Amber
    ]
    HEATMAP = LinearSegmentedColormap.from_list("ntk", cmap_colors, N=256)
    
    # Data
    n_points = 12
    x_train = np.linspace(-2, 2, n_points)
    y_train = np.sin(x_train)
    n_steps = 60
    
    # Compute kernels
    results = {}
    for label, width in [("Finite (width = 8)", 8), ("Infinite (width = 512)", 512)]:
        params_init = models.initialize_weights(width=width)
        K_init = models.compute_ntk_matrix(x_train, params_init)
        
        params = params_init
        for _ in range(n_steps):
            params = models.train_step(params, x_train, y_train, lr=0.002)
        K_trained = models.compute_ntk_matrix(x_train, params)
        
        drift = compute_drift(K_init, K_trained)
        results[label] = {"K_init": K_init, "K_trained": K_trained, "drift": drift}
    
    # Plot
    fig = plt.figure(figsize=(15, 7.5))
    fig.suptitle("The Frozen Kernel: NTK at Initialisation vs After Training",
                fontsize=15, fontweight="bold")
    
    # Determine color scale for fair comparison
    vmin = min(data["K_init"].min() for data in results.values())
    vmax = max(data["K_init"].max() for data in results.values())
    
    # Create grid layout
    outer = gridspec.GridSpec(1, 2, figure=fig,
                            left=0.05, right=0.97,
                            top=0.88, bottom=0.08,
                            wspace=0.10)
    
    for col_idx, (net_label, data) in enumerate(results.items()):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[col_idx], wspace=0.08
        )
        color = config.COLORS['danger'] if "Finite" in net_label else config.COLORS['cyan']
        
        for row_idx, (stage, K) in enumerate([
            ("Initialisation", data["K_init"]),
            ("After Training", data["K_trained"]),
        ]):
            ax = fig.add_subplot(inner[row_idx])
            ax.set_facecolor(config.COLORS['panel'])
            
            for spine in ax.spines.values():
                spine.set_edgecolor(config.COLORS['grey'])
            ax.tick_params(colors=config.COLORS['white'], labelsize=7)
            
            im = ax.imshow(K, cmap=HEATMAP, vmin=vmin, vmax=vmax,
                          aspect="auto", interpolation="nearest")
            
            ax.set_title(stage, color=config.COLORS['white'], fontsize=9,
                        fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add colorbar only for rightmost plot
            if row_idx == 1:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.06)
                cb = fig.colorbar(im, cax=cax)
                cb.ax.tick_params(colors=config.COLORS['white'], labelsize=6)
                cb.outline.set_edgecolor(config.COLORS['grey'])
        
        # Add section label
        inner_pos = outer[col_idx].get_position(fig)
        xc = (inner_pos.x0 + inner_pos.x1) / 2
        fig.text(xc, 0.935, net_label, ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=color)
        
        # Add drift annotation
        drift_pct = data["drift"] * 100
        drift_color = config.COLORS['danger'] if data["drift"] > 0.05 else config.COLORS['cyan']
        fig.text(xc, 0.02, f"Kernel drift: {drift_pct:.1f}%", 
                ha="center", va="bottom", fontsize=10, 
                color=drift_color, fontstyle="italic")
    
    # Add center annotation
    fig.text(0.50, 0.50,
             "← kernel drifts        kernel freezes →",
             ha="center", va="center", fontsize=9,
             color=config.COLORS['dim'], style="italic",
             bbox=dict(boxstyle="round,pad=0.4", 
                      facecolor=config.COLORS['bg'], 
                      edgecolor=config.COLORS['grey'], 
                      lw=0.8))
    
    plt.tight_layout()
    figure_path = visualization.save_figure(fig, save_path)
    
    return figure_path


def create_ntk_training_animation(save_path='ntk_training_animation.gif'):
    """
    Create animation showing NTK training dynamics.
    
    Args:
        save_path: Output file path
    
    Returns:
        Path to saved animation
    """
    print("Creating NTK training animation...")
    
    # Set random seeds
    np.random.seed(config.RANDOM_SEED)
    
    # Network parameters
    x0, y0 = 1.0, 0.8
    params = models.initialize_params(width=128)
    f0 = models.forward_scalar(x0, params)
    K = models.compute_ntk_matrix(np.array([x0]), params)[0, 0]
    
    n_frames = 80
    lr = 0.005
    xs = np.linspace(-3, 3, 300)
    
    # Pre-compute all frames
    f_pred_history = []
    K_history = []
    curve_history = []
    params_cur = params
    
    for frame in range(n_frames):
        f_pred_history.append(models.forward_scalar(x0, params_cur))
        K_history.append(models.compute_ntk_matrix(np.array([x0]), params_cur)[0, 0])
        curve_history.append([models.forward_scalar(x, params_cur) for x in xs])
        params_cur = models.train_step(params_cur, np.array([x0]), np.array([y0]), lr=lr)
    
    # Analytical trajectory
    t_vals = np.linspace(0, n_frames*lr, 300)
    f_analytic = y0 + (f0 - y0) * np.exp(-K * t_vals)
    
    # Create figure layout
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(r"NTK: Single Training Point – $f(t) = y_0 + (f(0)-y_0)\,e^{-Kt}$",
                fontsize=13, fontweight="bold")
    
    gs = gridspec.GridSpec(2, 2, figure=fig,
                        left=0.07, right=0.97,
                        top=0.90, bottom=0.08,
                        hspace=0.45, wspace=0.32)
    
    def create_subplot(subplot_spec, title):
        ax = fig.add_subplot(subplot_spec)
        ax.set_facecolor(config.COLORS['panel'])
        for spine in ax.spines.values():
            spine.set_edgecolor(config.COLORS['grey'])
        ax.tick_params(colors=config.COLORS['white'], labelsize=8)
        ax.set_title(title, color=config.COLORS['white'], fontsize=9, fontweight="bold")
        return ax
    
    # Create subplots
    ax_net = create_subplot(gs[0, 0], "Network function f(x)")
    ax_conv = create_subplot(gs[0, 1], "Training convergence f(x₀) vs time")
    ax_K = create_subplot(gs[1, 0], "Kernel value K(t) — finite vs infinite width")
    ax_err = create_subplot(gs[1, 1], "Error |f(x₀) − y₀| (log scale)")
    
    # Static elements
    # ax_net
    ax_net.axhline(y0, color=config.COLORS['gold'], lw=1.2, ls="--", alpha=0.6, label=f"y₀ = {y0}")
    ax_net.axvline(x0, color=config.COLORS['grey'], lw=0.8, ls=":")
    ax_net.set_xlim(-3, 3)
    ax_net.set_ylim(-2, 2)
    ax_net.set_xlabel("x", fontsize=8, color=config.COLORS['white'])
    ax_net.set_ylabel("f(x)", fontsize=8, color=config.COLORS['white'])
    ax_net.legend(fontsize=7)
    
    # ax_conv
    t_frames = np.arange(n_frames) * lr
    ax_conv.plot(t_vals, f_analytic, color=config.COLORS['gold'], lw=1.5, ls="--",
                label=f"Analytical e^(-{K:.2f}·t)", alpha=0.8)
    ax_conv.axhline(y0, color=config.COLORS['green'], lw=0.8, ls=":", alpha=0.6, label=f"y₀ = {y0}")
    ax_conv.set_xlim(0, n_frames*lr)
    ax_conv.set_ylim(min(f0, y0)-0.3, max(f0, y0)+0.3)
    ax_conv.set_xlabel("training time t", fontsize=8, color=config.COLORS['white'])
    ax_conv.set_ylabel("f(x₀)", fontsize=8, color=config.COLORS['white'])
    ax_conv.legend(fontsize=7)
    
    # ax_K
    ax_K.axhline(K, color=config.COLORS['cyan'], lw=1.5, ls="--", alpha=0.8,
                label=f"Infinite width K = {K:.2f} (frozen)")
    ax_K.set_xlim(0, n_frames*lr)
    ax_K.set_xlabel("training time t", fontsize=8, color=config.COLORS['white'])
    ax_K.set_ylabel("K(t)", fontsize=8, color=config.COLORS['white'])
    ax_K.legend(fontsize=7)
    ax_K.set_ylim(0, K*2)
    
    # ax_err
    ax_err.axhline(0.01, color=config.COLORS['grey'], lw=0.7, ls=":", alpha=0.5)
    ax_err.set_xlim(0, n_frames*lr)
    ax_err.set_xlabel("training time t", fontsize=8, color=config.COLORS['white'])
    ax_err.set_ylabel("|f(x₀) − y₀|", fontsize=8, color=config.COLORS['white'])
    ax_err.set_yscale("log")
    
    # Animated elements
    curve_line, = ax_net.plot([], [], color=config.COLORS['cyan'], lw=2, label="f(x) network")
    train_dot, = ax_net.plot([], [], 'o', color=config.COLORS['red'], ms=8, zorder=5, label="(x₀, y₀)")
    ax_net.legend(fontsize=7)
    
    conv_line, = ax_conv.plot([], [], color=config.COLORS['cyan'], lw=2, label="Actual f(x₀)")
    conv_dot, = ax_conv.plot([], [], 'o', color=config.COLORS['cyan'], ms=6, zorder=5)
    ax_conv.legend(fontsize=7)
    
    K_line, = ax_K.plot([], [], color=config.COLORS['red'], lw=1.8, label="Finite width K(t)")
    ax_K.legend(fontsize=7)
    
    err_line, = ax_err.plot([], [], color=config.COLORS['gold'], lw=2, label="|error| actual")
    err_analytic, = ax_err.plot(t_vals, 
                               np.abs((f0-y0)*np.exp(-K*t_vals)) + 1e-8,
                               color=config.COLORS['gold'], lw=1, ls="--", 
                               alpha=0.5, label="analytical")
    ax_err.legend(fontsize=7)
    
    # Time indicator lines
    t_vline_conv = ax_conv.axvline(0, color=config.COLORS['white'], lw=0.8, alpha=0.4)
    t_vline_K = ax_K.axvline(0, color=config.COLORS['white'], lw=0.8, alpha=0.4)
    t_vline_err = ax_err.axvline(0, color=config.COLORS['white'], lw=0.8, alpha=0.4)
    
    # K annotation
    K_text = ax_K.text(0.97, 0.90, "", transform=ax_K.transAxes,
                      ha="right", va="top", fontsize=8, color=config.COLORS['white'],
                      bbox=dict(boxstyle="round,pad=0.3", facecolor=config.COLORS['bg'], 
                               edgecolor=config.COLORS['grey'], lw=0.7))
    
    # Step text
    step_text = fig.text(0.50, 0.005, "", ha="center", fontsize=8,
                        color=config.COLORS['dim'])
    
    def update(frame):
        t = frame * lr
        
        # Network curve
        curve_line.set_data(xs, curve_history[frame])
        train_dot.set_data([x0], [y0])
        
        # Convergence
        conv_line.set_data(t_frames[:frame+1], f_pred_history[:frame+1])
        conv_dot.set_data([t], [f_pred_history[frame]])
        
        # Kernel
        K_line.set_data(t_frames[:frame+1], K_history[:frame+1])
        K_text.set_text(f"K(t) = {K_history[frame]:.3f}\nK(0) = {K:.3f}")
        
        # Error
        errs = [abs(fp - y0) + 1e-8 for fp in f_pred_history[:frame+1]]
        err_line.set_data(t_frames[:frame+1], errs)
        
        # Time markers
        for vl in (t_vline_conv, t_vline_K, t_vline_err):
            vl.set_xdata([t])
        
        # Step text
        step_text.set_text(f"step {frame+1}/{n_frames}   |   t = {t:.3f}   |   "
                          f"f(x₀) = {f_pred_history[frame]:.4f}   |   "
                          f"error = {abs(f_pred_history[frame]-y0):.4f}")
        
        return (curve_line, train_dot, conv_line, conv_dot,
                K_line, err_line, t_vline_conv, t_vline_K, t_vline_err,
                K_text, step_text)
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=n_frames, interval=80, blit=True)
    
    # Save animation
    animation_path = visualization.save_animation(anim, save_path)
    
    return animation_path


def create_ntk_theory_comparison(save_path='ntk_theory_comparison.png'):
    """
    Create figure comparing NTK theory with practice.
    
    Args:
        save_path: Output file path
    
    Returns:
        Path to saved figure
    """
    print("Creating NTK theory comparison figure...")
    
    # Set random seeds
    np.random.seed(config.RANDOM_SEED)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Neural Tangent Kernel Theory vs Practice', fontsize=16, fontweight='bold')
    
    # Plot 1: NTK Matrix Structure
    ax = axes[0, 0]
    ax.set_facecolor(config.COLORS['panel'])
    visualization.style_axes(ax)
    
    # Generate example NTK matrix
    n_points = 10
    x_points = np.linspace(-2, 2, n_points)
    params = models.initialize_weights(width=100)
    K = models.compute_ntk_matrix(x_points, params)
    
    im = ax.imshow(K, cmap='viridis', aspect='auto')
    ax.set_title('NTK Matrix Structure', fontsize=12, fontweight='bold')
    ax.set_xlabel('Input Point Index', fontsize=10)
    ax.set_ylabel('Input Point Index', fontsize=10)
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, label='NTK Value')
    
    # Plot 2: Width vs NTK Stability
    ax = axes[0, 1]
    ax.set_facecolor(config.COLORS['panel'])
    visualization.style_axes(ax)
    
    widths = [10, 50, 100, 500, 1000]
    stabilities = []
    
    x_test = np.linspace(-2, 2, 20)
    for width in widths:
        params = models.initialize_weights(width=width)
        K1 = models.compute_ntk_matrix(x_test, params)
        
        # Small training step
        params_new = models.train_step(params, x_test[:5], np.sin(x_test[:5]), lr=0.001)
        K2 = models.compute_ntk_matrix(x_test, params_new)
        
        stability = 1 - compute_drift(K1, K2)
        stabilities.append(stability)
    
    ax.plot(widths, stabilities, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Network Width', fontsize=10)
    ax.set_ylabel('NTK Stability (1 - drift)', fontsize=10)
    ax.set_title('NTK Stability vs Network Width', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Training Curves
    ax = axes[1, 0]
    ax.set_facecolor(config.COLORS['panel'])
    visualization.style_axes(ax)
    
    # Compare finite vs infinite width training
    x_train = np.linspace(-2, 2, 15)
    y_train = np.sin(x_train)
    
    # Finite width training
    width_finite = 50
    params_finite = models.initialize_weights(width=width_finite)
    losses_finite = []
    
    for _ in range(100):
        predictions = models.forward_batch(x_train, params_finite)
        loss = 0.5 * np.mean((predictions - y_train) ** 2)
        losses_finite.append(loss)
        params_finite = models.train_step(params_finite, x_train, y_train, lr=0.01)
    
    # Infinite width approximation
    params_inf = models.initialize_weights(width=1000)
    losses_inf = []
    
    for _ in range(100):
        predictions = models.forward_batch(x_train, params_inf)
        loss = 0.5 * np.mean((predictions - y_train) ** 2)
        losses_inf.append(loss)
        params_inf = models.train_step(params_inf, x_train, y_train, lr=0.01)
    
    iterations = np.arange(len(losses_finite))
    ax.plot(iterations, losses_finite, 'r-', linewidth=2, label='Finite Width')
    ax.plot(iterations, losses_inf, 'b-', linewidth=2, label='Infinite Width')
    ax.set_xlabel('Training Iteration', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    ax.set_title('Training Curves: Finite vs Infinite Width', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Kernel Function Evolution
    ax = axes[1, 1]
    ax.set_facecolor(config.COLORS['panel'])
    visualization.style_axes(ax)
    
    # Show how kernel function changes with training
    x_test = np.linspace(-2, 2, 100)
    x_ref = np.array([0.0])  # Reference point
    
    # Initial kernel
    params_init = models.initialize_weights(width=100)
    K_init = models.compute_ntk_matrix(x_test, params_init)
    k_init = K_init[:, np.where(x_test == x_ref)[0][0]]
    
    # Trained kernel
    params_trained = models.train_step(params_init, x_train, y_train, lr=0.01)
    K_trained = models.compute_ntk_matrix(x_test, params_trained)
    k_trained = K_trained[:, np.where(x_test == x_ref)[0][0]]
    
    ax.plot(x_test, k_init, 'b-', linewidth=2, label='Initial Kernel')
    ax.plot(x_test, k_trained, 'r-', linewidth=2, label='Trained Kernel')
    ax.axvline(x_ref, color='k', linestyle='--', alpha=0.5, label='Reference Point')
    ax.set_xlabel('Input x', fontsize=10)
    ax.set_ylabel('Kernel Value K(x, x_ref)', fontsize=10)
    ax.set_title('Kernel Function Evolution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    figure_path = visualization.save_figure(fig, save_path)
    
    return figure_path


def main():
    """Main function to generate all NTK visualizations."""
    print("=" * 70)
    print("GENERATING NEURAL TANGENT KERNEL VISUALIZATIONS")
    print("=" * 70)
    print("Demonstrating: Neural Tangent Kernel Theory")
    print("=" * 70)
    
    # Set random seeds for reproducibility
    set_random_seeds()
    
    # Create NTK demonstration
    print("\n1. Creating NTK demonstration...")
    demo_path = create_ntk_demonstration('ntk_demonstration.png')
    print(f"Saved: {demo_path}")
    
    # Create frozen kernel comparison
    print("\n2. Creating frozen kernel comparison...")
    frozen_path = create_frozen_kernel_comparison('frozen_kernel_comparison.png')
    print(f"Saved: {frozen_path}")
    
    # Create training animation
    print("\n3. Creating training animation...")
    animation_path = create_ntk_training_animation('ntk_training_animation.gif')
    print(f"Saved: {animation_path}")
    
    # Create theory comparison
    print("\n4. Creating theory comparison...")
    theory_path = create_ntk_theory_comparison('ntk_theory_comparison.png')
    print(f"Saved: {theory_path}")
    
    print("\n" + "=" * 70)
    print("ALL VISUALIZATIONS COMPLETED!")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"- {demo_path} (NTK demonstration)")
    print(f"- {frozen_path} (Frozen kernel comparison)")
    print(f"- {animation_path} (Training animation)")
    print(f"- {theory_path} (Theory vs practice)")
    print("\nThese visualizations demonstrate key concepts of Neural Tangent Kernel theory,")
    print("including the infinite width limit, kernel freezing, and training dynamics.")


if __name__ == "__main__":
    main()