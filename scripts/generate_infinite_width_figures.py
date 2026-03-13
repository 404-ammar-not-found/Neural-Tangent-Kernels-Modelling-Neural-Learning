"""
Generate infinite width network visualization figures.
This script creates visualizations demonstrating how neural network loss landscapes
become smoother as network width increases.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src import config
from src import data
from src import models
from src import visualization
from src.utils import set_random_seeds, calculate_smoothness_metric


def generate_data(n_samples=100, noise_std=0.1):
    """Generate sin(x) data with noise for regression task."""
    x = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    y = np.sin(x.flatten()) + noise_std * np.random.randn(n_samples)
    return x.flatten(), y


def sample_loss_landscape(width, x_data, y_data, param_indices=(0, 1), 
                         range_val=0.3, n_points=40):
    """
    Sample the loss landscape by varying 2 parameters around initialization.
    
    Args:
        width: Network width
        x_data, y_data: Training data
        param_indices: Which two parameters to vary (flattened index)
        range_val: How far to vary parameters from initialization
        n_points: Number of points in each dimension
    
    Returns:
        grid_x, grid_y: Parameter value grids
        loss_grid: Loss values at each point
    """
    # Initialize network parameters
    params = models.initialize_weights(width=width, n_layers=3)
    
    # Store original parameters
    original_params = []
    for W, b in params:
        original_params.append((W.copy(), b.copy()))
    
    # Select two parameters to vary
    idx1, idx2 = param_indices
    
    # Create grid of parameter perturbations
    delta_range = np.linspace(-range_val, range_val, n_points)
    grid_delta1, grid_delta2 = np.meshgrid(delta_range, delta_range, indexing='ij')
    
    # Loss function
    def compute_loss(params):
        predictions = models.forward_batch(x_data, params)
        return 0.5 * np.mean((predictions - y_data) ** 2)
    
    # Store loss values
    loss_grid = np.zeros((n_points, n_points))
    
    # Sample loss at each point in the grid
    for i in range(n_points):
        for j in range(n_points):
            # Create perturbed parameters
            perturbed_params = []
            for k, (W, b) in enumerate(params):
                if k == 0:  # First layer weights
                    W_perturbed = W.copy()
                    if W.size > idx1:
                        W_perturbed.ravel()[idx1] = original_params[k][0].ravel()[idx1] + grid_delta1[i, j]
                    if W.size > idx2:
                        W_perturbed.ravel()[idx2] = original_params[k][0].ravel()[idx2] + grid_delta2[i, j]
                    perturbed_params.append((W_perturbed, b.copy()))
                else:
                    perturbed_params.append((W.copy(), b.copy()))
            
            # Compute loss
            loss_grid[i, j] = compute_loss(perturbed_params)
    
    return grid_delta1, grid_delta2, loss_grid


def create_landscape_comparison(widths=[5, 50, 500], save_path='landscape_comparison.png'):
    """
    Create side-by-side comparison of loss landscapes for different widths.
    
    Args:
        widths: List of network widths to compare
        save_path: Output file path
    
    Returns:
        Tuple of (figure_path, animation_path) if applicable
    """
    print(f"Generating landscapes for widths: {widths}")
    
    # Generate data
    x_data, y_data = generate_data(n_samples=50)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(widths), figsize=(6*len(widths), 5))
    if len(widths) == 1:
        axes = [axes]
    
    landscapes = []
    
    for i, width in enumerate(widths):
        print(f"  Processing width = {width}...")
        
        # Sample loss landscape
        grid_x, grid_y, loss_grid = sample_loss_landscape(
            width, x_data, y_data, 
            param_indices=(0, 1),  # First two parameters
            range_val=0.3,
            n_points=40
        )
        
        landscapes.append((grid_x, grid_y, loss_grid))
        
        # Plot
        visualization.create_contour_plot(
            grid_x, grid_y, loss_grid, 
            ax=axes[i],
            levels=30,
            cmap='viridis',
            title=f'Width = {width}',
            xlabel='Parameter 1 Perturbation',
            ylabel='Parameter 2 Perturbation'
        )
        
        # Calculate and display smoothness metric
        smoothness = calculate_smoothness_metric(loss_grid)
        axes[i].text(0.02, 0.98, f'Smoothness: {smoothness:.4f}', 
                    transform=axes[i].transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=10)
    
    # Add main title
    fig.suptitle('Neural Network Loss Landscape Smoothness vs Width\n' +
                 'As width increases, the landscape becomes more convex and predictable',
                 fontsize=14, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    figure_path = visualization.save_figure(fig, save_path)
    
    return figure_path


def plot_cross_section(widths=[5, 50, 500], save_path='cross_section.png'):
    """
    Plot 1D cross-section of loss landscape to show smoothness quantitatively.
    
    Args:
        widths: List of network widths to compare
        save_path: Output file path
    
    Returns:
        Path to saved figure
    """
    print("Generating cross-section comparison...")
    x_data, y_data = generate_data(n_samples=50)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['red', 'blue', 'green']
    
    for i, width in enumerate(widths):
        print(f"  Processing width = {width}...")
        
        # Sample loss landscape
        grid_x, grid_y, loss_grid = sample_loss_landscape(
            width, x_data, y_data,
            param_indices=(0, 1),
            range_val=0.3,
            n_points=50
        )
        
        # Take cross-section at y=0 (middle of grid)
        mid_idx = len(grid_y) // 2
        cross_section = loss_grid[:, mid_idx]
        delta_values = grid_x[:, mid_idx]
        
        # Normalize loss to start at 0 for comparison
        cross_section = cross_section - cross_section.min()
        
        ax.plot(delta_values, cross_section, color=colors[i], 
                linewidth=2, label=f'Width = {width}')
    
    # Add quadratic fit to show convexity
    delta_values = landscapes[0][0][:, len(landscapes[0][1])//2] if 'landscapes' in dir() else grid_x[:, mid_idx]
    quadratic = delta_values ** 2
    quadratic = quadratic / quadratic.max() * cross_section.max()
    ax.plot(delta_values, quadratic, 'k--', linewidth=2, label='Perfect Quadratic (Convex)')
    
    ax.set_xlabel('Parameter Perturbation', fontsize=12)
    ax.set_ylabel('Normalized Loss', fontsize=12)
    ax.set_title('Loss Landscape Cross-Section: Width vs Smoothness\n' +
                 'Wider networks → More quadratic (convex) → Easier to optimize',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    figure_path = visualization.save_figure(fig, save_path)
    
    return figure_path


def plot_smoothness_metric(widths=[5, 10, 20, 50, 100, 200, 500], save_path='smoothness_metric.png'):
    """
    Plot how smoothness metric decreases with width.
    
    Args:
        widths: List of network widths to test
        save_path: Output file path
    
    Returns:
        Path to saved figure
    """
    print("Calculating smoothness metrics...")
    x_data, y_data = generate_data(n_samples=50)
    
    smoothness_values = []
    
    for width in widths:
        print(f"  Processing width = {width}...")
        grid_x, grid_y, loss_grid = sample_loss_landscape(
            width, x_data, y_data,
            param_indices=(0, 1),
            range_val=0.3,
            n_points=40
        )
        smoothness = calculate_smoothness_metric(loss_grid)
        smoothness_values.append(smoothness)
        print(f"    Smoothness metric: {smoothness:.6f}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(widths, smoothness_values, 'bo-', linewidth=2, markersize=8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Network Width (log scale)', fontsize=12)
    ax.set_ylabel('Smoothness Metric (log scale)\n(Lower = Smoother Landscape)', fontsize=12)
    ax.set_title('Quantitative Evidence: Loss Landscape Smoothness vs Network Width\n' +
                 'As width → ∞, landscape becomes perfectly smooth (convex)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    
    # Add trend line
    z = np.polyfit(np.log(widths), np.log(smoothness_values), 1)
    trend_line = np.exp(z[1]) * np.array(widths) ** z[0]
    ax.plot(widths, trend_line, 'r--', linewidth=2, label=f'Trend (slope = {z[0]:.2f})')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    figure_path = visualization.save_figure(fig, save_path)
    
    return figure_path


def create_3d_visualization(width=100, save_path='3d_landscape.png'):
    """
    Create 3D visualization of loss landscape.
    
    Args:
        width: Network width to visualize
        save_path: Output file path
    
    Returns:
        Path to saved figure
    """
    print(f"Creating 3D landscape for width = {width}...")
    
    # Generate data
    x_data, y_data = generate_data(n_samples=50)
    
    # Sample loss landscape
    grid_x, grid_y, loss_grid = sample_loss_landscape(
        width, x_data, y_data,
        param_indices=(0, 1),
        range_val=0.3,
        n_points=30
    )
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create surface plot
    surface = ax.plot_surface(grid_x, grid_y, loss_grid, cmap='viridis', 
                            alpha=0.8, edgecolor='none')
    
    # Add labels and title
    ax.set_xlabel('Parameter 1', fontsize=10)
    ax.set_ylabel('Parameter 2', fontsize=10)
    ax.set_zlabel('Loss', fontsize=10)
    ax.set_title(f'3D Loss Landscape (Width = {width})', fontsize=12, fontweight='bold')
    
    # Add colorbar
    plt.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
    
    figure_path = visualization.save_figure(fig, save_path)
    
    return figure_path


def main():
    """Main function to generate all infinite width visualizations."""
    print("=" * 70)
    print("GENERATING INFINITE WIDTH NETWORK VISUALIZATIONS")
    print("=" * 70)
    print("Demonstrating: Infinite Width → Smooth Loss Landscape")
    print("=" * 70)
    
    # Set random seeds for reproducibility
    set_random_seeds()
    
    # Generate landscape comparison
    print("\n1. Creating landscape comparison...")
    landscape_path = create_landscape_comparison(
        widths=[5, 50, 500], 
        save_path='landscape_comparison.png'
    )
    print(f"Saved: {landscape_path}")
    
    # Generate cross-section plot
    print("\n2. Creating cross-section plot...")
    cross_section_path = plot_cross_section(
        widths=[5, 50, 500],
        save_path='cross_section.png'
    )
    print(f"Saved: {cross_section_path}")
    
    # Generate smoothness metric plot
    print("\n3. Creating smoothness metric plot...")
    smoothness_path = plot_smoothness_metric(
        widths=[5, 10, 20, 50, 100, 200, 500],
        save_path='smoothness_metric.png'
    )
    print(f"Saved: {smoothness_path}")
    
    # Generate 3D visualization
    print("\n4. Creating 3D landscape visualization...")
    landscape_3d_path = create_3d_visualization(
        width=100,
        save_path='3d_landscape.png'
    )
    print(f"Saved: {landscape_3d_path}")
    
    print("\n" + "=" * 70)
    print("ALL VISUALIZATIONS COMPLETED!")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"- {landscape_path} (Width comparison)")
    print(f"- {cross_section_path} (Cross-section analysis)")
    print(f"- {smoothness_path} (Smoothness metric)")
    print(f"- {landscape_3d_path} (3D visualization)")
    print("\nThese figures demonstrate how neural network loss landscapes")
    print("become smoother and more convex as network width increases,")
    print("making optimization easier in the infinite width limit.")


if __name__ == "__main__":
    main()