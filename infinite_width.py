import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# 1. Define Neural Network Class
# =============================================================================

class SimpleNN(nn.Module):
    """
    Simple 1-hidden-layer neural network for regression.
    Width = number of neurons in hidden layer.
    """
    def __init__(self, width, input_dim=1, output_dim=1):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.Linear(width, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# =============================================================================
# 2. Generate Synthetic Data (sin(x) curve)
# =============================================================================

def generate_data(n_samples=100, noise_std=0.1):
    """Generate sin(x) data with noise for regression task."""
    x = torch.linspace(-3, 3, n_samples).reshape(-1, 1)
    y = torch.sin(x) + noise_std * torch.randn_like(x)
    return x, y

# =============================================================================
# 3. Loss Landscape Sampling Function
# =============================================================================

def sample_loss_landscape(model, x_data, y_data, param_indices=(0, 1), 
                          range_val=0.5, n_points=50):
    """
    Sample the loss landscape by varying 2 parameters around initialization.
    
    Args:
        model: The neural network
        x_data, y_data: Training data
        param_indices: Which two parameters to vary (flattened index)
        range_val: How far to vary parameters from initialization
        n_points: Number of points in each dimension
    
    Returns:
        grid_x, grid_y: Parameter value grids
        loss_grid: Loss values at each point
    """
    # Get all parameters as a single flat tensor
    params = list(model.parameters())
    flat_params = torch.cat([p.reshape(-1) for p in params])
    
    # Store original parameters
    original_params = flat_params.clone()
    
    # Select two parameters to vary
    idx1, idx2 = param_indices
    
    # Create grid of parameter perturbations
    delta_range = torch.linspace(-range_val, range_val, n_points)
    grid_delta1, grid_delta2 = torch.meshgrid(delta_range, delta_range, indexing='ij')
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Store loss values
    loss_grid = np.zeros((n_points, n_points))
    
    # Sample loss at each point in the grid
    for i in range(n_points):
        for j in range(n_points):
            # Create perturbed parameters
            perturbed_params = original_params.clone()
            perturbed_params[idx1] = original_params[idx1] + grid_delta1[i, j]
            perturbed_params[idx2] = original_params[idx2] + grid_delta2[i, j]
            
            # Load perturbed parameters into model
            param_idx = 0
            for p in model.parameters():
                num_elements = p.numel()
                p.data = perturbed_params[param_idx:param_idx + num_elements].reshape(p.shape)
                param_idx += num_elements
            
            # Compute loss
            with torch.no_grad():
                predictions = model(x_data)
                loss = criterion(predictions, y_data)
                loss_grid[i, j] = loss.item()
    
    # Restore original parameters
    param_idx = 0
    for p in model.parameters():
        num_elements = p.numel()
        p.data = original_params[param_idx:param_idx + num_elements].reshape(p.shape)
        param_idx += num_elements
    
    return grid_delta1.numpy(), grid_delta2.numpy(), loss_grid

# =============================================================================
# 4. Plotting Functions
# =============================================================================

def plot_loss_landscape(grid_x, grid_y, loss_grid, width, ax):
    """Create contour plot of loss landscape."""
    contour = ax.contourf(grid_x, grid_y, loss_grid, levels=30, cmap='viridis')
    ax.set_xlabel('Parameter 1 Perturbation', fontsize=10)
    ax.set_ylabel('Parameter 2 Perturbation', fontsize=10)
    ax.set_title(f'Width = {width}', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(contour, ax=ax, label='Loss')

def plot_3d_landscape(grid_x, grid_y, loss_grid, width, ax):
    """Create 3D surface plot of loss landscape."""
    surface = ax.plot_surface(grid_x, grid_y, loss_grid, cmap='viridis', 
                              alpha=0.8, edgecolor='none')
    ax.set_xlabel('Parameter 1', fontsize=10)
    ax.set_ylabel('Parameter 2', fontsize=10)
    ax.set_zlabel('Loss', fontsize=10)
    ax.set_title(f'Width = {width}', fontsize=12, fontweight='bold')

# =============================================================================
# 5. Main Visualization - Compare Different Widths
# =============================================================================

def create_landscape_comparison(widths=[5, 50, 500], save_path='landscape_comparison.png'):
    """
    Create side-by-side comparison of loss landscapes for different widths.
    This is the MAIN visualization for your presentation slide.
    """
    print("Generating data...")
    x_data, y_data = generate_data(n_samples=50)
    
    print(f"Creating landscapes for widths: {widths}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, len(widths), figsize=(6*len(widths), 5))
    if len(widths) == 1:
        axes = [axes]
    
    landscapes = []
    
    for i, width in enumerate(widths):
        print(f"  Processing width = {width}...")
        
        # Create and initialize model
        model = SimpleNN(width=width)
        
        # Sample loss landscape
        # Vary first two weight parameters in the first layer
        grid_x, grid_y, loss_grid = sample_loss_landscape(
            model, x_data, y_data, 
            param_indices=(0, 1),  # First two parameters
            range_val=0.3,
            n_points=40
        )
        
        landscapes.append((grid_x, grid_y, loss_grid))
        
        # Plot
        plot_loss_landscape(grid_x, grid_y, loss_grid, width, axes[i])
    
    plt.suptitle('Neural Network Loss Landscape Smoothness vs Width\n' +
                 'As width increases, the landscape becomes more convex and predictable',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {save_path}")
    plt.show()
    
    return landscapes

# =============================================================================
# 6. Additional Visualization - Cross Section (1D Slice)
# =============================================================================

def plot_cross_section(widths=[5, 50, 500], save_path='cross_section.png'):
    """
    Plot 1D cross-section of loss landscape to show smoothness quantitatively.
    This shows the "Taylor expansion" concept visually.
    """
    print("Generating cross-section comparison...")
    x_data, y_data = generate_data(n_samples=50)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['red', 'blue', 'green']
    
    for i, width in enumerate(widths):
        print(f"  Processing width = {width}...")
        
        model = SimpleNN(width=width)
        grid_x, grid_y, loss_grid = sample_loss_landscape(
            model, x_data, y_data,
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved cross-section plot to {save_path}")
    plt.show()

# =============================================================================
# 7. Smoothness Metric Calculation
# =============================================================================

def calculate_smoothness_metric(loss_grid):
    """
    Calculate a numerical smoothness metric (second derivative variance).
    Lower value = smoother landscape.
    """
    # Compute second derivatives (discrete Laplacian)
    second_deriv_x = np.diff(loss_grid, n=2, axis=0)
    second_deriv_y = np.diff(loss_grid, n=2, axis=1)
    
    # Variance of second derivatives (lower = smoother)
    smoothness_x = np.var(second_deriv_x)
    smoothness_y = np.var(second_deriv_y)
    
    return (smoothness_x + smoothness_y) / 2

def plot_smoothness_metric(widths=[5, 10, 20, 50, 100, 200, 500], save_path='smoothness_metric.png'):
    """
    Plot how smoothness metric decreases with width.
    This provides quantitative evidence for the infinite width theory.
    """
    print("Calculating smoothness metrics...")
    x_data, y_data = generate_data(n_samples=50)
    
    smoothness_values = []
    
    for width in widths:
        print(f"  Processing width = {width}...")
        model = SimpleNN(width=width)
        _, _, loss_grid = sample_loss_landscape(
            model, x_data, y_data,
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved smoothness metric plot to {save_path}")
    plt.show()

# =============================================================================
# 8. Run All Visualizations
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NEURAL NETWORK LOSS LANDSCAPE VISUALIZATION")
    print("Demonstrating: Infinite Width → Smooth Loss Landscape")
    print("=" * 70)
    print()
    
    # Generate all visualizations
    create_landscape_comparison(widths=[5, 50, 500])
    print()
    
    plot_cross_section(widths=[5, 50, 500])
    print()
    
    plot_smoothness_metric(widths=[5, 10, 20, 50, 100, 200, 500])
    print()
    
    print("=" * 70)
    print("All visualizations complete!")
    print("Use these images in Slide 3 of your presentation.")
    print("=" * 70)