"""
Generate gradient descent visualization figures.
This script creates visualizations for the gradient descent presentation slides.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import io

from src import config
from src import data
from src import visualization
from src.utils import set_random_seeds


def generate_synthetic_data(n_samples=100, noise_std=1.0, x_range=(-5, 5)):
    """Generate linear regression data for gradient descent visualization."""
    x = np.linspace(x_range[0], x_range[1], n_samples)
    y = 2 * x + 1 + noise_std * np.random.randn(n_samples)
    return x, y


def batch_gradient_descent(x, y, learning_rate=0.01, iterations=100):
    """
    Perform batch gradient descent for linear regression.
    
    Args:
        x: Input data
        y: Target data
        learning_rate: Learning rate
        iterations: Number of iterations
    
    Returns:
        Tuple of (m_history, b_history, error_history)
    """
    # Initialize parameters
    m, b = 0, 0
    
    # Store history for visualization
    m_history = []
    b_history = []
    error_history = []
    
    # Batch gradient descent
    for i in range(iterations):
        y_pred = m * x + b
        error = y - y_pred
        
        # Compute gradients
        m_gradient = -2 * np.sum(x * error) / len(x)
        b_gradient = -2 * np.sum(error) / len(x)
        
        # Update parameters
        m -= learning_rate * m_gradient
        b -= learning_rate * b_gradient
        
        # Store history
        m_history.append(m)
        b_history.append(b)
        error_history.append(np.mean(error**2))
    
    return m_history, b_history, error_history


def create_error_landscape(x, y, m_range=(-2, 4), b_range=(-2, 4), n_points=50):
    """
    Create error landscape for visualization.
    
    Args:
        x, y: Data points
        m_range: Range for slope parameter
        b_range: Range for intercept parameter
        n_points: Number of points in each dimension
    
    Returns:
        Tuple of (m_grid, b_grid, error_grid)
    """
    m_values = np.linspace(m_range[0], m_range[1], n_points)
    b_values = np.linspace(b_range[0], b_range[1], n_points)
    
    m_grid, b_grid = np.meshgrid(m_values, b_values)
    error_grid = np.zeros_like(m_grid)
    
    # Compute error for each parameter combination
    for i in range(n_points):
        for j in range(n_points):
            y_pred = m_grid[i, j] * x + b_grid[i, j]
            error_grid[i, j] = np.mean((y - y_pred)**2)
    
    return m_grid, b_grid, error_grid


def create_line_fitting_animation(x, y, m_history, b_history, save_path='line_fitting.gif'):
    """Create animation of line fitting process."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor(config.COLORS['panel'])
    
    # Plot data points
    ax.scatter(x, y, color=config.COLORS['primary'], s=50, alpha=0.7, label='Data')
    
    # Initialize line
    line, = ax.plot([], [], color=config.COLORS['danger'], linewidth=2, label='Fitted line')
    
    # Set axis limits
    ax.set_xlim(config.GRADIENT_DESCENT_SETTINGS['line_range'][0] - 1, 
                config.GRADIENT_DESCENT_SETTINGS['line_range'][1] + 1)
    ax.set_ylim(config.GRADIENT_DESCENT_SETTINGS['error_range'][0] - 1, 
                config.GRADIENT_DESCENT_SETTINGS['error_range'][1] + 1)
    
    ax.set_xlabel('x', fontsize=12, color=config.COLORS['white'])
    ax.set_ylabel('y', fontsize=12, color=config.COLORS['white'])
    ax.set_title('Batch Gradient Descent - Line Fitting', fontsize=14, fontweight='bold', color=config.COLORS['white'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Style axes
    visualization.style_axes(ax)
    
    def update(frame):
        # Update line
        y_pred = m_history[frame] * x + b_history[frame]
        line.set_data(x, y_pred)
        
        # Update title
        ax.set_title(f'Batch Gradient Descent - Line Fitting (Iteration {frame+1}/{len(m_history)})', 
                     fontsize=14, fontweight='bold', color=config.COLORS['white'])
        
        return line,
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(m_history), interval=100, blit=True, repeat=True)
    
    # Save animation
    visualization.save_animation(anim, save_path)
    
    # Also save static plot at final iteration
    plt.close()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor(config.COLORS['panel'])
    
    # Plot data points
    ax.scatter(x, y, color=config.COLORS['primary'], s=50, alpha=0.7, label='Data')
    
    # Plot final line
    final_y_pred = m_history[-1] * x + b_history[-1]
    ax.plot(x, final_y_pred, color=config.COLORS['danger'], linewidth=2, label='Final fitted line')
    
    ax.set_xlim(config.GRADIENT_DESCENT_SETTINGS['line_range'][0] - 1, 
                config.GRADIENT_DESCENT_SETTINGS['line_range'][1] + 1)
    ax.set_ylim(config.GRADIENT_DESCENT_SETTINGS['error_range'][0] - 1, 
                config.GRADIENT_DESCENT_SETTINGS['error_range'][1] + 1)
    
    ax.set_xlabel('x', fontsize=12, color=config.COLORS['white'])
    ax.set_ylabel('y', fontsize=12, color=config.COLORS['white'])
    ax.set_title(f'Final Result: y = {m_history[-1]:.3f}x + {b_history[-1]:.3f}', 
                 fontsize=14, fontweight='bold', color=config.COLORS['white'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    visualization.style_axes(ax)
    static_path = save_path.replace('.gif', '_final.png')
    visualization.save_figure(fig, static_path)
    
    return save_path, static_path


def create_error_landscape_animation(x, y, m_history, b_history, error_history, save_path='error_landscape.gif'):
    """Create animation of error landscape with optimization path."""
    # Create error landscape
    m_grid, b_grid, error_grid = create_error_landscape(x, y)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor(config.COLORS['panel'])
    
    # Create contour plot
    contour = ax.contourf(m_grid, b_grid, error_grid, levels=30, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='MSE Loss')
    
    # Plot optimization path
    path_line, = ax.plot([], [], 'r-', linewidth=2, label='Optimization Path')
    current_point, = ax.plot([], [], 'ro', markersize=8, label='Current Position')
    
    ax.set_xlabel('Slope (m)', fontsize=12, color=config.COLORS['white'])
    ax.set_ylabel('Intercept (b)', fontsize=12, color=config.COLORS['white'])
    ax.set_title('Error Landscape with Gradient Descent Path', fontsize=14, fontweight='bold', color=config.COLORS['white'])
    ax.legend()
    
    # Style axes
    visualization.style_axes(ax)
    
    def update(frame):
        # Update path
        path_line.set_data(m_history[:frame+1], b_history[:frame+1])
        current_point.set_data([m_history[frame]], [b_history[frame]])
        
        # Update title
        ax.set_title(f'Error Landscape - Iteration {frame+1}/{len(m_history)}\n'
                    f'Loss: {error_history[frame]:.4f}', 
                    fontsize=14, fontweight='bold', color=config.COLORS['white'])
        
        return path_line, current_point,
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(m_history), interval=100, blit=True, repeat=True)
    
    # Save animation
    visualization.save_animation(anim, save_path)
    
    # Also save static plot
    plt.close()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor(config.COLORS['panel'])
    
    # Create contour plot
    contour = ax.contourf(m_grid, b_grid, error_grid, levels=30, cmap='viridis')
    plt.colorbar(contour, ax=ax, label='MSE Loss')
    
    # Plot full optimization path
    ax.plot(m_history, b_history, 'r-', linewidth=2, label='Full Optimization Path')
    ax.plot(m_history[-1], b_history[-1], 'ro', markersize=10, label='Final Position')
    
    ax.set_xlabel('Slope (m)', fontsize=12, color=config.COLORS['white'])
    ax.set_ylabel('Intercept (b)', fontsize=12, color=config.COLORS['white'])
    ax.set_title('Complete Gradient Descent Path', fontsize=14, fontweight='bold', color=config.COLORS['white'])
    ax.legend()
    
    visualization.style_axes(ax)
    static_path = save_path.replace('.gif', '_final.png')
    visualization.save_figure(fig, static_path)
    
    return save_path, static_path


def create_loss_convergence_plot(error_history, save_path='loss_convergence.png'):
    """Create plot showing loss convergence over iterations."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor(config.COLORS['panel'])
    
    iterations = np.arange(len(error_history))
    ax.plot(iterations, error_history, color=config.COLORS['primary'], linewidth=2)
    
    # Add labels and title
    ax.set_xlabel('Iteration', fontsize=12, color=config.COLORS['white'])
    ax.set_ylabel('Mean Squared Error', fontsize=12, color=config.COLORS['white'])
    ax.set_title('Loss Convergence During Gradient Descent', fontsize=14, fontweight='bold', color=config.COLORS['white'])
    
    # Add grid and style
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    visualization.style_axes(ax)
    
    # Save figure
    visualization.save_figure(fig, save_path)
    return save_path


def main():
    """Main function to generate all gradient descent visualizations."""
    print("=" * 70)
    print("GENERATING GRADIENT DESCENT VISUALIZATIONS")
    print("=" * 70)
    
    # Set random seeds for reproducibility
    set_random_seeds()
    
    # Generate synthetic data
    print("Generating synthetic data...")
    x, y = generate_synthetic_data(
        n_samples=config.GRADIENT_DESCENT_SETTINGS['n_samples'],
        noise_std=1.0,
        x_range=(-5, 5)
    )
    
    # Perform gradient descent
    print("Running gradient descent...")
    m_history, b_history, error_history = batch_gradient_descent(
        x, y,
        learning_rate=config.GRADIENT_DESCENT_SETTINGS['learning_rate'],
        iterations=config.GRADIENT_DESCENT_SETTINGS['iterations']
    )
    
    print(f"Initial parameters: m={m_history[0]:.3f}, b={b_history[0]:.3f}")
    print(f"Final parameters: m={m_history[-1]:.3f}, b={b_history[-1]:.3f}")
    print(f"Final loss: {error_history[-1]:.6f}")
    
    # Create line fitting animation
    print("\nCreating line fitting animation...")
    line_gif, line_static = create_line_fitting_animation(
        x, y, m_history, b_history, 'line_fitting.gif'
    )
    print(f"Saved: {line_gif}, {line_static}")
    
    # Create error landscape animation
    print("\nCreating error landscape animation...")
    landscape_gif, landscape_static = create_error_landscape_animation(
        x, y, m_history, b_history, error_history, 'error_landscape.gif'
    )
    print(f"Saved: {landscape_gif}, {landscape_static}")
    
    # Create loss convergence plot
    print("\nCreating loss convergence plot...")
    loss_plot = create_loss_convergence_plot(error_history, 'loss_convergence.png')
    print(f"Saved: {loss_plot}")
    
    print("\n" + "=" * 70)
    print("ALL VISUALIZATIONS COMPLETED!")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"- {line_gif} (Animation)")
    print(f"- {line_static} (Final result)")
    print(f"- {landscape_gif} (Animation)")
    print(f"- {landscape_static} (Final result)")
    print(f"- {loss_plot} (Loss convergence)")
    print("\nThese figures are ready for your presentation slides.")


if __name__ == "__main__":
    main()