"""
Generate linear approximation visualization figures.
This script creates visualizations demonstrating the concept of linear approximation
and its relevance to neural network optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import matplotlib.patches as patches

from src import config
from src import visualization
from src.utils import set_random_seeds


def f(x):
    """Example function: f(x) = x^2"""
    return x**2


def f_prime(x):
    """Derivative of f(x) = x^2"""
    return 2*x


def create_interactive_plot(save_path='linear_approximation_interactive.png'):
    """
    Create interactive plot showing linear approximation.
    
    Args:
        save_path: Output file path for static version
    
    Returns:
        Path to saved figure
    """
    # Generate x values
    x = np.linspace(-2, 2, 1000)
    y = f(x)
    
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    plt.subplots_adjust(bottom=0.25, hspace=0.3)
    
    # Style axes
    for ax in [ax1, ax2]:
        ax.set_facecolor(config.COLORS['panel'])
        visualization.style_axes(ax)
    
    # Plot the function
    ax1.plot(x, y, color=config.COLORS['primary'], linewidth=2, label='f(x) = x²')
    
    # Initialize plot elements
    point_x, = ax1.plot([], [], 'ko', markersize=8, label='x')
    point_xh, = ax1.plot([], [], 'ro', markersize=8, label='x+h')
    tangent_line, = ax1.plot([], [], color=config.COLORS['success'], 
                           linewidth=2, linestyle='--', label='Tangent line')
    approx_line, = ax1.plot([], [], color=config.COLORS['danger'], 
                          linewidth=2, linestyle='--', label='Linear approximation')
    actual_point, = ax1.plot([], [], 'bo', markersize=6)
    approx_point, = ax1.plot([], [], 'ro', markersize=6)
    
    # Error plot
    h_values = []
    errors = []
    error_line, = ax2.plot([], [], color=config.COLORS['info'], linewidth=2)
    ax2.set_xlabel('h', fontsize=12, color=config.COLORS['white'])
    ax2.set_ylabel('Error |f(x+h) - [f(x) + h⋅f\'(x)]|', fontsize=12, color=config.COLORS['white'])
    ax2.set_title('Approximation Error vs h', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    # Add slider for x position
    ax_x = plt.axes([0.2, 0.1, 0.6, 0.03])
    x_slider = Slider(ax_x, 'x', -1.5, 1.5, valinit=1.0)
    
    # Add slider for h value
    ax_h = plt.axes([0.2, 0.05, 0.6, 0.03])
    h_slider = Slider(ax_h, 'h', 0.01, 1.0, valinit=0.5)
    
    def update(val):
        x_val = x_slider.val
        h_val = h_slider.val
        
        # Calculate function values
        y_x = f(x_val)
        y_xh = f(x_val + h_val)
        
        # Calculate tangent line
        tangent_x = np.linspace(x_val - 1, x_val + 1, 100)
        tangent_y = y_x + f_prime(x_val) * (tangent_x - x_val)
        
        # Calculate linear approximation
        approx_x = np.linspace(x_val, x_val + h_val, 100)
        approx_y = y_x + f_prime(x_val) * (approx_x - x_val)
        
        # Update plots
        point_x.set_data([x_val], [y_x])
        point_xh.set_data([x_val + h_val], [y_xh])
        tangent_line.set_data(tangent_x, tangent_y)
        approx_line.set_data(approx_x, approx_y)
        actual_point.set_data([x_val + h_val], [y_xh])
        approx_point.set_data([x_val + h_val], [y_x + f_prime(x_val) * h_val])
        
        # Calculate and update error
        error = abs(y_xh - (y_x + f_prime(x_val) * h_val))
        h_values.append(h_val)
        errors.append(error)
        
        # Keep only recent values for error plot
        if len(h_values) > 100:
            h_values.pop(0)
            errors.pop(0)
        
        error_line.set_data(h_values, errors)
        
        # Update error plot limits
        if h_values and errors:
            ax2.set_xlim(min(h_values) * 0.9, max(h_values) * 1.1)
            ax2.set_ylim(min(errors) * 0.9, max(errors) * 1.1)
        
        # Update title with current values
        ax1.set_title(f'Linear Approximation at x={x_val:.2f}, h={h_val:.3f}\n' + 
                     f'Error: {error:.6f}', fontsize=14, fontweight='bold')
        
        fig.canvas.draw_idle()
    
    # Connect sliders to update function
    x_slider.on_changed(update)
    h_slider.on_changed(update)
    
    # Initial update
    update(None)
    
    # Add text annotations
    ax1.text(0.02, 0.95, 'Blue: f(x) = x²', transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.text(0.02, 0.85, 'Green: Tangent line f(x) + f\'(x)(x₀-x)', transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.text(0.02, 0.75, 'Red: Linear approximation f(x₀) + f\'(x₀)h', transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax1.set_xlabel('x', fontsize=12, color=config.COLORS['white'])
    ax1.set_ylabel('f(x)', fontsize=12, color=config.COLORS['white'])
    ax1.legend(loc='lower right')
    
    # Save figure
    figure_path = visualization.save_figure(fig, save_path)
    
    return figure_path


def create_animation(save_path='linear_approximation_animation.gif'):
    """
    Create animation showing decreasing h values.
    
    Args:
        save_path: Output file path
    
    Returns:
        Path to saved animation
    """
    # Generate x values
    x = np.linspace(-2, 2, 1000)
    y = f(x)
    
    # Create animation figure
    fig_anim, ax_anim = plt.subplots(figsize=(10, 6))
    ax_anim.set_facecolor(config.COLORS['panel'])
    
    # Plot the function
    ax_anim.plot(x, y, color=config.COLORS['primary'], linewidth=2, label='f(x) = x²')
    
    # Initialize animation elements
    anim_point_x, = ax_anim.plot([], [], 'ko', markersize=8)
    anim_point_xh, = ax_anim.plot([], [], 'ro', markersize=8)
    anim_tangent, = ax_anim.plot([], [], color=config.COLORS['success'], 
                               linewidth=2, linestyle='--')
    anim_approx, = ax_anim.plot([], [], color=config.COLORS['danger'], 
                              linewidth=2, linestyle='--')
    anim_actual, = ax_anim.plot([], [], 'bo', markersize=6)
    anim_approx_point, = ax_anim.plot([], [], 'ro', markersize=6)
    
    ax_anim.set_xlabel('x', fontsize=12, color=config.COLORS['white'])
    ax_anim.set_ylabel('f(x)', fontsize=12, color=config.COLORS['white'])
    ax_anim.set_title('Linear Approximation Animation: Decreasing h', fontsize=14, fontweight='bold')
    ax_anim.grid(True, alpha=0.3)
    ax_anim.legend()
    
    visualization.style_axes(ax_anim)
    
    def animate(frame):
        # Decrease h exponentially
        h = 0.5 * (0.5 ** (frame / 20))
        x_val = 1.0  # Fixed x value for animation
        
        # Calculate values
        y_x = f(x_val)
        y_xh = f(x_val + h)
        
        # Tangent line
        tangent_x = np.linspace(x_val - 0.5, x_val + 0.5, 100)
        tangent_y = y_x + f_prime(x_val) * (tangent_x - x_val)
        
        # Linear approximation
        approx_x = np.linspace(x_val, x_val + h, 100)
        approx_y = y_x + f_prime(x_val) * (approx_x - x_val)
        
        # Update animation elements
        anim_point_x.set_data([x_val], [y_x])
        anim_point_xh.set_data([x_val + h], [y_xh])
        anim_tangent.set_data(tangent_x, tangent_y)
        anim_approx.set_data(approx_x, approx_y)
        anim_actual.set_data([x_val + h], [y_xh])
        anim_approx_point.set_data([x_val + h], [y_x + f_prime(x_val) * h])
        
        # Update title with current h and error
        error = abs(y_xh - (y_x + f_prime(x_val) * h))
        ax_anim.set_title(f'Linear Approximation Animation: h = {h:.6f}, Error = {error:.8f}', 
                         fontsize=14, fontweight='bold')
        
        return anim_point_x, anim_point_xh, anim_tangent, anim_approx, anim_actual, anim_approx_point
    
    # Create animation
    anim = FuncAnimation(fig_anim, animate, frames=100, interval=100, blit=True, repeat=True)
    
    # Save animation
    animation_path = visualization.save_animation(anim, save_path)
    
    return animation_path


def create_error_analysis_plot(save_path='linear_approximation_error_analysis.png'):
    """
    Create plot showing error analysis for different h values.
    
    Args:
        save_path: Output file path
    
    Returns:
        Path to saved figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for ax in [ax1, ax2]:
        ax.set_facecolor(config.COLORS['panel'])
        visualization.style_axes(ax)
    
    # Test different x values
    x_values = [0.5, 1.0, 1.5]
    h_range = np.logspace(-4, 0, 100)
    
    colors = [config.COLORS['primary'], config.COLORS['success'], config.COLORS['danger']]
    
    for i, x_val in enumerate(x_values):
        errors = []
        for h in h_range:
            y_true = f(x_val + h)
            y_approx = f(x_val) + f_prime(x_val) * h
            errors.append(abs(y_true - y_approx))
        
        ax1.loglog(h_range, errors, color=colors[i], linewidth=2, 
                  label=f'x = {x_val}')
    
    # Theoretical error bound (O(h²))
    theoretical_error = h_range**2
    ax1.loglog(h_range, theoretical_error, 'k--', linewidth=2, 
              label='Theoretical O(h²)')
    
    ax1.set_xlabel('h (log scale)', fontsize=12, color=config.COLORS['white'])
    ax1.set_ylabel('Absolute Error (log scale)', fontsize=12, color=config.COLORS['white'])
    ax1.set_title('Linear Approximation Error vs h', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Show convergence for different functions
    x_val = 1.0
    h_range = np.logspace(-4, 0, 100)
    
    # Test different functions
    functions = [
        (lambda x: x**2, lambda x: 2*x, 'f(x) = x²', config.COLORS['primary']),
        (lambda x: x**3, lambda x: 3*x**2, 'f(x) = x³', config.COLORS['success']),
        (lambda x: np.exp(x), lambda x: np.exp(x), 'f(x) = eˣ', config.COLORS['danger']),
        (lambda x: np.sin(x), lambda x: np.cos(x), 'f(x) = sin(x)', config.COLORS['info'])
    ]
    
    for func, func_prime, label, color in functions:
        errors = []
        for h in h_range:
            y_true = func(x_val + h)
            y_approx = func(x_val) + func_prime(x_val) * h
            errors.append(abs(y_true - y_approx))
        
        ax2.loglog(h_range, errors, color=color, linewidth=2, label=label)
    
    ax2.set_xlabel('h (log scale)', fontsize=12, color=config.COLORS['white'])
    ax2.set_ylabel('Absolute Error (log scale)', fontsize=12, color=config.COLORS['white'])
    ax2.set_title('Linear Approximation for Different Functions', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    figure_path = visualization.save_figure(fig, save_path)
    
    return figure_path


def create_comparison_plot(save_path='linear_approximation_comparison.png'):
    """
    Create comparison plot showing linear vs higher-order approximations.
    
    Args:
        save_path: Output file path
    
    Returns:
        Path to saved figure
    """
    x_val = 1.0
    h_range = np.linspace(-0.5, 0.5, 100)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor(config.COLORS['panel'])
    visualization.style_axes(ax)
    
    # Plot true function
    x_full = np.linspace(x_val - 1, x_val + 1, 200)
    y_full = f(x_full)
    ax.plot(x_full, y_full, color=config.COLORS['primary'], linewidth=3, 
           label='f(x) = x² (True)')
    
    # Linear approximation
    y_linear = f(x_val) + f_prime(x_val) * (x_full - x_val)
    ax.plot(x_full, y_linear, color=config.COLORS['success'], linewidth=2, 
           linestyle='--', label='Linear approximation')
    
    # Quadratic approximation
    y_quad = f(x_val) + f_prime(x_val) * (x_full - x_val) + 0.5 * 2 * (x_full - x_val)**2
    ax.plot(x_full, y_quad, color=config.COLORS['danger'], linewidth=2, 
           linestyle=':', label='Quadratic approximation')
    
    # Mark evaluation point
    ax.plot(x_val, f(x_val), 'ko', markersize=10, label=f'Evaluation point ({x_val}, {f(x_val)})')
    
    ax.set_xlabel('x', fontsize=12, color=config.COLORS['white'])
    ax.set_ylabel('f(x)', fontsize=12, color=config.COLORS['white'])
    ax.set_title('Linear vs Higher-Order Approximations', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    figure_path = visualization.save_figure(fig, save_path)
    
    return figure_path


def main():
    """Main function to generate all linear approximation visualizations."""
    print("=" * 70)
    print("GENERATING LINEAR APPROXIMATION VISUALIZATIONS")
    print("=" * 70)
    print("Demonstrating: Linear Approximation and Taylor Series")
    print("=" * 70)
    
    # Set random seeds for reproducibility
    set_random_seeds()
    
    # Create interactive plot
    print("\n1. Creating interactive plot...")
    interactive_path = create_interactive_plot('linear_approximation_interactive.png')
    print(f"Saved: {interactive_path}")
    
    # Create animation
    print("\n2. Creating animation...")
    animation_path = create_animation('linear_approximation_animation.gif')
    print(f"Saved: {animation_path}")
    
    # Create error analysis plot
    print("\n3. Creating error analysis plot...")
    error_path = create_error_analysis_plot('linear_approximation_error_analysis.png')
    print(f"Saved: {error_path}")
    
    # Create comparison plot
    print("\n4. Creating comparison plot...")
    comparison_path = create_comparison_plot('linear_approximation_comparison.png')
    print(f"Saved: {comparison_path}")
    
    print("\n" + "=" * 70)
    print("ALL VISUALIZATIONS COMPLETED!")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"- {interactive_path} (Interactive visualization)")
    print(f"- {animation_path} (Animation showing h → 0)")
    print(f"- {error_path} (Error analysis)")
    print(f"- {comparison_path} (Linear vs higher-order approximations)")
    print("\nThese visualizations demonstrate the concept of linear approximation")
    print("and its importance for understanding neural network optimization.")
    print("The first-order Taylor expansion forms the basis of gradient-based methods.")


if __name__ == "__main__":
    main()