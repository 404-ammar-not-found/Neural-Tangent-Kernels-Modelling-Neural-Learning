import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import matplotlib.patches as patches

# Define the function and its derivative
def f(x):
    """Example function: f(x) = x^2"""
    return x**2

def f_prime(x):
    """Derivative of f(x) = x^2"""
    return 2*x

# Generate x values
x = np.linspace(-2, 2, 1000)
y = f(x)

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
plt.subplots_adjust(bottom=0.25, hspace=0.3)

# Plot the function
ax1.plot(x, y, 'b-', linewidth=2, label=f'f(x) = x²')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.set_title('Linear Approximation: f(x+h) ≈ f(x) + h⋅f\'(x)')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Initialize plot elements
point_x, = ax1.plot([], [], 'ko', markersize=8, label='x')
point_xh, = ax1.plot([], [], 'ro', markersize=8, label='x+h')
tangent_line, = ax1.plot([], [], 'g--', linewidth=2, label='Tangent line')
approx_line, = ax1.plot([], [], 'r--', linewidth=2, label='Linear approximation')
actual_point, = ax1.plot([], [], 'bo', markersize=6)
approx_point, = ax1.plot([], [], 'ro', markersize=6)

# Error plot
h_values = []
errors = []
error_line, = ax2.plot([], [], 'purple', linewidth=2)
ax2.set_xlabel('h')
ax2.set_ylabel('Error |f(x+h) - [f(x) + h⋅f\'(x)]|')
ax2.set_title('Approximation Error vs h')
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
                  f'Error: {error:.6f}')
    
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

plt.show()

# Create animation showing decreasing h values
fig_anim, ax_anim = plt.subplots(figsize=(10, 6))
ax_anim.plot(x, y, 'b-', linewidth=2, label=f'f(x) = x²')

# Animation elements
anim_point_x, = ax_anim.plot([], [], 'ko', markersize=8)
anim_point_xh, = ax_anim.plot([], [], 'ro', markersize=8)
anim_tangent, = ax_anim.plot([], [], 'g--', linewidth=2)
anim_approx, = ax_anim.plot([], [], 'r--', linewidth=2)
anim_actual, = ax_anim.plot([], [], 'bo', markersize=6)
anim_approx_point, = ax_anim.plot([], [], 'ro', markersize=6)

ax_anim.set_xlabel('x')
ax_anim.set_ylabel('f(x)')
ax_anim.set_title('Linear Approximation Animation: Decreasing h')
ax_anim.grid(True, alpha=0.3)
ax_anim.legend()

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
    ax_anim.set_title(f'Linear Approximation Animation: h = {h:.6f}, Error = {error:.8f}')
    
    return anim_point_x, anim_point_xh, anim_tangent, anim_approx, anim_actual, anim_approx_point

# Create animation
anim = FuncAnimation(fig_anim, animate, frames=100, interval=100, blit=True, repeat=True)

# Save animation
anim.save('linear_approximation_animation.gif', writer='pillow', fps=10)
print("Saved linear_approximation_animation.gif")

plt.show()