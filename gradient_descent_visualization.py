import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import io

# Generate synthetic data
np.random.seed(42)
X = np.linspace(-5, 5, 100)
y = 2 * X + 1 + np.random.normal(0, 1, 100)

# Initialize parameters
m, b = 0, 0
learning_rate = 0.01
iterations = 100

# Store history for visualization
m_history = []
b_history = []
error_history = []

# Batch gradient descent
for i in range(iterations):
    y_pred = m * X + b
    error = y - y_pred
    m_gradient = -2 * np.sum(X * error) / len(X)
    b_gradient = -2 * np.sum(error) / len(X)
    
    m -= learning_rate * m_gradient
    b -= learning_rate * b_gradient
    
    m_history.append(m)
    b_history.append(b)
    error_history.append(np.mean(error**2))

# Create line fitting animation
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.scatter(X, y, color='blue', label='Data')
line, = ax1.plot([], [], color='red', label='Fitted line')
ax1.set_xlim(-6, 6)
ax1.set_ylim(-10, 15)
ax1.set_xlabel('X')
ax1.set_ylabel('y')
ax1.set_title('Batch Gradient Descent - Line Fitting')
ax1.legend()

def update_line(i):
    line.set_data(X, m_history[i] * X + b_history[i])
    ax1.set_title(f'Iteration {i+1}/{iterations}')
    return line,

anim_line = FuncAnimation(fig1, update_line, frames=range(iterations), interval=100, repeat=False)
fig1.savefig('line_fitting.png')

# Create error landscape animation
m_range = np.linspace(m-3, m+3, 50)
b_range = np.linspace(b-3, b+3, 50)
M, B = np.meshgrid(m_range, b_range)
Z = np.zeros_like(M)

for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        y_pred = M[i,j] * X + B[i,j]
        Z[i,j] = np.mean((y - y_pred)**2)

fig2, ax2 = plt.subplots(figsize=(8, 6))
contour = ax2.contourf(M, B, Z, levels=20, cmap='viridis')
ax2.set_xlabel('Slope (m)')
ax2.set_ylabel('Intercept (b)')
ax2.set_title('Error Landscape')
scatter, = ax2.plot([], [], 'ro', markersize=5)

def update_error(i):
    scatter.set_data([m_history[i]], [b_history[i]])
    ax2.set_title(f'Iteration {i+1}/{iterations}')
    return scatter,

anim_error = FuncAnimation(fig2, update_error, frames=range(iterations), interval=100, repeat=False)
fig2.savefig('error_landscape.png')

# Save animations as GIFs
def save_animation_as_gif(anim, filename):
    # Use the correct method to save animation as GIF
    anim.save(filename, writer='pillow')
    print(f"Saved {filename}")

save_animation_as_gif(anim_line, 'line_fitting.gif')
save_animation_as_gif(anim_error, 'error_landscape.gif')

print("Animations saved successfully!")
print("line_fitting.gif - Shows the evolution of the fitted line")
print("error_landscape.gif - Shows the path through the error landscape")