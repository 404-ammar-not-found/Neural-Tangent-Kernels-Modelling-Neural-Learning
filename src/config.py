"""
Configuration settings for Neural Tangent Kernels visualizations.
"""

import os
import yaml
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Paths
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Random seeds
RANDOM_SEED = 42
TORCH_SEED = 42
NP_SEED = 42

# Figure settings
FIGURE_SETTINGS = {
    'dpi': 300,
    'style': "seaborn",
    'size': (10, 6),
    'animation_fps': 15
}
FIGURE_DPI = 300
FIGURE_STYLE = "seaborn"
FIGURE_SIZE = (10, 6)
ANIMATION_FPS = 15

# Colors
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ffbb78',
    'info': '#17becf',
    'bg': '#ffffff',
    'panel': '#f0f2f8',
    'white': '#1a1a2e',
    'grey': '#c0c4d8',
    'dim': '#8888aa',
    'gold': '#c47a00',
    'cyan': '#0077aa',
    'red': '#cc2233',
    'green': '#007744'
}

# Color maps
COLORMAPS = {
    'viridis': 'viridis',
    'plasma': 'plasma',
    'inferno': 'inferno',
    'magma': 'magma',
    'cividis': 'cividis',
    'ntk': ['#050515', '#0a0a1a', '#94c544']
}

# Data settings
DATA_SETTINGS = {
    'n_samples': 100,
    'noise_std': 0.1,
    'x_range': (-3, 3),
    'test_points': 50
}

# Network settings
NETWORK_SETTINGS = {
    'hidden_layers': [100],
    'activation': 'tanh',
    'learning_rate': 0.005,
    'n_iterations': 100,
    'widths': [5, 50, 500],
    'n_layers': 3
}

# Gradient descent settings
GRADIENT_DESCENT_SETTINGS = {
    'n_samples': 100,
    'learning_rate': 0.01,
    'iterations': 100,
    'line_range': (-6, 6),
    'error_range': (-10, 15)
}

# NTK settings
NTK_SETTINGS = {
    'n_points': 12,
    'n_steps': 60,
    'lr': 0.002,
    'finite_width': 8,
    'infinite_width': 512
}

def load_config(config_file=None):
    """Load configuration from YAML file if provided."""
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    return {}

# Create directories if they don't exist
for dir_path in [DATA_DIR, FIGURES_DIR, SCRIPTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)