"""
Data loading and preprocessing utilities for Neural Tangent Kernels visualizations.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from .config import DATA_SETTINGS, RANDOM_SEED


def set_random_seeds(seed: int = RANDOM_SEED):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def generate_synthetic_data(n_samples: int = DATA_SETTINGS['n_samples'],
                           noise_std: float = DATA_SETTINGS['noise_std'],
                           x_range: Tuple[float, float] = DATA_SETTINGS['x_range'],
                           data_type: str = 'sin') -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for regression tasks.
    
    Args:
        n_samples: Number of data points
        noise_std: Standard deviation of noise
        x_range: Range of x values (min, max)
        data_type: Type of function to generate ('sin', 'linear', 'quadratic')
    
    Returns:
        Tuple of (x, y) arrays
    """
    set_random_seeds()
    
    x = np.linspace(x_range[0], x_range[1], n_samples).reshape(-1, 1)
    
    if data_type == 'sin':
        y = np.sin(x).flatten()
    elif data_type == 'linear':
        y = 2 * x.flatten() + 1
    elif data_type == 'quadratic':
        y = x.flatten() ** 2
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    # Add noise
    y_noisy = y + noise_std * np.random.randn(n_samples)
    
    return x.flatten(), y_noisy


def generate_linear_regression_data(n_samples: int = 100,
                                   noise_std: float = 1.0,
                                   x_range: Tuple[float, float] = (-5, 5)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate linear regression data for gradient descent visualization.
    
    Args:
        n_samples: Number of data points
        noise_std: Standard deviation of noise
        x_range: Range of x values
    
    Returns:
        Tuple of (x, y) arrays
    """
    set_random_seeds()
    
    x = np.linspace(x_range[0], x_range[1], n_samples)
    y = 2 * x + 1 + noise_std * np.random.randn(n_samples)
    
    return x, y


def load_mnist_data(data_dir: Path, 
                   train: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load MNIST data (placeholder for actual implementation).
    
    Args:
        data_dir: Directory containing MNIST data
        train: Whether to load training or test data
    
    Returns:
        Tuple of (images, labels)
    """
    # This is a placeholder - actual MNIST loading would use
    # torchvision.datasets.MNIST or similar
    data_path = data_dir / "MNIST" / "raw"
    
    if not data_path.exists():
        raise FileNotFoundError(f"MNIST data not found at {data_path}")
    
    print(f"MNIST data found at {data_path}")
    print("Note: Actual MNIST loading implementation would go here")
    
    # Return dummy data for now
    n_samples = 1000 if train else 200
    return np.random.randn(n_samples, 28, 28), np.random.randint(0, 10, n_samples)


def create_data_split(x: np.ndarray, 
                     y: np.ndarray,
                     train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create train/validation data split.
    
    Args:
        x: Input data
        y: Target data
        train_ratio: Ratio of training data
    
    Returns:
        Tuple of (x_train, x_val, y_train, y_val)
    """
    set_random_seeds()
    
    n_samples = len(x)
    n_train = int(n_samples * train_ratio)
    
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    x_train, x_val = x[train_indices], x[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    
    return x_train, x_val, y_train, y_val


def normalize_data(x: np.ndarray) -> np.ndarray:
    """
    Normalize data to zero mean and unit variance.
    
    Args:
        x: Input data
    
    Returns:
        Normalized data
    """
    mean = np.mean(x)
    std = np.std(x)
    
    if std == 0:
        return x - mean
    else:
        return (x - mean) / std


def create_test_points(x_range: Tuple[float, float] = DATA_SETTINGS['x_range'],
                      n_points: int = DATA_SETTINGS['test_points']) -> np.ndarray:
    """
    Create test points for evaluation.
    
    Args:
        x_range: Range of x values
        n_points: Number of test points
    
    Returns:
        Array of test points
    """
    return np.linspace(x_range[0], x_range[1], n_points)


def batch_generator(x: np.ndarray, 
                   y: np.ndarray,
                   batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generator for batched data.
    
    Args:
        x: Input data
        y: Target data
        batch_size: Size of each batch
    
    Yields:
        Tuple of (batch_x, batch_y)
    """
    set_random_seeds()
    
    n_samples = len(x)
    indices = np.random.permutation(n_samples)
    
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_x = x[batch_indices]
        batch_y = y[batch_indices]
        
        yield batch_x, batch_y