"""
Utility functions for Neural Tangent Kernels visualizations.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union
import yaml
import json

from .config import RANDOM_SEED


def set_random_seeds(seed: int = RANDOM_SEED):
    """Set random seeds for reproducibility across all libraries."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: Union[str, Path]):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def format_float(value: float, precision: int = 4) -> str:
    """
    Format float value with specified precision.
    
    Args:
        value: Float value to format
        precision: Number of decimal places
    
    Returns:
        Formatted string
    """
    return f"{value:.{precision}f}"


def format_percentage(value: float, precision: int = 2) -> str:
    """
    Format value as percentage.
    
    Args:
        value: Value to format (0-1)
        precision: Number of decimal places
    
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{precision}f}%"


def create_directory_structure(base_path: Union[str, Path]):
    """
    Create standard directory structure.
    
    Args:
        base_path: Base directory path
    """
    base_path = Path(base_path)
    
    directories = [
        'figures',
        'figures/gradient_descent',
        'figures/infinite_width',
        'figures/linear_approximation',
        'figures/neuron_comparison',
        'figures/ntk',
        'scripts',
        'notebooks',
        'notebooks/dev',
        'notebooks/production',
        'data',
        'data/MNIST',
        'data/MNIST/raw'
    ]
    
    for directory in directories:
        (base_path / directory).mkdir(parents=True, exist_ok=True)


def calculate_smoothness_metric(loss_grid: np.ndarray) -> float:
    """
    Calculate smoothness metric for loss landscapes.
    
    Args:
        loss_grid: 2D loss landscape grid
    
    Returns:
        Smoothness metric (lower = smoother)
    """
    # Compute second derivatives (discrete Laplacian)
    second_deriv_x = np.diff(loss_grid, n=2, axis=0)
    second_deriv_y = np.diff(loss_grid, n=2, axis=1)
    
    # Variance of second derivatives (lower = smoother)
    smoothness_x = np.var(second_deriv_x)
    smoothness_y = np.var(second_deriv_y)
    
    return (smoothness_x + smoothness_y) / 2


def normalize_array(arr: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize array using specified method.
    
    Args:
        arr: Input array
        method: Normalization method ('minmax', 'standard', 'max')
    
    Returns:
        Normalized array
    """
    if method == 'minmax':
        min_val, max_val = arr.min(), arr.max()
        if max_val > min_val:
            return (arr - min_val) / (max_val - min_val)
        else:
            return arr - min_val
    
    elif method == 'standard':
        mean, std = arr.mean(), arr.std()
        if std > 0:
            return (arr - mean) / std
        else:
            return arr - mean
    
    elif method == 'max':
        max_val = np.abs(arr).max()
        if max_val > 0:
            return arr / max_val
        else:
            return arr
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def moving_average(arr: np.ndarray, window_size: int) -> np.ndarray:
    """
    Compute moving average of array.
    
    Args:
        arr: Input array
        window_size: Size of moving window
    
    Returns:
        Moving average array
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1")
    
    if window_size > len(arr):
        window_size = len(arr)
    
    # Pad array to handle edges
    padded = np.pad(arr, (window_size // 2, window_size - window_size // 2), 
                    mode='edge')
    
    # Compute moving average
    weights = np.ones(window_size) / window_size
    return np.convolve(padded, weights, mode='valid')


def find_local_minima(arr: np.ndarray, threshold: float = 0.01) -> List[int]:
    """
    Find local minima in array.
    
    Args:
        arr: Input array
        threshold: Minimum relative difference from neighbors
    
    Returns:
        List of indices of local minima
    """
    minima = []
    
    for i in range(1, len(arr) - 1):
        if (arr[i] < arr[i - 1] and 
            arr[i] < arr[i + 1] and 
            (arr[i - 1] - arr[i]) / arr[i - 1] > threshold):
            minima.append(i)
    
    return minima


def compute_drift(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
    """
    Compute relative drift between two matrices.
    
    Args:
        matrix1: First matrix
        matrix2: Second matrix
    
    Returns:
        Relative drift (Frobenius norm)
    """
    norm_diff = np.linalg.norm(matrix2 - matrix1, 'fro')
    norm_1 = np.linalg.norm(matrix1, 'fro')
    
    if norm_1 > 0:
        return norm_diff / norm_1
    else:
        return 0.0


def create_experiment_id(prefix: str = "exp") -> str:
    """
    Create unique experiment ID.
    
    Args:
        prefix: Prefix for experiment ID
    
    Returns:
        Unique experiment ID
    """
    import time
    timestamp = int(time.time())
    return f"{prefix}_{timestamp}"


def save_results(results: Dict[str, Any], 
                filename: str,
                save_dir: Union[str, Path] = "results"):
    """
    Save experiment results to JSON file.
    
    Args:
        results: Results dictionary
        filename: Output filename
        save_dir: Directory to save results
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = save_dir / filename
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    json_results = convert_for_json(results)
    
    with open(filepath, 'w') as f:
        json.dump(json_results, f, indent=2)


def load_results(filename: str, 
                save_dir: Union[str, Path] = "results") -> Dict[str, Any]:
    """
    Load experiment results from JSON file.
    
    Args:
        filename: Input filename
        save_dir: Directory containing results
    
    Returns:
        Results dictionary
    """
    save_dir = Path(save_dir)
    filepath = save_dir / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        True if valid, False otherwise
    """
    required_keys = {
        'paths': ['data_dir', 'figures_dir', 'scripts_dir'],
        'seeds': ['random', 'torch', 'numpy'],
        'figure_settings': ['dpi', 'style', 'size'],
        'colors': ['primary', 'secondary', 'bg'],
        'data_settings': ['n_samples', 'noise_std', 'x_range'],
        'network_settings': ['hidden_layers', 'activation', 'learning_rate']
    }
    
    for section, keys in required_keys.items():
        if section not in config:
            return False
        
        for key in keys:
            if key not in config[section]:
                return False
    
    return True


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage.
    
    Returns:
        Dictionary with memory usage in MB
    """
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    return {
        'rss': process.memory_info().rss / 1024 / 1024,  # Resident Set Size
        'vms': process.memory_info().vms / 1024 / 1024,  # Virtual Memory Size
    }


def time_function(func, *args, **kwargs):
    """
    Time function execution.
    
    Args:
        func: Function to time
        *args: Function arguments
        **kwargs: Function keyword arguments
    
    Returns:
        Tuple of (result, execution_time_seconds)
    """
    import time
    
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    return result, end_time - start_time


def batch_process(items: List[Any], 
                 process_func: callable,
                 batch_size: int = 32,
                 show_progress: bool = False) -> List[Any]:
    """
    Process items in batches.
    
    Args:
        items: List of items to process
        process_func: Function to apply to each batch
        batch_size: Size of each batch
        show_progress: Whether to show progress
    
    Returns:
        List of results
    """
    results = []
    n_batches = (len(items) + batch_size - 1) // batch_size
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(items))
        batch = items[start_idx:end_idx]
        
        if show_progress:
            print(f"Processing batch {i + 1}/{n_batches} "
                  f"({start_idx}-{end_idx - 1} of {len(items)})")
        
        batch_results = process_func(batch)
        results.extend(batch_results)
    
    return results