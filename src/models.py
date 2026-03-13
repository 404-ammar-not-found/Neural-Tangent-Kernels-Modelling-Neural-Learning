"""
Neural network models and training utilities for Neural Tangent Kernels.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Dict, Any, Callable
from sklearn.neural_network import MLPRegressor

from .config import NETWORK_SETTINGS, RANDOM_SEED


# Detect device availability
def get_device():
    """Get the best available device (MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


DEVICE = get_device()


class SimpleNN(nn.Module):
    """
    Simple neural network with configurable width and depth.
    """
    
    def __init__(self, width: int = 100, n_layers: int = 3, 
                 input_dim: int = 1, output_dim: int = 1,
                 activation: str = 'tanh'):
        """
        Initialize neural network.
        
        Args:
            width: Number of neurons in hidden layers
            n_layers: Total number of layers (including input/output)
            input_dim: Input dimension
            output_dim: Output dimension
            activation: Activation function ('tanh', 'relu', 'sigmoid')
        """
        super(SimpleNN, self).__init__()
        
        self.width = width
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Choose activation function
        if activation == 'tanh':
            self.activation_fn = torch.tanh
        elif activation == 'relu':
            self.activation_fn = torch.relu
        elif activation == 'sigmoid':
            self.activation_fn = torch.sigmoid
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        layers = []
        dims = [input_dim] + [width] * (n_layers - 2) + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # Don't add activation after last layer
                layers.append(self.activation_fn)
        
        self.network = nn.Sequential(*layers)
        
        # Move to device
        self.to(DEVICE)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


def initialize_weights(width: int = 100, n_layers: int = 3) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Initialize network weights and biases.
    
    Args:
        width: Width of hidden layers
        n_layers: Number of layers
    
    Returns:
        List of (weights, biases) tuples
    """
    np.random.seed(RANDOM_SEED)
    
    dims = [1] + [width] * (n_layers - 2) + [1]
    params = []
    
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        # Initialize weights with Xavier/Glorot initialization
        W = np.random.randn(d_out, d_in) / np.sqrt(d_in)
        b = np.zeros(d_out)
        params.append((W, b))
    
    return params


def forward_scalar(x: float, params: List[Tuple[np.ndarray, np.ndarray]]) -> float:
    """
    Forward pass for scalar input.
    
    Args:
        x: Scalar input
        params: List of (weights, biases) tuples
    
    Returns:
        Scalar output
    """
    h = np.array([[x]])  # Shape (1, 1)
    
    for i, (W, b) in enumerate(params):
        z = h @ W.T + b
        if i < len(params) - 1:  # Don't apply activation after last layer
            h = np.tanh(z)
        else:
            h = z
    
    return h[0, 0]


def forward_batch(x: np.ndarray, params: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Forward pass for batch of inputs.
    
    Args:
        x: Input array of shape (n_samples,)
        params: List of (weights, biases) tuples
    
    Returns:
        Output array of shape (n_samples,)
    """
    return np.array([forward_scalar(xi, params) for xi in x])


def compute_jacobian(x: float, params: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Compute Jacobian of output with respect to all parameters.
    
    Args:
        x: Scalar input
        params: List of (weights, biases) tuples
    
    Returns:
        Flattened Jacobian vector
    """
    eps = 1e-4
    flat_params = np.concatenate([np.concatenate([W.ravel(), b]) for W, b in params])
    
    def output_from_flat(flat_params):
        idx = 0
        new_params = []
        for W, b in params:
            sz_W = W.size
            sz_b = b.size
            new_params.append((flat_params[idx:idx+sz_W].reshape(W.shape), 
                            flat_params[idx+sz_W:idx+sz_W+sz_b]))
            idx += sz_W + sz_b
        return forward_scalar(x, new_params)
    
    jac = np.zeros(len(flat_params))
    for k in range(len(flat_params)):
        e = np.zeros(len(flat_params))
        e[k] = eps
        jac[k] = (output_from_flat(flat_params + e) - output_from_flat(flat_params - e)) / (2 * eps)
    
    return jac


def compute_ntk_matrix(x_train: np.ndarray, params: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Compute Neural Tangent Kernel matrix.
    
    Args:
        x_train: Training inputs
        params: Network parameters
    
    Returns:
        NTK matrix of shape (n_samples, n_samples)
    """
    n = len(x_train)
    jacobians = [compute_jacobian(xi, params) for xi in x_train]
    
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            K[i, j] = K[j, i] = jacobians[i] @ jacobians[j]
    
    return K


def train_step(params: List[Tuple[np.ndarray, np.ndarray]], 
               x_train: np.ndarray, y_train: np.ndarray,
               lr: float = NETWORK_SETTINGS['learning_rate']) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Perform one training step using gradient descent.
    
    Args:
        params: Current network parameters
        x_train: Training inputs
        y_train: Training targets
        lr: Learning rate
    
    Returns:
        Updated parameters
    """
    eps = 1e-5
    flat_params = np.concatenate([np.concatenate([W.ravel(), b]) for W, b in params])
    
    def loss_from_flat(flat_params):
        idx = 0
        new_params = []
        for W, b in params:
            sz_W = W.size
            sz_b = b.size
            new_params.append((flat_params[idx:idx+sz_W].reshape(W.shape), 
                            flat_params[idx+sz_W:idx+sz_W+sz_b]))
            idx += sz_W + sz_b
        
        # Compute predictions
        predictions = np.array([forward_scalar(xi, new_params) for xi in x_train])
        return 0.5 * np.mean((predictions - y_train) ** 2)
    
    # Compute gradient
    grad = np.zeros_like(flat_params)
    for k in range(len(flat_params)):
        e = np.zeros_like(flat_params)
        e[k] = eps
        grad[k] = (loss_from_flat(flat_params + e) - loss_from_flat(flat_params - e)) / (2 * eps)
    
    # Update parameters
    flat_new = flat_params - lr * grad
    
    # Convert back to parameter format
    idx = 0
    new_params = []
    for W, b in params:
        sz_W = W.size
        sz_b = b.size
        new_params.append((flat_new[idx:idx+sz_W].reshape(W.shape),
                        flat_new[idx+sz_W:idx+sz_W+sz_b]))
        idx += sz_W + sz_b
    
    return new_params


def train_network(x_train: np.ndarray, y_train: np.ndarray,
                 width: int = 100, n_iterations: int = 100,
                 lr: float = 0.005) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[float]]:
    """
    Train a neural network using gradient descent.
    
    Args:
        x_train: Training inputs
        y_train: Training targets
        width: Network width
        n_iterations: Number of training iterations
        lr: Learning rate
    
    Returns:
        Tuple of (final_params, loss_history)
    """
    # Initialize parameters
    params = initialize_weights(width=width, n_layers=NETWORK_SETTINGS['n_layers'])
    
    loss_history = []
    
    for i in range(n_iterations):
        # Compute current loss
        predictions = forward_batch(x_train, params)
        loss = 0.5 * np.mean((predictions - y_train) ** 2)
        loss_history.append(loss)
        
        # Update parameters
        params = train_step(params, x_train, y_train, lr=lr)
    
    return params, loss_history


def create_torch_nn(width: int = 100, n_layers: int = 3, 
                   input_dim: int = 1, output_dim: int = 1,
                   activation: str = 'tanh') -> SimpleNN:
    """
    Create PyTorch neural network with device support.
    
    Args:
        width: Width of hidden layers
        n_layers: Number of layers
        input_dim: Input dimension
        output_dim: Output dimension
        activation: Activation function
    
    Returns:
        Configured SimpleNN model
    """
    return SimpleNN(width=width, n_layers=n_layers, 
                    input_dim=input_dim, output_dim=output_dim,
                    activation=activation)


def train_torch_nn(model: SimpleNN, x_train: torch.Tensor, y_train: torch.Tensor,
                  n_iterations: int = 100, lr: float = 0.005) -> List[float]:
    """
    Train PyTorch neural network.
    
    Args:
        model: PyTorch neural network
        x_train: Training inputs (torch.Tensor)
        y_train: Training targets (torch.Tensor)
        n_iterations: Number of training iterations
        lr: Learning rate
    
    Returns:
        Loss history
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    loss_history = []
    
    for i in range(n_iterations):
        # Forward pass
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
    
    return loss_history


def create_sklearn_model(hidden_layer_sizes: Tuple[int, ...] = (100, 100),
                        activation: str = 'relu',
                        max_iter: int = 1000,
                        random_state: int = RANDOM_SEED) -> MLPRegressor:
    """
    Create scikit-learn MLP regressor.
    
    Args:
        hidden_layer_sizes: Sizes of hidden layers
        activation: Activation function
        max_iter: Maximum iterations
        random_state: Random seed
    
    Returns:
        Configured MLPRegressor
    """
    return MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        max_iter=max_iter,
        random_state=random_state
    )


def get_model_info(model: SimpleNN) -> Dict[str, Any]:
    """
    Get information about the PyTorch model.
    
    Args:
        model: PyTorch neural network
    
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'device': str(model.device),
        'architecture': str(model)
    }