"""
Neural Tangent Kernels: Modelling Neural Learning

A modular, reproducible codebase for generating presentation visualizations
on Neural Tangent Kernels and related neural network concepts.
"""

__version__ = "1.0.0"
__author__ = "Neural Tangent Kernels Team"
__email__ = "contact@example.com"

# Import core modules
from . import config
from . import data
from . import models
from . import visualization
from . import utils

# Set default random seeds
from .utils import set_random_seeds
set_random_seeds()

__all__ = [
    'config',
    'data', 
    'models',
    'visualization',
    'utils',
    'set_random_seeds'
]