# Neural Tangent Kernels: Modelling Neural Learning

A production-ready, reproducible repository for generating presentation visualizations on Neural Tangent Kernels and related neural network concepts.

## Abstract for the talk 

Neural networks are expensive and time-consuming to train, therefore is it possible
to model the evolution of the network with much less compute power? Modern
neural networks are over-parameterised and seen as a mysterious black box. However, when considering their width to be infinite, their training dynamics through
gradient descent corresponds to previously defined kernel methods. Introduced in
2018, Jacot, Gabriel and Hongler defined a neural tangent kernel (NTK), providing
a strong theoretical framework, formalising how neural networks can learn and
generalise. In this talk, I will introduce NTKs and define the mathematics and
intuition behind them. I will first define how neural networks can be modelled as a
high-parameter function and then highlight how their learning dynamics linearise
as their width increases. Then this linearisation leads to predictive generalisation
results, connecting neural networks to Gaussian processes. Then, I shall introduce
the applications of NTKs in practical machine learning and show a demonstration
of their application on a sin(x) curve. This talk will highlight the intersection
between functional analysis, differential equations and probability and their contribution to define a key concept for mathematical machine learning.

## 📁 Repository Structure

```
Neural Tangent Kernels: Modelling Neural Learning/
├── src/                          # Core source modules
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration management
│   ├── data.py                  # Data loading and preprocessing
│   ├── models.py                # Neural network models (MPS/CUDA/CPU support)
│   ├── visualization.py         # Common plotting utilities
│   └── utils.py                 # Utility functions
├── notebooks/                   # Development notebooks
│   ├── dev/                     # Exploratory development notebooks
│   │   ├── linear_regression_gradient_descent.ipynb
│   │   ├── neuron_comparison.ipynb
│   │   └── NTK.ipynb
│   └── production/             # Production-ready scripts
├── scripts/                     # Production visualization scripts
│   ├── generate_gradient_descent_figures.py
│   ├── generate_infinite_width_figures.py
│   ├── generate_linear_approximation_figures.py
│   └── generate_ntk_figures.py
├── figures/                     # Generated visualization files
│   ├── gradient_descent/       # Gradient descent visualizations
│   ├── infinite_width/          # Infinite width network visualizations
│   ├── linear_approximation/    # Linear approximation visualizations
│   ├── neuron_comparison/      # Neuron comparison visualizations
│   └── ntk/                     # Neural Tangent Kernel visualizations
├── data/                        # Data storage
│   └── MNIST/                   # MNIST dataset
├── config.yaml                 # Configuration settings
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Environment Setup

Create and activate a conda environment:

```bash
# Create conda environment
conda create -n ntk_modelling_neural_networks python=3.10

# Activate environment
conda activate ntk_modelling_neural_networks

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

Generate all visualization figures:

```bash
# Generate gradient descent visualizations
python scripts/generate_gradient_descent_figures.py

# Generate infinite width network visualizations
python scripts/generate_infinite_width_figures.py

# Generate linear approximation visualizations
python scripts/generate_linear_approximation_figures.py

# Generate NTK visualizations
python scripts/generate_ntk_figures.py
```

## 📊 Visualization Scripts Overview

### 1. Gradient Descent Visualization
**Script**: `scripts/generate_gradient_descent_figures.py`

**Generated Files**:
- `figures/gradient_descent/line_fitting.gif` - Animation of line fitting
- `figures/gradient_descent/line_fitting_final.png` - Final fitted line
- `figures/gradient_descent/error_landscape.gif` - Error landscape animation
- `figures/gradient_descent/error_landscape_final.png` - Complete optimization path
- `figures/gradient_descent/loss_convergence.png` - Loss convergence plot

**Purpose**: Demonstrates batch gradient descent for linear regression, including error landscapes and optimization paths.

### 2. Infinite Width Network Visualization
**Script**: `scripts/generate_infinite_width_figures.py`

**Generated Files**:
- `figures/infinite_width/landscape_comparison.png` - Width comparison
- `figures/infinite_width/cross_section.png` - Cross-section analysis
- `figures/infinite_width/smoothness_metric.png` - Smoothness vs width
- `figures/infinite_width/3d_landscape.png` - 3D landscape visualization

**Purpose**: Shows how neural network loss landscapes become smoother as network width increases.

### 3. Linear Approximation Visualization
**Script**: `scripts/generate_linear_approximation_figures.py`

**Generated Files**:
- `figures/linear_approximation/linear_approximation_interactive.png` - Interactive visualization
- `figures/linear_approximation/linear_approximation_animation.gif` - Animation
- `figures/linear_approximation/linear_approximation_error_analysis.png` - Error analysis
- `figures/linear_approximation/linear_approximation_comparison.png` - Comparison plot

**Purpose**: Demonstrates linear approximation concepts and Taylor series.

### 4. Neural Tangent Kernel Visualization
**Script**: `scripts/generate_ntk_figures.py`

**Generated Files**:
- `figures/ntk/ntk_demonstration.png` - NTK demonstration
- `figures/ntk/frozen_kernel_comparison.png` - Frozen kernel comparison
- `figures/ntk/ntk_training_animation.gif` - Training animation
- `figures/ntk/ntk_theory_comparison.png` - Theory vs practice

**Purpose**: Demonstrates NTK theory, kernel freezing, and training dynamics.

## ⚙️ Configuration

### Configuration File: `config.yaml`

Key configuration sections:

```yaml
# Figure settings
figure_settings:
  dpi: 300                    # Resolution
  style: "seaborn"            # Plotting style
  size: [10, 6]               # Figure size

# Network settings
network_settings:
  hidden_layers: [100]        # Hidden layer sizes
  activation: "tanh"          # Activation function
  learning_rate: 0.005       # Learning rate

# Data settings
data_settings:
  n_samples: 100              # Number of samples
  noise_std: 0.1              # Noise standard deviation
  x_range: [-3, 3]            # Input range
```

### Environment Detection

The code automatically detects and uses the best available device:

```python
# Automatic device detection (MPS > CUDA > CPU)
from src.models import get_device
device = get_device()  # Returns torch.device('mps'), 'cuda', or 'cpu')
```

## 🔧 Advanced Usage

### Custom Configuration

Create a custom configuration file:

```python
from src.config import load_config
config = load_config('custom_config.yaml')
```

### Running Individual Scripts

```bash
# Generate specific visualization with custom parameters
python scripts/generate_ntk_figures.py
```

### Using Development Notebooks

Exploratory notebooks are available in `notebooks/dev/` for experimentation:

```bash
jupyter notebook notebooks/dev/
```

### Batch Processing

Generate multiple visualizations simultaneously:

```bash
# Run all scripts in sequence
for script in scripts/generate_*.py; do
    python "$script"
done
```

## 📈 Dependencies

### Core Dependencies
- `numpy>=1.21.0` - Numerical computing
- `scipy>=1.7.0` - Scientific computing
- `matplotlib>=3.5.0` - Plotting
- `torch>=1.10.0` - Deep learning (MPS/CUDA/CPU support)
- `scikit-learn>=1.0.0` - Machine learning
- `PyYAML>=5.4.0` - Configuration management
- `Pillow>=8.3.0` - Image processing

### Optional Dependencies
- `jupyter>=1.0.0` - Interactive notebooks
- `pytest>=6.2.0` - Testing
- `black>=21.0.0` - Code formatting
- `flake8>=3.9.0` - Linting

### Device Support
- **Apple Silicon (MPS)**: Automatic detection and optimization
- **NVIDIA CUDA**: Automatic detection and optimization
- **CPU**: Fallback for all systems

## 🧪 Testing and Validation

### Running Tests

```bash
# Install test dependencies
pip install pytest

# Run tests
pytest tests/
```

### Reproducibility

All scripts use fixed random seeds for reproducible results:

```python
from src.utils import set_random_seeds
set_random_seeds(42)  # Fixed seed for reproducibility
```

## 📊 Performance Considerations

### Device Optimization

- **MPS (Apple Silicon)**: Optimized for Apple M1/M2 chips
- **CUDA**: GPU acceleration for NVIDIA GPUs
- **CPU**: CPU fallback for maximum compatibility

### Memory Management

- Batch processing for large datasets
- Automatic memory cleanup
- Efficient tensor operations on GPU

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Device Issues**: The code automatically handles device detection
   ```python
   from src.models import get_device
   print(f"Using device: {get_device()}")
   ```

3. **Memory Issues**: Reduce batch size or use smaller networks
   ```yaml
   # In config.yaml
   network_settings:
     hidden_layers: [50]  # Smaller network
   ```

### Performance Optimization

- Use GPU acceleration when available
- Adjust figure resolution in `config.yaml`
- Reduce network complexity for faster training

## 📝 Development

### Code Style

The project follows PEP8 standards:

```bash
# Format code
black src/ scripts/

# Lint code
flake8 src/ scripts/
```

### Adding New Visualizations

1. Create new script in `scripts/`
2. Add corresponding figure directory in `figures/`
3. Update configuration in `config.yaml`
4. Document in `README.md`

### Contributing

1. Fork the repository
2. Create a feature branch
3. Follow PEP8 standards
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For questions or issues:
- Check the troubleshooting section
- Review the development notebooks
- Create an issue in the repository

---

**Note**: This repository is designed for educational and presentation purposes. All visualizations are generated programmatically and can be regenerated with the provided scripts.