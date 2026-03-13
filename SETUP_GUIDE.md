# Neural Tangent Kernels: Modelling Neural Learning - Setup Guide

## Overview
This repository contains a comprehensive implementation for visualizing Neural Tangent Kernel (NTK) theory and neural network optimization concepts. The project includes production-ready scripts for generating various visualization figures.

## Quick Start

### 1. Environment Setup
```bash
# Create conda environment
conda create -n ntk_modelling_neural_networks python=3.10
conda activate ntk_modelling_neural_networks

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy matplotlib pyyaml scikit-learn
pip install jupyter ipywidget
```

### 2. Run Visualization Scripts
```bash
# Set PYTHONPATH and run scripts
export PYTHONPATH="/path/to/repo:$PYTHONPATH"

# Generate gradient descent visualizations
python scripts/generate_gradient_descent_figures.py

# Generate infinite width network visualizations
python scripts/generate_infinite_width_figures.py

# Generate linear approximation visualizations
python scripts/generate_linear_approximation_figures.py

# Generate NTK visualizations
python scripts/generate_ntk_figures.py
```

## Project Structure

```
Neural Tangent Kernels: Modelling Neural Learning/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration settings
│   ├── data.py              # Data generation utilities
│   ├── models.py            # Neural network models
│   ├── utils.py             # Utility functions
│   └── visualization.py     # Visualization utilities
├── scripts/
│   ├── generate_gradient_descent_figures.py
│   ├── generate_infinite_width_figures.py
│   ├── generate_linear_approximation_figures.py
│   └── generate_ntk_figures.py
├── notebooks/
│   ├── dev/                 # Development notebooks
│   └── production/          # Production notebooks
├── figures/                 # Generated visualization files
├── requirements.txt         # Python dependencies
├── config.yaml             # Configuration file
├── environment.yml         # Conda environment export
└── README.md              # This file
```

## Generated Visualizations

### 1. Gradient Descent Visualizations
- **line_fitting.gif**: Animation showing line fitting process
- **line_fitting_final.png**: Final fitted line
- **error_landscape.gif**: Animation of error landscape with optimization path
- **error_landscape_final.png**: Complete optimization path
- **loss_convergence.png**: Loss convergence over iterations

### 2. Infinite Width Network Visualizations
- **landscape_comparison.png**: Comparison of loss landscapes for different widths
- **cross_section.png**: Cross-section analysis of loss landscapes
- **smoothness_metric.png**: Smoothness metric vs network width
- **3d_landscape.png**: 3D visualization of loss landscape

### 3. Linear Approximation Visualizations
- **linear_approximation_interactive.png**: Interactive visualization
- **linear_approximation_animation.gif**: Animation showing h → 0
- **linear_approximation_error_analysis.png**: Error analysis
- **linear_approximation_comparison.png**: Linear vs higher-order approximations

### 4. NTK Visualizations
- **ntk_demonstration.png**: NTK demonstration figure
- **frozen_kernel_comparison.png**: Frozen kernel comparison
- **ntk_training_animation.gif**: Training animation
- **ntk_theory_comparison.png**: Theory vs practice comparison

## Configuration

### Environment Variables
- `PYTHONPATH`: Must include the repository root directory
- `CONDA_DEFAULT_ENV`: Should be set to `ntk_modelling_neural_networks`

### Configuration Files
- `src/config.py`: Python configuration with all settings
- `config.yaml`: YAML configuration file
- `environment.yml`: Conda environment specification

## Dependencies

### Core Dependencies
- **torch**: Deep learning framework
- **torchvision**: Computer vision utilities
- **torchaudio**: Audio processing utilities
- **numpy**: Numerical computing
- **scipy**: Scientific computing
- **matplotlib**: Plotting library
- **scikit-learn**: Machine learning utilities
- **pyyaml**: YAML parsing

### Optional Dependencies
- **jupyter**: Interactive computing
- **ipywidgets**: Interactive widgets

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH` is set correctly
   ```bash
   export PYTHONPATH="/path/to/repo:$PYTHONPATH"
   ```

2. **Module Not Found**: Make sure you're in the correct directory
   ```bash
   cd "/path/to/repo"
   ```

3. **Conda Environment Issues**: Activate the correct environment
   ```bash
   conda activate ntk_modelling_neural_networks
   ```

4. **PyTorch Installation**: Use CPU version if CUDA is not available
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

### Runtime Warnings
Some scripts may produce runtime warnings due to numerical computations. These are generally safe to ignore.

## Development

### Adding New Visualizations
1. Create new script in `scripts/` directory
2. Import necessary modules from `src/`
3. Use configuration from `src.config`
4. Save figures to `figures/` directory using `visualization.save_figure()`

### Code Style
The project follows PEP8 guidelines with some flexibility for line length in visualization code.

## Performance Notes

- Some visualizations may take several minutes to generate
- Memory usage depends on the complexity of the visualization
- For faster execution, reduce the number of iterations or data points in configuration

## License

This project is provided for educational and research purposes.

## Support

For issues or questions, please refer to the project documentation or create an issue in the repository.