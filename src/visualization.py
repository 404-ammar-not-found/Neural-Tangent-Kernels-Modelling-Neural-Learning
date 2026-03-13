"""
Common visualization utilities for Neural Tangent Kernels.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

from .config import (
    FIGURE_SETTINGS, COLORS, COLORMAPS, 
    FIGURE_DPI, FIGURE_SIZE, ANIMATION_FPS
)


def set_plotting_style(style: str = FIGURE_SETTINGS['style']):
    """Set matplotlib plotting style."""
    plt.style.use(style)


def create_figure(figsize: Tuple[float, float] = FIGURE_SIZE,
                 dpi: int = FIGURE_DPI,
                 facecolor: str = COLORS['bg'],
                 **kwargs) -> plt.Figure:
    """
    Create a figure with consistent styling.
    
    Args:
        figsize: Figure size
        dpi: Dots per inch
        facecolor: Background color
        **kwargs: Additional arguments to plt.figure
    
    Returns:
        Figure object
    """
    fig = plt.figure(figsize=figsize, dpi=dpi, 
                    facecolor=facecolor, **kwargs)
    return fig


def create_subplot_grid(rows: int, cols: int, 
                       figsize: Tuple[float, float] = FIGURE_SIZE,
                       dpi: int = FIGURE_DPI,
                       sharex: bool = False,
                       sharey: bool = False,
                       **kwargs) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a grid of subplots with consistent styling.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        figsize: Figure size
        dpi: Dots per inch
        sharex: Whether to share x-axis
        sharey: Whether to share y-axis
        **kwargs: Additional arguments to plt.subplots
    
    Returns:
        Tuple of (figure, axes)
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi,
                           sharex=sharex, sharey=sharey, **kwargs)
    
    # Ensure axes is always a 2D array
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Style all axes
    for ax in axes.flat:
        style_axes(ax)
    
    return fig, axes


def style_axes(ax: plt.Axes, 
               facecolor: str = COLORS['panel'],
               grid_alpha: float = 0.3,
               labelsize: int = 12,
               tick_color: str = COLORS['white'],
               spine_color: str = COLORS['grey']):
    """
    Apply consistent styling to axes.
    
    Args:
        ax: Axes object to style
        facecolor: Background color
        grid_alpha: Grid transparency
        labelsize: Font size for labels
        tick_color: Color for tick labels
        spine_color: Color for axis spines
    """
    ax.set_facecolor(facecolor)
    ax.grid(True, alpha=grid_alpha)
    
    # Style spines
    for spine in ax.spines.values():
        spine.set_edgecolor(spine_color)
        spine.set_alpha(0.8)
    
    # Style ticks and labels
    ax.tick_params(colors=tick_color, labelsize=labelsize)
    ax.xaxis.label.set_color(tick_color)
    ax.yaxis.label.set_color(tick_color)
    ax.title.set_color(tick_color)


def save_figure(fig: plt.Figure, 
                filename: str,
                dpi: int = FIGURE_DPI,
                bbox_inches: str = 'tight',
                **kwargs) -> Path:
    """
    Save figure to file.
    
    Args:
        fig: Figure to save
        filename: Output filename
        dpi: Resolution
        bbox_inches: Bounding box settings
        **kwargs: Additional arguments to fig.savefig
    
    Returns:
        Path to saved file
    """
    # Ensure figures directory exists
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Create full path
    filepath = figures_dir / filename
    
    # Save figure
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    plt.close(fig)
    
    return filepath


def save_animation(anim: FuncAnimation, 
                 filename: str,
                 fps: int = ANIMATION_FPS,
                 dpi: int = FIGURE_DPI,
                 **kwargs) -> Path:
    """
    Save animation as GIF.
    
    Args:
        anim: Animation object
        filename: Output filename
        fps: Frames per second
        dpi: Resolution
        **kwargs: Additional arguments to anim.save
    
    Returns:
        Path to saved file
    """
    # Ensure figures directory exists
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Create full path
    filepath = figures_dir / filename
    
    # Save animation
    writer = PillowWriter(fps=fps)
    anim.save(filepath, writer=writer, dpi=dpi, **kwargs)
    
    return filepath


def create_contour_plot(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                       ax: plt.Axes,
                       levels: int = 30,
                       cmap: str = COLORMAPS['viridis'],
                       title: str = '',
                       xlabel: str = '',
                       ylabel: str = '',
                       colorbar: bool = True,
                       **kwargs) -> plt.Axes:
    """
    Create a contour plot.
    
    Args:
        x, y: Grid coordinates
        z: Function values
        ax: Axes to plot on
        levels: Number of contour levels
        cmap: Colormap
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        colorbar: Whether to add colorbar
        **kwargs: Additional arguments to contourf
    
    Returns:
        Axes object
    """
    contour = ax.contourf(x, y, z, levels=levels, cmap=cmap, **kwargs)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(contour, cax=cax)
    
    return ax


def create_3d_surface_plot(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                         ax: plt.Axes,
                         cmap: str = COLORMAPS['viridis'],
                         title: str = '',
                         xlabel: str = '',
                         ylabel: str = '',
                         zlabel: str = '',
                         **kwargs) -> plt.Axes:
    """
    Create a 3D surface plot.
    
    Args:
        x, y: Grid coordinates
        z: Function values
        ax: Axes to plot on
        cmap: Colormap
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        zlabel: Z-axis label
        **kwargs: Additional arguments to plot_surface
    
    Returns:
        Axes object
    """
    surface = ax.plot_surface(x, y, z, cmap=cmap, 
                            alpha=0.8, edgecolor='none', **kwargs)
    
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_zlabel(zlabel, fontsize=10)
    
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    return ax


def create_error_plot(x_values: np.ndarray,
                     y_values: np.ndarray,
                     ax: plt.Axes,
                     xlabel: str = '',
                     ylabel: str = '',
                     title: str = '',
                     log_scale: bool = False,
                     **kwargs) -> plt.Axes:
    """
    Create an error/loss plot.
    
    Args:
        x_values: X-axis values
        y_values: Y-axis values
        ax: Axes to plot on
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        log_scale: Whether to use log scale
        **kwargs: Additional arguments to plot
    
    Returns:
        Axes object
    """
    line, = ax.plot(x_values, y_values, **kwargs)
    
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    
    ax.grid(True, alpha=0.3)
    
    return ax


def create_animation_data(x_values: np.ndarray,
                          y_values: np.ndarray,
                          frames: int = 100,
                          method: str = 'linear') -> List[Dict[str, Any]]:
    """
    Generate data for animations.
    
    Args:
        x_values: X-axis values
        y_values: Y-axis values
        frames: Number of animation frames
        method: Interpolation method ('linear', 'exponential')
    
    Returns:
        List of dictionaries containing frame data
    """
    frame_data = []
    
    if method == 'linear':
        indices = np.linspace(0, len(x_values) - 1, frames, dtype=int)
    elif method == 'exponential':
        indices = np.logspace(0, np.log10(len(x_values) - 1), frames, dtype=int)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    for i, idx in enumerate(indices):
        frame_data.append({
            'frame': i,
            'index': idx,
            'x': x_values[idx],
            'y': y_values[idx],
            'progress': i / (frames - 1)
        })
    
    return frame_data


def create_custom_colormap(colors: List[str], name: str = 'custom') -> LinearSegmentedColormap:
    """
    Create a custom colormap.
    
    Args:
        colors: List of colors
        name: Colormap name
    
    Returns:
        LinearSegmentedColormap object
    """
    return LinearSegmentedColormap.from_list(name, colors, N=256)


def add_text_annotation(ax: plt.Axes,
                      text: str,
                      position: Tuple[float, float] = (0.02, 0.95),
                      transform: str = 'axes',
                      fontsize: int = 10,
                      bbox_color: str = 'wheat',
                      bbox_alpha: float = 0.5,
                      **kwargs) -> plt.Text:
    """
    Add text annotation to axes.
    
    Args:
        ax: Axes to add annotation to
        text: Text to add
        position: Position (x, y)
        transform: Coordinate system ('axes' or 'figure')
        fontsize: Font size
        bbox_color: Background color
        bbox_alpha: Background transparency
        **kwargs: Additional arguments to ax.text
    
    Returns:
        Text object
    """
    if transform == 'axes':
        text_obj = ax.text(position[0], position[1], text,
                          transform=ax.transAxes,
                          fontsize=fontsize,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', 
                                  facecolor=bbox_color, 
                                  alpha=bbox_alpha),
                          **kwargs)
    else:
        text_obj = ax.text(position[0], position[1], text,
                          fontsize=fontsize,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', 
                                  facecolor=bbox_color, 
                                  alpha=bbox_alpha),
                          **kwargs)
    
    return text_obj


def create_legend(ax: plt.Axes,
                 labels: List[str],
                 colors: List[str] = None,
                 loc: str = 'best',
                 fontsize: int = 10,
                 **kwargs) -> Any:
    """
    Create a legend with consistent styling.
    
    Args:
        ax: Axes to add legend to
        labels: List of labels
        colors: List of colors (optional)
        loc: Location of legend
        fontsize: Font size
        **kwargs: Additional arguments to ax.legend
    
    Returns:
        Legend object
    """
    handles = []
    for i, label in enumerate(labels):
        color = colors[i] if colors else None
        handles.append(plt.Line2D([0], [0], color=color, lw=2, label=label))
    
    legend = ax.legend(handles=handles,
                     loc=loc,
                     fontsize=fontsize,
                     framealpha=0.8,
                     **kwargs)
    
    return legend


def format_axis_labels(ax: plt.Axes,
                     xticks: List[float] = None,
                     xticklabels: List[str] = None,
                     yticks: List[float] = None,
                     yticklabels: List[str] = None,
                     rotation: int = 0) -> plt.Axes:
    """
    Format axis labels and ticks.
    
    Args:
        ax: Axes to format
        xticks: X-axis tick positions
        xticklabels: X-axis tick labels
        yticks: Y-axis tick positions
        yticklabels: Y-axis tick labels
        rotation: Label rotation
    
    Returns:
        Axes object
    """
    if xticks is not None and xticklabels is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=rotation)
    
    if yticks is not None and yticklabels is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, rotation=rotation)
    
    return ax