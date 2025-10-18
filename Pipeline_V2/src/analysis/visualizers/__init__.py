"""
Visualization modules for Pokemon Red RL training analysis

This package provides visualization functions for reward plots, heatmaps,
and multi-variant comparisons. All functions are designed to work with
data loaded via TrainingSampler.

Modules:
    reward_plots: Learning curves and reward component visualization
    heatmap_plots: Position density heatmaps
"""

from .reward_plots import plot_reward_curves
from .heatmap_plots import (
    prepare_heatmap_data,
    plot_all_maps,
    plot_position_heatmap,
)

__all__ = [
    'plot_reward_curves',
    'prepare_heatmap_data',
    'plot_position_heatmap',
    'plot_all_maps',
]
