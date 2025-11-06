"""
Visualization helpers for sampling training logs and creating plots.

This package originated from the exploratory notebooks but is now part of the
library so the tooling can be invoked via ``python -m poke_pipeline.visualization.sampling_cli``.
"""

from .data_sampling import (
    StreamingDataSampler,
    TrainingSampler,
    flatten_dict,
    load_variants_for_comparison,
)
from .heatmap_plots import plot_all_maps, plot_position_heatmap
from .map_utils import GLOBAL_MAP_SHAPE, MAP_DATA, local_to_global
from .plot_helpers import (
    OUTPUT_FORMATS,
    PLOT_CONFIG,
    REWARD_COLORS,
    create_plot_title,
    get_reward_columns,
    save_plot,
    setup_plot_style,
    smooth_series,
)
from .reward_plots import plot_rewards_over_time
__all__ = [
    "StreamingDataSampler",
    "TrainingSampler",
    "flatten_dict",
    "load_variants_for_comparison",
    "plot_all_maps",
    "plot_position_heatmap",
    "plot_rewards_over_time",
    "MAP_DATA",
    "GLOBAL_MAP_SHAPE",
    "local_to_global",
    "PLOT_CONFIG",
    "OUTPUT_FORMATS",
    "REWARD_COLORS",
    "create_plot_title",
    "get_reward_columns",
    "save_plot",
    "setup_plot_style",
    "smooth_series",
]
