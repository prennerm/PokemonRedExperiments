#!/usr/bin/env python3
"""
Enhanced Heatmap Visualization with Pokemon Red Map Overlay

Creates position heatmaps overlaid on the actual Pokemon Red map background.
Uses global coordinate transformation to accurately place visit data.

Requirements:
- conda environment: poke_viz_extended
- Activate with: conda activate poke_viz_extended

Dependencies: numpy, matplotlib, PIL, pandas, tqdm, mediapy, einops
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Add project paths
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from utils.map_helpers import get_map_name, load_pokemon_map_background
from utils.data_loader import find_latest_experiment_dir, find_json_log_files
from src.poke_pipeline.global_map import local_to_global, GLOBAL_MAP_SHAPE


class EnhancedHeatmapProcessor:
    """
    Processes JSON logs to create enhanced heatmaps with Pokemon Red map overlay
    """

    def __init__(self, experiment_dir: Path, verbose: bool = True):
        self.experiment_dir = experiment_dir
        self.verbose = verbose
        self.position_data = defaultdict(int)
        self.map_visit_counts = defaultdict(int)
        self.pixel_position_data = defaultdict(int)

    def process_json_logs(self, max_samples: Optional[int] = None, max_files: Optional[int] = None) -> Dict:
        """Process all JSON log files and aggregate position data"""

        json_files = find_json_log_files(self.experiment_dir)

        if not json_files:
            raise FileNotFoundError(f"No JSON log files found in {self.experiment_dir}")

        # Sample files for better time distribution
        if max_files and len(json_files) > max_files:
            # Take every Nth file to get even distribution across training time
            step = max(1, len(json_files) // max_files)
            json_files = json_files[::step]
            if self.verbose:
                print(f"Sampling every {step}th file: {len(json_files)} files selected")

        total_entries = 0
        processed_samples = 0

        if self.verbose:
            print(f"Processing {len(json_files)} JSON log files...")

        for json_file in tqdm(json_files, desc="Processing JSON files"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                for entry in data:
                    if max_samples and processed_samples >= max_samples:
                        break

                    # Extract position data
                    if 'position' in entry:
                        pos = entry['position']
                        x, y, map_id = pos.get('x', 0), pos.get('y', 0), pos.get('map', 0)

                        # Store raw position data
                        position_key = (x, y, map_id)
                        self.position_data[position_key] += 1
                        self.map_visit_counts[map_id] += 1

                        # Convert to global coordinates using the project's coordinate system
                        try:
                            # Use the existing global_map.py coordinate transformation
                            global_y, global_x = local_to_global(y, x, map_id)  # Note: r=y, c=x
                            global_key = (global_x, global_y, map_id)  # Include map_id for multi-panel layout
                            self.pixel_position_data[global_key] += 1
                        except Exception as e:
                            if self.verbose:
                                print(f"Warning: Global coordinate conversion failed for ({x}, {y}, {map_id}): {e}")

                        total_entries += 1
                        processed_samples += 1

                    if max_samples and processed_samples >= max_samples:
                        break

            except Exception as e:
                if self.verbose:
                    print(f"Warning: Error processing {json_file}: {e}")
                continue

        if self.verbose:
            print(f"Processed {total_entries} position entries")
            print(f"Found {len(self.position_data)} unique positions")
            print(f"Visited {len(self.map_visit_counts)} different maps")

        return {
            'position_data': dict(self.position_data),
            'pixel_position_data': dict(self.pixel_position_data),
            'map_visit_counts': dict(self.map_visit_counts),
            'total_entries': total_entries
        }

    def create_map_summary(self) -> str:
        """Create a summary of visited maps"""
        if not self.map_visit_counts:
            return "No map data available"

        summary = ["Map Visit Summary:"]

        # Sort by visit count (descending)
        sorted_maps = sorted(self.map_visit_counts.items(), key=lambda x: x[1], reverse=True)

        for map_id, count in sorted_maps[:10]:  # Top 10 most visited maps
            map_name = get_map_name(map_id)
            percentage = (count / sum(self.map_visit_counts.values())) * 100
            summary.append(f"  {map_name} (ID: {map_id}): {count:,} visits ({percentage:.1f}%)")

        if len(sorted_maps) > 10:
            remaining = len(sorted_maps) - 10
            summary.append(f"  ... and {remaining} other maps")

        return "\n".join(summary)


def plot_enhanced_heatmap(stats: Dict, experiment_name: str, output_dir: Path,
                         output_format: str = 'png', verbose: bool = True):
    """Create enhanced heatmap with Pokemon Red map overlay - multi-panel layout"""

    if verbose:
        print("Creating enhanced heatmap with map overlay...")

    # Load Pokemon Red map background
    try:
        background_map = load_pokemon_map_background()
        if verbose:
            print(f"Map background loaded: {background_map.shape}")
        use_background = True
    except Exception as e:
        if verbose:
            print(f"Warning: Could not load map background: {e}")
            print("Creating heatmap using global coordinate system only...")
        use_background = False

    # Get global position data with map information
    global_data = stats.get('pixel_position_data', {})
    map_visit_counts = stats.get('map_visit_counts', {})

    if not global_data:
        print("Warning: No global position data available for heatmap")
        return

    # Group data by map for multi-panel layout
    map_data = {}
    for coord_key, count in global_data.items():
        if len(coord_key) == 3:  # (x, y, map_id)
            x, y, map_id = coord_key
            if map_id not in map_data:
                map_data[map_id] = {'coords': [], 'counts': []}
            map_data[map_id]['coords'].append((x, y))
            map_data[map_id]['counts'].append(count)

    if not map_data:
        print("Warning: No map-specific position data available")
        return

    # Sort maps by visit frequency (most visited first)
    sorted_maps = sorted(map_data.keys(), key=lambda m: sum(map_data[m]['counts']), reverse=True)

    if verbose:
        print(f"Creating multi-panel heatmap for {len(sorted_maps)} maps")
        for map_id in sorted_maps:
            map_name = get_map_name(map_id)
            total_visits = sum(map_data[map_id]['counts'])
            print(f"  {map_name}: {total_visits} visits")

    # Create subplot layout
    n_maps = len(sorted_maps)
    cols = min(3, n_maps)  # Max 3 columns
    rows = (n_maps + cols - 1) // cols  # Ceiling division

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n_maps == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if n_maps > 1 else [axes]
    else:
        axes = axes.flatten()

    # Hide unused subplots
    for i in range(n_maps, len(axes)):
        axes[i].set_visible(False)

    # Plot each map in its own panel
    for i, map_id in enumerate(sorted_maps):
        ax = axes[i]
        coords = map_data[map_id]['coords']
        counts = map_data[map_id]['counts']

        x_coords = [c[0] for c in coords]
        y_coords = [c[1] for c in coords]

        if verbose:
            print(f"Panel {i+1}: {get_map_name(map_id)} - {len(coords)} positions")
            print(f"  X range: {min(x_coords)} - {max(x_coords)}")
            print(f"  Y range: {min(y_coords)} - {max(y_coords)}")

        # Calculate map-specific bounds with padding
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        x_pad = max(5, (x_max - x_min) * 0.1)
        y_pad = max(5, (y_max - y_min) * 0.1)

        # Set bounds for this specific map area
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_max + y_pad, y_min - y_pad)  # Flip Y axis

        if use_background:
            # Crop background to this map's region
            # Scale background coordinates to match global coordinate system
            bg_x_min = int((x_min - x_pad) * background_map.shape[1] / GLOBAL_MAP_SHAPE[1])
            bg_x_max = int((x_max + x_pad) * background_map.shape[1] / GLOBAL_MAP_SHAPE[1])
            bg_y_min = int((y_min - y_pad) * background_map.shape[0] / GLOBAL_MAP_SHAPE[0])
            bg_y_max = int((y_max + y_pad) * background_map.shape[0] / GLOBAL_MAP_SHAPE[0])

            # Ensure bounds are within image
            bg_x_min = max(0, bg_x_min)
            bg_x_max = min(background_map.shape[1], bg_x_max)
            bg_y_min = max(0, bg_y_min)
            bg_y_max = min(background_map.shape[0], bg_y_max)

            if bg_x_max > bg_x_min and bg_y_max > bg_y_min:
                cropped_bg = background_map[bg_y_min:bg_y_max, bg_x_min:bg_x_max]
                ax.imshow(cropped_bg, extent=[x_min - x_pad, x_max + x_pad, y_max + y_pad, y_min - y_pad],
                         aspect='equal', alpha=0.7)

        # Use logarithmic scaling for better visualization
        counts_log = np.log1p(counts)

        # Scale point sizes appropriately for the zoomed view
        point_sizes = np.clip(counts_log * 15, 8, 100)  # Smaller points for detailed view

        # Create scatter plot
        scatter = ax.scatter(
            x_coords, y_coords,
            c=counts_log,
            s=point_sizes,
            cmap='plasma',
            alpha=0.9,
            edgecolors='white',
            linewidth=1
        )

        # Set title for this panel
        map_name = get_map_name(map_id)
        total_visits = sum(counts)
        ax.set_title(f"{map_name}\\n{len(coords)} positions, {total_visits:,} visits",
                    fontsize=12, pad=10)

        # Clean up axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    # Add overall title
    total_visits = sum(sum(map_data[m]['counts']) for m in map_data)
    total_positions = sum(len(map_data[m]['coords']) for m in map_data)
    fig.suptitle(f"Pokemon Red Agent Movement Heatmap (Multi-Panel)\\n{experiment_name}\\n"
                f"{total_positions:,} positions, {total_visits:,} total visits",
                fontsize=16, y=0.98)

    # Add a shared colorbar
    if n_maps > 0:
        # Use the last scatter plot for colorbar reference
        cbar = fig.colorbar(scatter, ax=axes[:n_maps], shrink=0.6, pad=0.02)
        cbar.set_label('Visit Frequency (log scale)', rotation=270, labelpad=20)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # Make room for suptitle

    # Save the plot
    output_file = output_dir / f"{experiment_name}_enhanced_heatmap_multipanel.{output_format}"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')

    if verbose:
        print(f"Enhanced multi-panel heatmap saved: {output_file}")

    plt.close(fig)

    return output_file


def plot_map_visit_distribution(stats: Dict, experiment_name: str, output_dir: Path,
                               output_format: str = 'png', verbose: bool = True):
    """Create bar chart showing visit distribution across maps"""

    map_counts = stats.get('map_visit_counts', {})

    if not map_counts:
        print("Warning: No map visit data available")
        return

    # Prepare data for plotting
    map_ids = list(map_counts.keys())
    counts = list(map_counts.values())
    map_names = [get_map_name(mid) for mid in map_ids]

    # Sort by visit count
    sorted_data = sorted(zip(map_names, counts, map_ids), key=lambda x: x[1], reverse=True)
    sorted_names, sorted_counts, sorted_ids = zip(*sorted_data)

    # Take top 15 most visited maps
    top_n = min(15, len(sorted_names))
    plot_names = sorted_names[:top_n]
    plot_counts = sorted_counts[:top_n]
    plot_ids = sorted_ids[:top_n]

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    bars = ax.bar(range(len(plot_names)), plot_counts, color='steelblue', alpha=0.7)

    # Customize the plot
    ax.set_xlabel('Map Location', fontsize=12)
    ax.set_ylabel('Visit Count', fontsize=12)
    ax.set_title(f'Map Visit Distribution\\n{experiment_name}', fontsize=14, pad=20)

    # Set x-axis labels
    ax.set_xticks(range(len(plot_names)))
    ax.set_xticklabels([f"{name}\\n(ID: {mid})" for name, mid in zip(plot_names, plot_ids)],
                       rotation=45, ha='right', fontsize=10)

    # Add value labels on bars
    for bar, count in zip(bars, plot_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(plot_counts)*0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=9)

    # Add grid for readability
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save the plot
    output_file = output_dir / f"{experiment_name}_map_distribution.{output_format}"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')

    if verbose:
        print(f"Map distribution chart saved: {output_file}")

    plt.close(fig)

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Enhanced Pokemon Red heatmap visualization")
    parser.add_argument("--experiment-dir", type=Path,
                       help="Path to experiment directory (default: auto-detect latest)")
    parser.add_argument("--variant", choices=["v1", "v2", "v3", "v4"], default="v4",
                       help="Training variant to visualize")
    parser.add_argument("--output-dir", type=Path,
                       help="Output directory for plots (default: experiment_dir/plots)")
    parser.add_argument("--output-format", choices=["png", "pdf", "svg"], default="png",
                       help="Output format for plots")
    parser.add_argument("--max-samples", type=int,
                       help="Maximum number of samples to process (for testing)")
    parser.add_argument("--max-files", type=int, default=50,
                       help="Maximum number of JSON files to process (default: 50)")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Verbose output")

    args = parser.parse_args()

    # Find experiment directory
    if args.experiment_dir is None:
        args.experiment_dir = find_latest_experiment_dir(args.variant)

    if not args.experiment_dir or not args.experiment_dir.exists():
        print(f"Error: Experiment directory not found: {args.experiment_dir}")
        return 1

    # Set default output directory to experiment_dir/plots if not specified
    if args.output_dir is None:
        args.output_dir = args.experiment_dir / "plots"

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process the data
    processor = EnhancedHeatmapProcessor(args.experiment_dir, args.verbose)
    stats = processor.process_json_logs(args.max_samples, args.max_files)

    if args.verbose:
        print(processor.create_map_summary())

    # Generate experiment name
    experiment_name = f"{args.variant}_{args.experiment_dir.name}"

    # Create visualizations
    try:
        plot_enhanced_heatmap(stats, experiment_name, args.output_dir,
                            args.output_format, args.verbose)
        plot_map_visit_distribution(stats, experiment_name, args.output_dir,
                                  args.output_format, args.verbose)

        if args.verbose:
            print("Enhanced heatmap visualization completed successfully!")

    except Exception as e:
        print(f"Error creating visualizations: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())