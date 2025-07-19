"""
Learning curves and comparison plots generation.

This module provides classes for generating learning curve plots and
comparison visualizations for the Multi-Agent Benchmark Suite.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Configure matplotlib and seaborn for better plots
plt.style.use('default')
sns.set_palette("husl")


class LearningCurvePlotter:
    """
    Generates learning curve plots for individual agents and environments.
    """
    
    def __init__(self, output_dir: Path, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the plotter.
        
        Args:
            output_dir: Directory to save plots
            figsize: Figure size (width, height)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
    
    def plot_learning_curves(self, 
                           learning_data: Dict[str, Tuple[List[float], List[int]]],
                           env_name: str,
                           metric_name: str = 'Episode Reward',
                           smooth_window: int = 10,
                           save_filename: Optional[str] = None) -> Path:
        """
        Plot learning curves for multiple agents on single environment.
        
        Args:
            learning_data: Dictionary mapping agent names to (values, timesteps)
            env_name: Environment name for plot title
            metric_name: Name of the metric being plotted
            smooth_window: Window size for smoothing curves
            save_filename: Optional custom filename
            
        Returns:
            Path to saved plot file
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot each agent's learning curve
        plotted_any = False
        for agent_name, (values, timesteps) in learning_data.items():
            if not values:  # Skip if no data
                continue
            
            # Smooth the curve
            smoothed_values = self._smooth_curve(values, smooth_window)
            
            # Plot the curve
            ax.plot(timesteps, smoothed_values, label=agent_name, linewidth=2)
            plotted_any = True
        
        # Customize plot
        ax.set_xlabel('Training Steps')
        ax.set_ylabel(metric_name)
        ax.set_title(f'Learning Curves - {env_name}')
        
        # Only add legend if we actually plotted something
        if plotted_any:
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No training data available', 
                   transform=ax.transAxes, ha='center', va='center')
        
        ax.grid(True, alpha=0.3)
        
        # Save plot
        if save_filename is None:
            save_filename = f"{env_name}_learning_curves.jpg"
        
        save_path = self.output_dir / save_filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return save_path
    
    def _smooth_curve(self, values: List[float], window_size: int) -> List[float]:
        """Smooth learning curve using moving average."""
        if len(values) < window_size:
            return values
        
        smoothed = []
        for i in range(len(values)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(values), i + window_size // 2 + 1)
            smoothed.append(np.mean(values[start_idx:end_idx]))
        
        return smoothed


class ComparisonPlotter:
    """
    Generates comparison plots between multiple agents across environments.
    """
    
    def __init__(self, output_dir: Path, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize the plotter.
        
        Args:
            output_dir: Directory to save plots
            figsize: Figure size (width, height)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
    
    def plot_comparison_matrix(self, 
                             results: Dict[str, Dict[str, Any]],
                             metric_key: str = 'mean_reward',
                             save_filename: str = 'comparison_matrix.jpg') -> Path:
        """
        Plot comparison matrix showing agent performance across environments.
        
        Args:
            results: Nested dict {env_name: {agent_name: result_dict}}
            metric_key: Key to extract from result_dict for comparison
            save_filename: Filename for saved plot
            
        Returns:
            Path to saved plot file
        """
        # Extract data for matrix
        env_names = list(results.keys())
        agent_names = []
        
        # Get all agent names
        for env_results in results.values():
            for agent_name in env_results.keys():
                if agent_name not in agent_names:
                    agent_names.append(agent_name)
        
        # Create performance matrix
        matrix = np.zeros((len(env_names), len(agent_names)))
        
        for i, env_name in enumerate(env_names):
            for j, agent_name in enumerate(agent_names):
                if agent_name in results[env_name]:
                    matrix[i, j] = results[env_name][agent_name].get(metric_key, 0)
                else:
                    matrix[i, j] = np.nan
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=self.figsize)
        
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(agent_names)))
        ax.set_yticks(range(len(env_names)))
        ax.set_xticklabels(agent_names, rotation=45, ha='right')
        ax.set_yticklabels(env_names)
        
        # Add text annotations
        for i in range(len(env_names)):
            for j in range(len(agent_names)):
                if not np.isnan(matrix[i, j]):
                    text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                                 ha='center', va='center', color='black', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric_key.replace('_', ' ').title())
        
        # Customize plot
        ax.set_title('Agent Performance Comparison Matrix')
        ax.set_xlabel('Agents')
        ax.set_ylabel('Environments')
        
        # Save plot
        save_path = self.output_dir / save_filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return save_path
    
    def plot_performance_bar_chart(self,
                                 results: Dict[str, Dict[str, Any]],
                                 metric_key: str = 'mean_reward',
                                 save_filename: str = 'performance_bars.jpg') -> Path:
        """
        Plot bar chart comparing agent performance across environments.
        
        Args:
            results: Nested dict {env_name: {agent_name: result_dict}}
            metric_key: Key to extract from result_dict for comparison
            save_filename: Filename for saved plot
            
        Returns:
            Path to saved plot file
        """
        # Prepare data
        env_names = list(results.keys())
        agent_names = []
        
        # Get all agent names
        for env_results in results.values():
            for agent_name in env_results.keys():
                if agent_name not in agent_names:
                    agent_names.append(agent_name)
        
        # Create subplots for each environment
        n_envs = len(env_names)
        n_cols = min(2, n_envs)
        n_rows = (n_envs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=self.figsize)
        if n_envs == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, env_name in enumerate(env_names):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Get performance data for this environment
            env_results = results[env_name]
            agents = []
            performances = []
            
            for agent_name in agent_names:
                if agent_name in env_results:
                    agents.append(agent_name)
                    performances.append(env_results[agent_name].get(metric_key, 0))
            
            # Create bar chart
            bars = ax.bar(agents, performances)
            
            # Customize subplot
            ax.set_title(f'{env_name}')
            ax.set_ylabel(metric_key.replace('_', ' ').title())
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, perf in zip(bars, performances):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{perf:.1f}', ha='center', va='bottom')
        
        # Hide unused subplots
        for i in range(n_envs, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.output_dir / save_filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return save_path


def create_publication_plots(results: Dict[str, Any], 
                           tensorboard_data: Dict[str, Tuple[List[float], List[int]]],
                           output_dir: Path) -> List[Path]:
    """
    Create publication-ready plots for the benchmark results.
    
    Args:
        results: Complete benchmark results
        tensorboard_data: TensorBoard learning curve data
        output_dir: Directory to save plots
        
    Returns:
        List of paths to created plot files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_plots = []
    
    # Initialize plotters
    learning_plotter = LearningCurvePlotter(output_dir)
    comparison_plotter = ComparisonPlotter(output_dir)
    
    # Generate learning curves for each environment
    for env_name, env_results in results.items():
        if env_name == 'summary':
            continue
        
        # Extract learning curves for this environment
        env_learning_data = {}
        for agent_name in env_results.get('individual_results', {}):
            tb_key = f"{env_name}_{agent_name}"
            if tb_key in tensorboard_data:
                tb_data = tensorboard_data[tb_key]
                # Handle both dictionary format and tuple format
                if isinstance(tb_data, dict):
                    rewards = tb_data.get('rewards', [])
                    timesteps = tb_data.get('timesteps', [])
                    env_learning_data[agent_name] = (rewards, timesteps)
                else:
                    # Assume it's already in tuple format
                    env_learning_data[agent_name] = tb_data
        
        if env_learning_data:
            plot_path = learning_plotter.plot_learning_curves(
                env_learning_data, env_name, save_filename=f"{env_name}_learning_curves.jpg"
            )
            saved_plots.append(plot_path)
    
    # Generate comparison matrix
    comparison_data = {}
    for env_name, env_results in results.items():
        if env_name == 'summary':
            continue
        comparison_data[env_name] = env_results.get('individual_results', {})
    
    if comparison_data:
        matrix_path = comparison_plotter.plot_comparison_matrix(comparison_data)
        saved_plots.append(matrix_path)
        
        bar_path = comparison_plotter.plot_performance_bar_chart(comparison_data)
        saved_plots.append(bar_path)
    
    return saved_plots


def save_plots(plots_data: Dict[str, Any], output_dir: Path) -> List[Path]:
    """
    Save plots in JPG format.
    
    Args:
        plots_data: Dictionary containing plot data
        output_dir: Directory to save plots
        
    Returns:
        List of paths to saved plot files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_plots = []
    
    # This is a convenience function that can be extended
    # Currently, individual plot creation is handled by the plotter classes
    
    return saved_plots