"""
TensorBoard data extraction utilities.

This module provides functions to extract learning curves and metrics
from TensorBoard log files for comparison and visualization.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logger.warning("TensorBoard not available. Install with: pip install tensorboard")


def extract_tensorboard_data(log_dir: Union[str, Path], 
                           metric_name: str = 'rollout/ep_rew_mean') -> Tuple[List[float], List[int]]:
    """
    Extract learning curve data from TensorBoard logs.
    
    Args:
        log_dir: Directory containing TensorBoard log files
        metric_name: Name of the metric to extract (default: episode reward mean)
        
    Returns:
        Tuple of (values, timesteps) lists
        
    Raises:
        ImportError: If TensorBoard is not installed
        RuntimeError: If no valid log files found
    """
    if not TENSORBOARD_AVAILABLE:
        raise ImportError("TensorBoard not available. Install with: pip install tensorboard")
    
    log_dir = Path(log_dir)
    
    if not log_dir.exists():
        raise RuntimeError(f"Log directory does not exist: {log_dir}")
    
    # Find TensorBoard log files
    log_files = list(log_dir.rglob("events.out.tfevents.*"))
    if not log_files:
        raise RuntimeError(f"No TensorBoard log files found in {log_dir}")
    
    # Load event accumulator
    event_accumulator = EventAccumulator(str(log_dir))
    event_accumulator.Reload()
    
    # Check if metric exists
    available_scalars = event_accumulator.Tags()['scalars']
    if metric_name not in available_scalars:
        logger.warning(f"Metric '{metric_name}' not found. Available metrics: {available_scalars}")
        return [], []
    
    # Extract scalar events
    scalar_events = event_accumulator.Scalars(metric_name)
    
    # Extract values and timesteps
    timesteps = [event.step for event in scalar_events]
    values = [event.value for event in scalar_events]
    
    logger.info(f"✅ Extracted {len(values)} data points for '{metric_name}' from {log_dir}")
    
    return values, timesteps


def extract_multiple_metrics(log_dir: Union[str, Path], 
                           metrics: List[str]) -> Dict[str, Tuple[List[float], List[int]]]:
    """
    Extract multiple metrics from TensorBoard logs.
    
    Args:
        log_dir: Directory containing TensorBoard log files
        metrics: List of metric names to extract
        
    Returns:
        Dictionary mapping metric names to (values, timesteps) tuples
    """
    if not TENSORBOARD_AVAILABLE:
        raise ImportError("TensorBoard not available. Install with: pip install tensorboard")
    
    log_dir = Path(log_dir)
    results = {}
    
    event_accumulator = EventAccumulator(str(log_dir))
    event_accumulator.Reload()
    
    available_scalars = event_accumulator.Tags()['scalars']
    
    for metric in metrics:
        if metric in available_scalars:
            scalar_events = event_accumulator.Scalars(metric)
            timesteps = [event.step for event in scalar_events]
            values = [event.value for event in scalar_events]
            results[metric] = (values, timesteps)
        else:
            logger.warning(f"Metric '{metric}' not found in {log_dir}")
            results[metric] = ([], [])
    
    return results


def find_tensorboard_logs(base_dir: Union[str, Path], 
                         agent_name: Optional[str] = None,
                         env_name: Optional[str] = None) -> List[Path]:
    """
    Find TensorBoard log directories.
    
    Args:
        base_dir: Base directory to search in
        agent_name: Optional agent name to filter by
        env_name: Optional environment name to filter by
        
    Returns:
        List of paths to TensorBoard log directories
    """
    base_dir = Path(base_dir)
    
    if not base_dir.exists():
        return []
    
    # Find all directories containing TensorBoard files
    log_dirs = []
    
    for path in base_dir.rglob("*"):
        if path.is_dir():
            # Check if directory contains TensorBoard files
            if list(path.glob("events.out.tfevents.*")):
                # Apply filters if provided
                if agent_name and agent_name not in path.name:
                    continue
                if env_name and env_name not in path.name:
                    continue
                
                log_dirs.append(path)
    
    return sorted(log_dirs)


def get_available_metrics(log_dir: Union[str, Path]) -> List[str]:
    """
    Get list of available metrics in TensorBoard logs.
    
    Args:
        log_dir: Directory containing TensorBoard log files
        
    Returns:
        List of available metric names
    """
    if not TENSORBOARD_AVAILABLE:
        return []
    
    log_dir = Path(log_dir)
    
    if not log_dir.exists():
        return []
    
    try:
        event_accumulator = EventAccumulator(str(log_dir))
        event_accumulator.Reload()
        return event_accumulator.Tags()['scalars']
    except Exception as e:
        logger.warning(f"Could not read metrics from {log_dir}: {e}")
        return []


def smooth_curve(values: List[float], window_size: int = 10) -> List[float]:
    """
    Smooth learning curve using moving average.
    
    Args:
        values: List of values to smooth
        window_size: Size of the moving average window
        
    Returns:
        Smoothed values
    """
    if len(values) < window_size:
        return values
    
    smoothed = []
    for i in range(len(values)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(values), i + window_size // 2 + 1)
        smoothed.append(np.mean(values[start_idx:end_idx]))
    
    return smoothed


def extract_training_metrics(log_dir: Union[str, Path]) -> Dict[str, Tuple[List[float], List[int]]]:
    """
    Extract standard training metrics for agent comparison.
    
    Args:
        log_dir: Directory containing TensorBoard log files
        
    Returns:
        Dictionary with common training metrics
    """
    standard_metrics = [
        'rollout/ep_rew_mean',      # Episode reward mean
        'rollout/ep_len_mean',      # Episode length mean
        'train/value_loss',         # Value function loss
        'train/policy_loss',        # Policy loss
        'train/entropy_loss',       # Entropy loss
        'train/learning_rate',      # Learning rate
        'train/explained_variance'  # Explained variance
    ]
    
    return extract_multiple_metrics(log_dir, standard_metrics)


def compare_learning_curves(log_dirs: Dict[str, Path], 
                          metric_name: str = 'rollout/ep_rew_mean') -> Dict[str, Tuple[List[float], List[int]]]:
    """
    Extract learning curves from multiple log directories for comparison.
    
    Args:
        log_dirs: Dictionary mapping agent names to log directories
        metric_name: Metric to extract for comparison
        
    Returns:
        Dictionary mapping agent names to (values, timesteps) tuples
    """
    results = {}
    
    for agent_name, log_dir in log_dirs.items():
        try:
            values, timesteps = extract_tensorboard_data(log_dir, metric_name)
            results[agent_name] = (values, timesteps)
        except Exception as e:
            logger.warning(f"Could not extract data for {agent_name}: {e}")
            results[agent_name] = ([], [])
    
    return results


def validate_tensorboard_logs(log_dir: Union[str, Path]) -> bool:
    """
    Validate that TensorBoard logs exist and are readable.
    
    Args:
        log_dir: Directory to validate
        
    Returns:
        True if valid TensorBoard logs found, False otherwise
    """
    if not TENSORBOARD_AVAILABLE:
        return False
    
    log_dir = Path(log_dir)
    
    if not log_dir.exists():
        return False
    
    # Check for TensorBoard files
    log_files = list(log_dir.rglob("events.out.tfevents.*"))
    if not log_files:
        return False
    
    # Try to read one file
    try:
        event_accumulator = EventAccumulator(str(log_dir))
        event_accumulator.Reload()
        return len(event_accumulator.Tags()['scalars']) > 0
    except Exception:
        return False