"""
Path management and session directory utilities.

Simple path utilities for the Multi-Agent Benchmark Suite.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional, Union


def get_project_root() -> Path:
    """
    Find the project root directory.
    
    Looks for setup.py or environment.yml in parent directories.
    """
    current_path = Path(__file__).resolve()
    
    for parent in current_path.parents:
        if (parent / 'setup.py').exists() or (parent / 'environment.yml').exists():
            return parent
    
    # Fallback: assume we're in src/poke_pipeline/sanity_check/...
    if 'src' in current_path.parts:
        src_index = current_path.parts.index('src')
        return Path(*current_path.parts[:src_index])
    
    raise RuntimeError("Could not find project root directory")


def create_session_directory(base_dir: Optional[Union[str, Path]] = None, 
                            session_name: Optional[str] = None) -> Path:
    """
    Create a timestamped session directory in results/ with standard subdirectories.
    
    Args:
        base_dir: Base directory to create session in (defaults to project_root/results)
        session_name: Optional custom name (defaults to timestamp)
        
    Returns:
        Path to the created session directory (base_dir/session_name/)
    """
    # Determine base directory
    if base_dir is None:
        # Use sanity_check/results instead of project root results
        current_path = Path(__file__).resolve()
        sanity_check_root = current_path.parent.parent.parent  # Go up from utils/paths.py to sanity_check/
        results_dir = sanity_check_root / 'results'
    else:
        results_dir = Path(base_dir)
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if session_name is None:
        session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    session_dir = results_dir / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Create standard subdirectories
    for subdir in ['logs', 'models', 'tensorboard', 'plots', 'reports']:
        (session_dir / subdir).mkdir(exist_ok=True)
    
    return session_dir