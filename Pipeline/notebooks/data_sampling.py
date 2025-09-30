#!/usr/bin/env python3
"""
Reusable Data Sampling Strategy for Pokemon Red RL Training Analysis

This module provides consistent, reproducible sampling across all analysis scripts.
Ensures fair comparison between variants by sampling uniformly across:
1. Training runs (from start to end or max 100M steps)
2. Multiple JSON files within a training run
3. Entries within individual JSON files

Usage:
    from data_sampling import TrainingSampler

    sampler = TrainingSampler(max_steps=100_000_000)
    v3_data = sampler.load_variant_data("../experiments/v3/20250620_070120/json_logs", "v3")
    v4_data = sampler.load_variant_data("../experiments/v4_production/20250813_125101/json_logs", "v4_prod")
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import gc
from tqdm import tqdm

class TrainingSampler:
    """Consistent sampling strategy for training data analysis"""

    def __init__(self, max_steps: int = 100_000_000, target_samples: int = 5000, verbose: bool = True):
        """
        Initialize sampler with constraints

        Args:
            max_steps: Maximum training steps to consider (default: 100M)
            target_samples: Target number of samples per variant
            verbose: Print detailed sampling information
        """
        self.max_steps = max_steps
        self.target_samples = target_samples
        self.verbose = verbose

    def _print(self, message: str):
        """Print if verbose mode enabled"""
        if self.verbose:
            print(message)

    def _sample_from_file(self, file_path: Path, target_samples: int) -> List[dict]:
        """
        Sample uniformly from a single JSON file

        Args:
            file_path: Path to JSON file
            target_samples: Number of samples to extract

        Returns:
            List of sampled entries
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Filter by max_steps constraint
        if len(data) > 0 and 'step' in data[0]:
            data = [entry for entry in data if entry['step'] <= self.max_steps]

        if len(data) == 0:
            return []

        # Uniform sampling within file
        if len(data) <= target_samples:
            return data
        else:
            # Systematic sampling for uniform distribution
            step_size = len(data) / target_samples
            indices = [int(i * step_size) for i in range(target_samples)]
            return [data[i] for i in indices]

    def _normalize_entry_format(self, entry: dict, variant_name: str) -> dict:
        """
        Normalize different JSON formats to consistent structure

        Args:
            entry: Raw JSON entry
            variant_name: Variant identifier

        Returns:
            Normalized entry dictionary
        """
        if 'rewards' in entry:  # v4 format (nested)
            return {
                'step': entry['step'],
                'total_steps': entry.get('total_steps', entry['step']),
                'total_reward': entry['rewards']['total'],
                'badge_reward': entry['rewards']['components']['badge'],
                'event_reward': entry['rewards']['components']['event'],
                'level_reward': entry['rewards']['components']['level'],
                'heal_reward': entry['rewards']['components']['heal'],
                'explore_reward': entry['rewards']['components']['explore'],
                'dead_penalty': entry['rewards']['components']['dead'],
                'stuck_penalty': entry['rewards']['components']['stuck'],
                'position_x': entry['position']['x'],
                'position_y': entry['position']['y'],
                'map_id': entry['position']['map'],
                'badges': entry['player_status']['badges'],
                'levels_sum': entry['player_status']['levels_sum'],
                'pokemon_count': entry['player_status']['pokemon_count'],
                'health': entry['player_status']['health'],
                'exploration': entry['statistics']['exploration_coords'],
                'deaths': entry['statistics']['deaths'],
                'last_action': entry['actions']['last_action'],
                'variant': variant_name
            }
        else:  # v3 format (flat)
            return {
                'step': entry['step'],
                'total_steps': entry['step'],  # Not available in v3
                'total_reward': entry.get('event', 0) + entry.get('badge', 0),  # Approximate
                'badge_reward': entry.get('badge', 0),
                'event_reward': entry.get('event', 0),
                'level_reward': 0,  # Not available in v3
                'heal_reward': entry.get('healr', 0),
                'explore_reward': 0,  # Not available in v3
                'dead_penalty': 0,  # Not directly available
                'stuck_penalty': 0,  # Not directly available
                'position_x': entry['x'],
                'position_y': entry['y'],
                'map_id': entry['map'],
                'badges': entry.get('badge', 0),
                'levels_sum': entry.get('levels_sum', 0),
                'pokemon_count': entry.get('pcount', 0),
                'health': entry.get('hp', 0),
                'exploration': entry.get('coord_count', 0),
                'deaths': entry.get('deaths', 0),
                'last_action': entry.get('last_action', 0),
                'variant': variant_name
            }

    def load_variant_data(self, experiment_path: str, variant_name: str,
                         max_files: Optional[int] = None) -> pd.DataFrame:
        """
        Load and sample data from a training variant

        Args:
            experiment_path: Path to experiment's json_logs directory
            variant_name: Name identifier for this variant
            max_files: Maximum number of files to process (None = all files)

        Returns:
            DataFrame with sampled and normalized training data
        """
        experiment_path = Path(experiment_path)
        self._print(f"Loading {variant_name} from {experiment_path}")

        # Get all JSON files, sorted by filename for consistent ordering
        json_files = sorted(list(experiment_path.glob("*.json")))
        if max_files is not None:
            json_files = json_files[:max_files]

        self._print(f"  Found {len(json_files)} files to process")

        if len(json_files) == 0:
            self._print(f"  No JSON files found!")
            return pd.DataFrame()

        # Calculate samples per file for uniform distribution
        samples_per_file = max(1, self.target_samples // len(json_files))

        all_data = []
        total_files_processed = 0

        for i, json_file in enumerate(tqdm(json_files, desc=f"Processing {variant_name}")):
            try:
                file_size_mb = json_file.stat().st_size / 1024 / 1024
                self._print(f"  File {i+1}/{len(json_files)}: {json_file.name} ({file_size_mb:.1f} MB)")

                # Sample from this file
                sampled_entries = self._sample_from_file(json_file, samples_per_file)
                self._print(f"    Sampled {len(sampled_entries)} entries")

                # Normalize format and add to dataset
                for entry in sampled_entries:
                    if entry['step'] <= self.max_steps:  # Double-check step constraint
                        normalized = self._normalize_entry_format(entry, variant_name)
                        all_data.append(normalized)

                total_files_processed += 1
                gc.collect()  # Memory cleanup

            except Exception as e:
                self._print(f"    Error processing {json_file.name}: {e}")
                continue

        # Create DataFrame
        df = pd.DataFrame(all_data)

        if len(df) > 0:
            # Ensure step constraint
            df = df[df['step'] <= self.max_steps]

            # Sort by step for consistent ordering
            df = df.sort_values('step').reset_index(drop=True)

            self._print(f"  Total loaded for {variant_name}: {len(df)} entries")
            self._print(f"    Step range: {df['step'].min()} - {df['step'].max()}")
            self._print(f"    Reward range: {df['total_reward'].min():.3f} - {df['total_reward'].max():.3f}")
            self._print(f"    Files processed: {total_files_processed}/{len(json_files)}")
        else:
            self._print(f"  No valid data loaded for {variant_name}")

        return df

    def normalize_step_ranges(self, *dataframes: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
        """
        Ensure all variants have the same step range for fair comparison

        Args:
            *dataframes: Variable number of DataFrames to normalize

        Returns:
            Tuple of normalized DataFrames with consistent step ranges
        """
        if not dataframes or all(len(df) == 0 for df in dataframes):
            return dataframes

        # Find the minimum maximum step across all variants
        max_steps = []
        for i, df in enumerate(dataframes):
            if len(df) > 0:
                max_step = df['step'].max()
                max_steps.append(max_step)
                self._print(f"DataFrame {i} original max step: {max_step}")

        if not max_steps:
            return dataframes

        # Use the minimum of the maximum steps
        common_max_step = min(max_steps)
        self._print(f"Using common max step: {common_max_step}")

        # Filter all datasets
        filtered_dfs = []
        for i, df in enumerate(dataframes):
            if len(df) > 0:
                filtered_df = df[df['step'] <= common_max_step].copy()
                self._print(f"DataFrame {i} filtered: {len(filtered_df)} entries, "
                          f"step range: {filtered_df['step'].min()} - {filtered_df['step'].max()}")
                filtered_dfs.append(filtered_df)
            else:
                filtered_dfs.append(df)

        return tuple(filtered_dfs)

    def get_summary_stats(self, df: pd.DataFrame, variant_name: str) -> Dict:
        """
        Calculate summary statistics for a variant

        Args:
            df: DataFrame with variant data
            variant_name: Name of the variant

        Returns:
            Dictionary with summary statistics
        """
        if len(df) == 0:
            return {'variant': variant_name, 'samples': 0}

        return {
            'variant': variant_name,
            'samples': len(df),
            'step_range': (df['step'].min(), df['step'].max()),
            'mean_total_reward': df['total_reward'].mean(),
            'std_total_reward': df['total_reward'].std(),
            'max_badges': df['badges'].max(),
            'max_exploration': df['exploration'].max(),
            'mean_exploration': df['exploration'].mean(),
            'mean_deaths': df['deaths'].mean(),
            'mean_health': df['health'].mean()
        }

# Convenience function for common usage pattern
def load_variants_for_comparison(variant_configs: List[Tuple[str, str]],
                               max_steps: int = 100_000_000,
                               target_samples: int = 5000,
                               max_files: Optional[int] = None,
                               normalize_steps: bool = True) -> Tuple[List[pd.DataFrame], TrainingSampler]:
    """
    Load multiple variants with consistent sampling for comparison

    Args:
        variant_configs: List of (experiment_path, variant_name) tuples
        max_steps: Maximum steps to consider
        target_samples: Target samples per variant
        max_files: Maximum files per variant
        normalize_steps: Whether to normalize step ranges across variants

    Returns:
        Tuple of (list of DataFrames, sampler instance)
    """
    sampler = TrainingSampler(max_steps=max_steps, target_samples=target_samples)

    dataframes = []
    for experiment_path, variant_name in variant_configs:
        df = sampler.load_variant_data(experiment_path, variant_name, max_files)
        dataframes.append(df)

    if normalize_steps and len(dataframes) > 1:
        print("\n=== Normalizing Step Ranges for Fair Comparison ===")
        dataframes = list(sampler.normalize_step_ranges(*dataframes))

    return dataframes, sampler