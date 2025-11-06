#!/usr/bin/env python3
"""
Scalable Data Sampling Strategy for Pokemon Red RL Training Analysis.

Ported from the exploratory notebooks into the library so the tooling can be
used programmatically (or via the sampling CLI).
"""

import gc
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from tqdm import tqdm


def flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
    """Flatten a nested dictionary structure."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class StreamingDataSampler:
    """
    Memory-efficient streaming sampler based on the visualize_training notebook
    architecture. Uses reservoir sampling to keep memory usage bounded.
    """

    def __init__(self, max_steps: int = 100_000_000, target_samples: int = 5000, verbose: bool = True):
        self.max_steps = max_steps
        self.target_samples = target_samples
        self.verbose = verbose

        self.reservoir_samples: List[Dict] = []
        self.total_entries_seen = 0
        self.current_variant: Optional[str] = None
        self.step_range = [float("inf"), float("-inf")]
        self.map_visit_counts: Dict[int, Dict[Tuple[int, int], int]] = self._make_heatmap_store()
        self.latest_map_counts: Dict[int, Dict[Tuple[int, int], int]] = {}
        self.latest_sampling_meta: Dict[str, object] = {}

    def _print(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _reservoir_sample(self, entry: Dict) -> None:
        self.total_entries_seen += 1

        if len(self.reservoir_samples) < self.target_samples:
            self.reservoir_samples.append(entry)
        else:
            j = random.randint(0, self.total_entries_seen - 1)
            if j < self.target_samples:
                self.reservoir_samples[j] = entry

    @staticmethod
    def _make_heatmap_store() -> Dict[int, Dict[Tuple[int, int], int]]:
        return defaultdict(lambda: defaultdict(int))

    def _reset_heatmap_counts(self) -> None:
        self.map_visit_counts = self._make_heatmap_store()

    def _count_entries_in_file(self, file_path: Path) -> int:
        suffix = file_path.suffix.lower()
        try:
            if suffix == ".csv":
                with file_path.open("r", encoding="utf-8") as fh:
                    # subtract header if present
                    return max(sum(1 for _ in fh) - 1, 0)
            if suffix in {".json", ".jsonl"}:
                with file_path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    return len(data)
                if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
                    return len(data["data"])
        except Exception as exc:
            self._print(f"[count_entries] Failed to inspect {file_path.name}: {exc}")
        return 0

    @staticmethod
    def _evenly_spaced_indices(total: int, target: int) -> List[int]:
        if target <= 0 or total <= 0:
            return []
        if target >= total:
            return list(range(total))
        step = (total - 1) / (target - 1) if target > 1 else total
        indices: List[int] = []
        last_idx = -1
        for i in range(target):
            candidate = int(round(i * step))
            candidate = min(max(candidate, last_idx + 1), total - 1)
            indices.append(candidate)
            last_idx = candidate
        # ensure uniqueness and correct length
        seen = set()
        unique = []
        for idx in indices:
            if idx not in seen:
                unique.append(idx)
                seen.add(idx)
        cur = 0
        while len(unique) < target and cur < total:
            if cur not in seen:
                unique.append(cur)
                seen.add(cur)
            cur += 1
        unique.sort()
        return unique

    def _plan_files_for_max_points(
        self,
        files: List[Path],
        max_data_points: int,
    ) -> List[Tuple[Path, Optional[Set[int]]]]:
        if not files or max_data_points <= 0:
            return []

        approx_entries = self._count_entries_in_file(files[0]) or 1
        file_count = len(files)

        capacity_based = max(1, int(np.ceil(max_data_points / approx_entries)))
        density_based = max(1, int(np.ceil(np.sqrt(max_data_points))))
        num_files = min(file_count, max_data_points, max(capacity_based, density_based))

        selected_indices = sorted(set(np.linspace(0, file_count - 1, num=num_files, dtype=int)))
        if not selected_indices:
            selected_indices = [0]
        if len(selected_indices) > max_data_points:
            selected_indices = selected_indices[:max_data_points]

        selected_count = len(selected_indices)
        target_total = min(max_data_points, approx_entries * selected_count)

        if target_total >= approx_entries * selected_count:
            return [(files[idx], None) for idx in selected_indices]

        base_alloc = target_total // selected_count
        remainder = target_total % selected_count

        plan: List[Tuple[Path, Optional[Set[int]]]] = []
        for position, file_idx in enumerate(selected_indices):
            alloc = base_alloc + (1 if position < remainder else 0)
            if alloc <= 0:
                continue
            if alloc >= approx_entries:
                plan.append((files[file_idx], None))
            else:
                positions = set(self._evenly_spaced_indices(approx_entries, alloc))
                plan.append((files[file_idx], positions))
        return plan

    def _process_file_streaming(
        self,
        file_path: Path,
        selected_positions: Optional[Set[int]] = None,
    ) -> Tuple[int, int]:
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            return self._process_csv_streaming(file_path, selected_positions)
        if suffix not in {".json", ".jsonl"}:
            self._print(f"Skipping unsupported file format: {file_path.name}")
            return 0, 0

        total_entries = 0
        processed_entries = 0

        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                return 0, 0

            total_entries = len(data)
            positions = (
                sorted(selected_positions)
                if selected_positions is not None
                else (
                    range(0, total_entries, max(1, total_entries // 200))
                    if total_entries > 1000
                    else range(total_entries)
                )
            )

            for idx in positions:
                if idx >= total_entries:
                    break
                entry = data[idx]
                if not isinstance(entry, dict):
                    continue

                flat_entry = flatten_dict(entry)
                step_val = self._extract_step_value(flat_entry)
                if step_val is None or step_val > self.max_steps:
                    continue

                normalized_entry = self._normalize_entry_format(flat_entry, self.current_variant or "unknown")
                self._reservoir_sample(normalized_entry)
                self._update_statistics(step_val)
                self._update_heatmap_counts_from_entry(normalized_entry)
                processed_entries += 1
                if (
                    selected_positions is not None
                    and processed_entries >= len(selected_positions)
                ):
                    break

        except Exception as exc:
            self._print(f"Error processing {file_path.name}: {exc}")
            return 0, 0

        gc.collect()
        return total_entries, processed_entries

    def _process_csv_streaming(
        self,
        csv_file: Path,
        selected_positions: Optional[Set[int]] = None,
    ) -> Tuple[int, int]:
        total_entries = 0
        processed_entries = 0
        max_entries = len(selected_positions) if selected_positions is not None else None
        positions_lookup = selected_positions if selected_positions is not None else None
        entry_index = 0

        try:
            for chunk in pd.read_csv(csv_file, chunksize=5000):
                total_entries += len(chunk)
                for entry in chunk.to_dict(orient="records"):
                    if positions_lookup is not None and entry_index not in positions_lookup:
                        entry_index += 1
                        continue
                    step_val = self._extract_step_value(entry)
                    if step_val is None or step_val > self.max_steps:
                        entry_index += 1
                        continue
                    normalized_entry = self._normalize_entry_format(entry, self.current_variant or "unknown")
                    self._reservoir_sample(normalized_entry)
                    self._update_statistics(step_val)
                    self._update_heatmap_counts_from_entry(normalized_entry)
                    processed_entries += 1
                    entry_index += 1
                    if max_entries is not None and processed_entries >= max_entries:
                        break
                if max_entries is not None and processed_entries >= max_entries:
                    break
        except Exception as exc:
            self._print(f"Error processing {csv_file.name}: {exc}")
            return 0, 0

        gc.collect()
        return total_entries, processed_entries

    def _extract_step_value(self, entry: Dict) -> Optional[int]:
        for key in ("total_steps", "step", "timesteps", "global_step"):
            if key in entry and pd.notna(entry[key]):
                try:
                    return int(entry[key])
                except (TypeError, ValueError):
                    continue
        step_keys = [k for k in entry.keys() if "step" in k.lower()]
        for key in step_keys:
            try:
                return int(entry[key])
            except (TypeError, ValueError):
                continue
        return None

    def _update_statistics(self, step_val: int) -> None:
        self.step_range[0] = min(self.step_range[0], step_val)
        self.step_range[1] = max(self.step_range[1], step_val)

    def _update_heatmap_counts_from_entry(self, entry: Dict) -> None:
        map_id = entry.get("map_id")
        if map_id is None:
            return
        try:
            map_id_int = int(float(map_id))
        except (ValueError, TypeError):
            return

        x = entry.get("position_x")
        y = entry.get("position_y")
        if x is None or y is None:
            return
        try:
            x_int = int(float(x))
            y_int = int(float(y))
        except (ValueError, TypeError):
            return

        self.map_visit_counts[map_id_int][(x_int, y_int)] += 1

    def _normalize_entry_format(self, entry: dict, variant_name: str) -> dict:
        if any("." in key for key in entry.keys()):
            entry = {key.replace(".", "_"): value for key, value in entry.items()}

        step_val = self._extract_step_value(entry)

        if "rewards_total" in entry:
            normalised = {
                "step": step_val,
                "total_steps": entry.get("total_steps", step_val),
                "total_reward": entry.get("rewards_total", 0),
                "badge_reward": entry.get("rewards_components_badge", 0),
                "event_reward": entry.get("rewards_components_event", 0),
                "level_reward": entry.get("rewards_components_level", 0),
                "heal_reward": entry.get("rewards_components_heal", 0),
                "explore_reward": entry.get("rewards_components_explore", 0),
                "dead_penalty": entry.get("rewards_components_dead", 0),
                "stuck_penalty": entry.get("rewards_components_stuck", 0),
                "position_x": entry.get("position_x", 0),
                "position_y": entry.get("position_y", 0),
                "map_id": entry.get("position_map", 0),
                "badges": entry.get("player_status_badges", 0),
                "levels_sum": entry.get("player_status_levels_sum", 0),
                "pokemon_count": entry.get("player_status_pokemon_count", 0),
                "health": entry.get("player_status_health", 0),
                "exploration": entry.get("statistics_exploration_coords", 0),
                "deaths": entry.get("statistics_deaths", 0),
                "last_action": entry.get("actions_last_action", 0),
                "variant": variant_name,
            }
        else:
            normalised = {
                "step": step_val,
                "total_steps": step_val,
                "total_reward": entry.get("event", 0) + entry.get("badge", 0),
                "badge_reward": entry.get("badge", 0),
                "event_reward": entry.get("event", 0),
                "level_reward": entry.get("level", 0),
                "heal_reward": entry.get("healr", 0),
                "explore_reward": entry.get("explore", 0),
                "dead_penalty": entry.get("dead", 0),
                "stuck_penalty": entry.get("stuck", 0),
                "position_x": entry.get("x", 0),
                "position_y": entry.get("y", 0),
                "map_id": entry.get("map", 0),
                "badges": entry.get("badge", 0),
                "levels_sum": entry.get("levels_sum", 0),
                "pokemon_count": entry.get("pcount", 0),
                "health": entry.get("hp", 0),
                "exploration": entry.get("coord_count", 0),
                "deaths": entry.get("deaths", 0),
                "last_action": entry.get("last_action", 0),
                "variant": variant_name,
            }

        training_metrics = {
            key: entry[key]
            for key in entry.keys()
            if key.startswith("training_")
        }
        if training_metrics:
            normalised.update(training_metrics)

        return normalised

    def load_variant_data(
        self,
        experiment_path: str,
        variant_name: str,
        max_files: Optional[int] = None,
        max_data_points: Optional[int] = None,
    ) -> pd.DataFrame:
        self.reservoir_samples = []
        self.total_entries_seen = 0
        self.current_variant = variant_name
        self.step_range = [float("inf"), float("-inf")]
        self._reset_heatmap_counts()
        self.latest_map_counts = {}
        self.latest_sampling_meta = {}

        experiment_path = Path(experiment_path)
        self._print(f"Streaming load: {variant_name} from {experiment_path}")

        stat_files = sorted(
            list(experiment_path.glob("*.json"))
            + list(experiment_path.glob("*.jsonl"))
            + list(experiment_path.glob("*.csv"))
        )
        total_available = len(stat_files)
        if max_data_points is not None and max_files is not None:
            raise ValueError("max_files and max_data_points cannot be used together.")

        selected_positions: List[Optional[Set[int]]]
        if max_data_points is not None:
            file_plan = self._plan_files_for_max_points(stat_files, max_data_points)
            stat_files = [item[0] for item in file_plan]
            selected_positions = [item[1] for item in file_plan]
            if max_data_points > 0:
                self._print(
                    f"Targeting {max_data_points:,} data points across {len(stat_files)} files "
                    f"(out of {total_available} total)"
                )
        else:
            if max_files is not None and stat_files:
                stat_files = stat_files[:max_files]
                self._print(f"Limited to {max_files} files (out of {total_available} total)")
            selected_positions = [None] * len(stat_files)

        self._print(f"Found {len(stat_files)} files to process")
        if len(stat_files) == 0:
            self._print("No statistics files found!")
            return pd.DataFrame()

        total_entries = 0
        total_processed = 0

        for i, stat_file in enumerate(tqdm(stat_files, desc=f"Streaming {variant_name}")):
            try:
                if self.verbose and (i < 5 or (i + 1) % 50 == 0):
                    file_size_mb = stat_file.stat().st_size / 1024 / 1024
                    self._print(f"File {i+1}/{len(stat_files)}: {stat_file.name} ({file_size_mb:.1f} MB)")

                file_entries, processed_entries = self._process_file_streaming(
                    stat_file, selected_positions[i]
                )
                total_entries += file_entries
                total_processed += processed_entries

                if (i + 1) % 100 == 0:
                    sampling_ratio = len(self.reservoir_samples) / max(1, self.total_entries_seen) * 100
                    self._print(
                        f"Progress: {i+1}/{len(stat_files)} files, "
                        f"{self.total_entries_seen:,} entries seen, "
                        f"{len(self.reservoir_samples):,} in reservoir ({sampling_ratio:.2f}%)"
                    )
            except Exception as exc:
                self._print(f"Error processing {stat_file.name}: {exc}")
                continue

        if not self.reservoir_samples:
            self._print(f"No valid data loaded for {variant_name}")
            return pd.DataFrame()

        df = pd.DataFrame(self.reservoir_samples)
        step_col = "total_steps" if "total_steps" in df.columns and df["total_steps"].notna().any() else "step"
        df = df.sort_values(step_col).reset_index(drop=True)

        raw_sampling_ratio = len(self.reservoir_samples) / max(1, self.total_entries_seen)
        sampling_ratio = raw_sampling_ratio * 100

        self._print(f"Streaming complete for {variant_name}:")
        self._print(f"   Total entries in files: {total_entries:,}")
        self._print(f"   Entries processed: {total_processed:,}")
        self._print(f"   Final samples: {len(df):,}")
        self._print(f"   Sampling ratio: {sampling_ratio:.2f}%")
        self._print(f"   Step range: {df[step_col].min():,} - {df[step_col].max():,}")
        if "total_reward" in df.columns:
            self._print(f"   Reward range: {df['total_reward'].min():.3f} - {df['total_reward'].max():.3f}")

        self.latest_sampling_meta = {
            "samples": len(df),
            "entries_seen": self.total_entries_seen,
            "sampling_ratio": raw_sampling_ratio,
            "step_range": (df[step_col].min(), df[step_col].max()),
        }
        self.latest_map_counts = {
            map_id: dict(counts)
            for map_id, counts in self.map_visit_counts.items()
        }

        return df

    def normalize_step_ranges(self, *dataframes: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
        if not dataframes or all(len(df) == 0 for df in dataframes):
            return dataframes

        max_steps = []
        for df in dataframes:
            if len(df) == 0:
                continue
            step_col = "total_steps" if "total_steps" in df.columns and df["total_steps"].notna().any() else "step"
            max_steps.append(df[step_col].max())

        if not max_steps:
            return dataframes

        common_max_step = min(max_steps)
        self._print(f"Using common max step: {common_max_step:,}")

        filtered_dfs = []
        for df in dataframes:
            if len(df) == 0:
                filtered_dfs.append(df)
                continue
            step_col = "total_steps" if "total_steps" in df.columns and df["total_steps"].notna().any() else "step"
            filtered_df = df[df[step_col] <= common_max_step].copy()
            self._print(
                f"Filtered dataset: {len(filtered_df):,} entries, "
                f"step range: {filtered_df[step_col].min():,} - {filtered_df[step_col].max():,}"
            )
            filtered_dfs.append(filtered_df)
        return tuple(filtered_dfs)

    def get_summary_stats(self, df: pd.DataFrame, variant_name: str) -> Dict:
        if len(df) == 0:
            return {"variant": variant_name, "samples": 0}

        step_col = "total_steps" if "total_steps" in df.columns and df["total_steps"].notna().any() else "step"

        return {
            "variant": variant_name,
            "samples": len(df),
            "step_range": (df[step_col].min(), df[step_col].max()),
            "mean_total_reward": df["total_reward"].mean(),
            "std_total_reward": df["total_reward"].std(),
            "max_badges": df["badges"].max(),
            "max_exploration": df["exploration"].max(),
            "mean_exploration": df["exploration"].mean(),
            "mean_deaths": df["deaths"].mean(),
            "mean_health": df["health"].mean(),
        }


class TrainingSampler(StreamingDataSampler):
    """Backward compatibility alias for StreamingDataSampler."""


def load_variants_for_comparison(
    variant_configs: List[Tuple[str, str]],
    max_steps: int = 100_000_000,
    target_samples: int = 5000,
    max_files: Optional[int] = None,
    max_data_points: Optional[int] = None,
    normalize_steps: bool = True,
    verbose: bool = True,
) -> Tuple[List[pd.DataFrame], TrainingSampler]:
    sampler = TrainingSampler(max_steps=max_steps, target_samples=target_samples, verbose=verbose)
    sampler.variant_heatmap_counts = {}
    sampler.variant_sampling_meta = {}

    dataframes = []
    for experiment_path, variant_name in variant_configs:
        df = sampler.load_variant_data(
            experiment_path,
            variant_name,
            max_files=max_files,
            max_data_points=max_data_points,
        )
        dataframes.append(df)
        sampler.variant_heatmap_counts[variant_name] = sampler.latest_map_counts
        sampler.variant_sampling_meta[variant_name] = sampler.latest_sampling_meta

    if normalize_steps and len(dataframes) > 1:
        if verbose:
            print("\n=== Normalizing Step Ranges for Fair Comparison ===")
        dataframes = list(sampler.normalize_step_ranges(*dataframes))

    return dataframes, sampler


__all__ = [
    "StreamingDataSampler",
    "TrainingSampler",
    "flatten_dict",
    "load_variants_for_comparison",
]
