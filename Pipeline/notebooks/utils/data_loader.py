#!/usr/bin/env python3
"""
data_loader.py - JSON loading utilities for training data visualization
~80-120 Zeilen (JSON loading + flattening logic)
"""

import json
import pandas as pd
from pathlib import Path
import glob
import re
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
import sys


def find_latest_experiment_dir(variant: str, base_path: Path = None) -> Optional[Path]:
    """
    Findet das neueste Experiment-Verzeichnis für eine gegebene Variante.
    
    Args:
        variant: Agent-Variante (v1, v2, v3, v4)
        base_path: Basis-Pfad für Experimente (default: "../experiments")
    
    Returns:
        Path zum neuesten Experiment-Verzeichnis oder None
    """
    if base_path is None:
        # Da das Skript im notebooks/ Ordner läuft, gehe eine Ebene höher
        current_dir = Path(__file__).parent.parent  # Von utils/ zu notebooks/
        base_path = current_dir.parent / "experiments"  # Von notebooks/ zu Pipeline/experiments/
    
    variant_path = base_path / variant
    
    if not variant_path.exists():
        raise FileNotFoundError(f"Variant path not found: {variant_path}")
    
    # Finde alle Zeitstempel-Verzeichnisse
    experiment_dirs = [d for d in variant_path.iterdir() 
                      if d.is_dir() and re.match(r'\d{8}_\d{6}', d.name)]
    
    if not experiment_dirs:
        raise FileNotFoundError(f"No experiment directories found in {variant_path}")
    
    # Sortiere nach Zeitstempel (neuestes zuerst)
    experiment_dirs.sort(key=lambda x: x.name, reverse=True)
    
    return experiment_dirs[0]


def find_json_log_files(experiment_dir: Path) -> List[Path]:
    """
    Findet alle JSON-Log-Dateien in einem Experiment-Verzeichnis.
    
    Args:
        experiment_dir: Pfad zum Experiment-Verzeichnis
    
    Returns:
        Liste der JSON-Log-Dateien, sortiert nach Änderungszeit
    """
    json_logs_dir = experiment_dir / "json_logs"
    
    if not json_logs_dir.exists():
        raise FileNotFoundError(f"JSON logs directory not found: {json_logs_dir}")
    
    # Suche nach stats_*.json Dateien
    json_files = list(json_logs_dir.glob("stats_*.json"))
    
    if not json_files:
        raise FileNotFoundError(f"No JSON log files found in {json_logs_dir}")
    
    # Sortiere nach Änderungszeit (neueste zuerst)
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return json_files


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Rekursives Flattening eines verschachtelten Dictionaries.
    Basiert auf der Logik aus debug_logs.py
    
    Args:
        d: Dictionary zum Flatten
        parent_key: Übergeordneter Schlüssel
        sep: Separator für verschachtelte Schlüssel
    
    Returns:
        Geflattetes Dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_and_flatten_json_data(json_files: List[Path], max_files: int = None) -> pd.DataFrame:
    """
    Lädt und flacht JSON-Daten von mehreren Dateien.
    
    Args:
        json_files: Liste der JSON-Dateien
        max_files: Maximale Anzahl zu ladender Dateien
    
    Returns:
        Kombinierter und geflatteter DataFrame
    """
    if max_files:
        json_files = json_files[:max_files]
    
    all_data = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                print(f"Warning: {json_file.name} does not contain a list")
                continue
            
            # Flatten jedes Entry
            for entry in data:
                if isinstance(entry, dict):
                    flat_entry = flatten_dict(entry)
                    all_data.append(flat_entry)
                else:
                    all_data.append(entry)
                    
        except Exception as e:
            print(f"Warning: Failed to load {json_file.name}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid data could be loaded from JSON files")
    
    # Erstelle DataFrame
    df = pd.DataFrame(all_data)
    
    return df


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validiert ob der DataFrame die erwarteten Spalten für Visualisierung enthält.
    
    Args:
        df: Zu validierender DataFrame
    
    Returns:
        Tuple (is_valid, missing_requirements)
    """
    missing = []
    
    # Prüfe auf Step-Spalte
    step_cols = [col for col in df.columns if 'step' in col.lower()]
    if not step_cols:
        missing.append("step column")
    
    # Prüfe auf Position-Spalten
    position_cols = [col for col in df.columns if any(pos in col.lower() for pos in ['x', 'y', 'position'])]
    if len(position_cols) < 2:
        missing.append("position columns (x, y)")
    
    # Prüfe auf Reward-Spalten
    reward_cols = [col for col in df.columns if 'reward' in col.lower()]
    if not reward_cols:
        missing.append("reward columns")
    
    return len(missing) == 0, missing
