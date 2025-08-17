#!/usr/bin/env python3
"""
plot_helpers.py - Common plotting functions and configuration
~80-120 Zeilen (plotting functions + config constants)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.patches as patches

# Configuration Constants
PLOT_CONFIG = {
    'figsize': (12, 8),
    'dpi': 300,
    'style': 'whitegrid',
    'palette': 'tab10',
    'font_size': 12,
    'title_size': 14,
    'label_size': 11,
    'legend_size': 10,
    'line_width': 1.5,
    'alpha': 0.7,
    'grid_alpha': 0.3
}

REWARD_COLORS = {
    'total': '#1f77b4',      # Blue
    'event': '#ff7f0e',      # Orange  
    'level': '#2ca02c',      # Green
    'heal': '#d62728',       # Red
    'badge': '#9467bd',      # Purple
    'explore': '#8c564b',    # Brown
    'dead': '#e377c2',       # Pink
    'stuck': '#7f7f7f'       # Gray
}

OUTPUT_FORMATS = ['png', 'pdf', 'svg']


def setup_plot_style():
    """Konfiguriert den Plot-Stil für konsistente Visualisierungen."""
    plt.style.use('default')
    sns.set_style(PLOT_CONFIG['style'])
    sns.set_palette(PLOT_CONFIG['palette'])
    
    plt.rcParams.update({
        'font.size': PLOT_CONFIG['font_size'],
        'axes.titlesize': PLOT_CONFIG['title_size'],
        'axes.labelsize': PLOT_CONFIG['label_size'],
        'legend.fontsize': PLOT_CONFIG['legend_size'],
        'figure.dpi': PLOT_CONFIG['dpi'],
        'savefig.dpi': PLOT_CONFIG['dpi'],
        'figure.figsize': PLOT_CONFIG['figsize']
    })


def smooth_series(series: pd.Series, window: int = 100) -> pd.Series:
    """
    Anwendung eines Moving Average zur Glättung von Zeitreihen.
    
    Args:
        series: Zu glättende Pandas Series
        window: Fenstergröße für Moving Average
    
    Returns:
        Geglättete Series
    """
    if len(series) < window:
        window = max(1, len(series) // 4)
    
    return series.rolling(window=window, center=True, min_periods=1).mean()


def get_reward_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Identifiziert Reward-Spalten im DataFrame und mappt sie zu Standardnamen.
    
    Args:
        df: DataFrame mit Trainingsdaten
    
    Returns:
        Dictionary mit Mapping von Standardnamen zu tatsächlichen Spaltennamen
    """
    reward_mapping = {}
    
    # Suche nach verschiedenen Reward-Spalten
    for col in df.columns:
        col_lower = col.lower()
        
        if 'reward' in col_lower:
            # Versuche spezifische Reward-Typen zu identifizieren
            if any(reward_type in col_lower for reward_type in ['total', 'sum']):
                reward_mapping['total'] = col
            elif 'event' in col_lower:
                reward_mapping['event'] = col
            elif 'level' in col_lower:
                reward_mapping['level'] = col
            elif 'heal' in col_lower:
                reward_mapping['heal'] = col
            elif 'badge' in col_lower:
                reward_mapping['badge'] = col
            elif 'explore' in col_lower:
                reward_mapping['explore'] = col
            elif 'dead' in col_lower:
                reward_mapping['dead'] = col
            elif 'stuck' in col_lower:
                reward_mapping['stuck'] = col
            else:
                # Falls kein spezifischer Typ erkannt wird, nutze als total
                if 'total' not in reward_mapping:
                    reward_mapping['total'] = col
    
    return reward_mapping


def get_position_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Identifiziert X- und Y-Position-Spalten im DataFrame.
    
    Args:
        df: DataFrame mit Trainingsdaten
    
    Returns:
        Tuple (x_column, y_column) oder (None, None) falls nicht gefunden
    """
    x_col, y_col = None, None
    
    for col in df.columns:
        col_lower = col.lower()
        
        if ('x' in col_lower and any(pos in col_lower for pos in ['pos', 'coord'])) or col_lower == 'x':
            x_col = col
        elif ('y' in col_lower and any(pos in col_lower for pos in ['pos', 'coord'])) or col_lower == 'y':
            y_col = col
    
    return x_col, y_col


def create_plot_title(variant: str, timestamp: str, additional_info: str = "") -> str:
    """
    Erstellt einen standardisierten Plot-Titel.
    
    Args:
        variant: Agent-Variante (v1, v2, v3, v4)
        timestamp: Zeitstempel des Trainings
        additional_info: Zusätzliche Informationen für den Titel
    
    Returns:
        Formatierter Titel-String
    """
    base_title = f"Agent {variant.upper()} - Training {timestamp}"
    if additional_info:
        return f"{base_title} - {additional_info}"
    return base_title


def save_plot(fig: plt.Figure, output_path: Path, filename: str, formats: List[str] = None):
    """
    Speichert einen Plot in verschiedenen Formaten.
    
    Args:
        fig: Matplotlib Figure Objekt
        output_path: Ausgabe-Verzeichnis
        filename: Basis-Dateiname (ohne Erweiterung)
        formats: Liste der zu speichernden Formate
    """
    if formats is None:
        formats = ['png']
    
    # Erstelle Ausgabe-Verzeichnis falls nötig
    output_path.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        if fmt in OUTPUT_FORMATS:
            filepath = output_path / f"{filename}.{fmt}"
            fig.savefig(filepath, format=fmt, bbox_inches='tight', 
                       dpi=PLOT_CONFIG['dpi'], facecolor='white')
            print(f"Plot saved: {filepath}")
        else:
            print(f"Warning: Unsupported format '{fmt}', skipping")


def apply_plot_styling(ax: plt.Axes, title: str, xlabel: str, ylabel: str):
    """
    Wendet konsistentes Styling auf eine Axes an.
    
    Args:
        ax: Matplotlib Axes Objekt
        title: Plot-Titel
        xlabel: X-Achsen-Label
        ylabel: Y-Achsen-Label
    """
    ax.set_title(title, fontsize=PLOT_CONFIG['title_size'], fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=PLOT_CONFIG['label_size'])
    ax.set_ylabel(ylabel, fontsize=PLOT_CONFIG['label_size'])
    ax.grid(True, alpha=PLOT_CONFIG['grid_alpha'])
    ax.legend(fontsize=PLOT_CONFIG['legend_size'])
