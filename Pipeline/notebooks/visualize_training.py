#!/usr/bin/env python3
"""
visualize_training.py - Streaming-basierte Visualisierung für massive Datasets
Implementiert Best Practices: Streaming, inkrementelle Aggregation, intelligentes Sampling

Basiert auf visualization_guidelines.md:
- Verarbeitet ALLE JSON-Dateien eines Training-Runs
- Verwendet Reservoir Sampling für gleichmäßige Datenverteilung  
- Memory-bounded Processing (konstant 10k Samples im RAM)
- Inkrementelle Aggregation für Position-Heatmaps
"""

import argparse
import sys
import gc
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Dict, Tuple
import json
from collections import defaultdict
import random

# Import utility modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_loader import (
    find_latest_experiment_dir, 
    find_json_log_files, 
    flatten_dict,
    validate_dataframe
)
from utils.plot_helpers import (
    setup_plot_style,
    smooth_series,
    get_reward_columns,
    get_position_columns,
    create_plot_title,
    save_plot,
    apply_plot_styling,
    REWARD_COLORS,
    PLOT_CONFIG
)


class StreamingProcessor:
    """Streaming processor für massive JSON datasets mit intelligenter Aggregation"""
    
    def __init__(self, target_sample_size: int = 10000):
        self.target_sample_size = target_sample_size
        self.reservoir_samples = []  # Reservoir sampling buffer
        self.total_entries_seen = 0
        
        # Rolling statistics für inkrementelle Aggregation
        self.reward_stats = defaultdict(list)
        self.step_range = [float('inf'), float('-inf')]
        
        # Heatmap aggregation
        self.position_counts = defaultdict(int)
        
    def process_file_streaming(self, json_file: Path, debug_first_entry: bool = False) -> Tuple[int, Dict]:
        """
        Verarbeitet eine JSON-Datei mit Streaming und Reservoir Sampling
        
        Returns:
            (entries_count, file_stats)
        """
        entries_count = 0
        file_stats = {'rewards': {}, 'steps': [], 'positions': []}
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                return 0, file_stats
                
            entries_count = len(data)
            
            # Intelligentes Sampling: Zeitlich gleichverteilt
            # Nimm jeden N-ten Eintrag für gleichmäßige zeitliche Verteilung
            if entries_count > 1000:
                sample_step = max(1, entries_count // 100)  # Max 100 samples per file
                sampled_indices = list(range(0, entries_count, sample_step))
            else:
                sampled_indices = list(range(entries_count))
            
            for idx in sampled_indices:
                entry = data[idx]
                if isinstance(entry, dict):
                    flat_entry = flatten_dict(entry)
                    
                    # Debug: Zeige ersten Eintrag
                    if debug_first_entry and idx == sampled_indices[0]:
                        print(f"🔍 Debug - Erste 10 Keys des geflatteten Eintrags:")
                        for i, (key, value) in enumerate(list(flat_entry.items())[:10]):
                            print(f"   '{key}': {value}")
                        print(f"🔍 Debug - Environment Step: {flat_entry.get('step', 'N/A')}")
                        print(f"🔍 Debug - Globaler Step: {flat_entry.get('total_steps', 'N/A')}")
                        print(f"🔍 Debug - Alle Keys die 'step' enthalten:")
                        step_keys = [k for k in flat_entry.keys() if 'step' in k.lower()]
                        for key in step_keys:
                            print(f"   '{key}': {flat_entry[key]}")
                        print(f"🔍 Debug - Alle Keys die 'position' oder 'x'/'y' enthalten:")
                        pos_keys = [k for k in flat_entry.keys() if any(pos in k.lower() for pos in ['position', 'x', 'y'])]
                        for key in pos_keys:
                            print(f"   '{key}': {flat_entry[key]}")
                    
                    # Reservoir Sampling für finale Visualisierung
                    self._reservoir_sample(flat_entry)
                    
                    # Inkrementelle Statistiken
                    self._update_rolling_stats(flat_entry)
                    
        except Exception as e:
            print(f"Fehler beim Streaming von {json_file.name}: {e}")
            return 0, file_stats
        
        # Garbage Collection nach jeder Datei
        gc.collect()
        return entries_count, file_stats
    
    def _reservoir_sample(self, entry: Dict):
        """Implementiert Reservoir Sampling für gleichmäßige Verteilung"""
        self.total_entries_seen += 1
        
        if len(self.reservoir_samples) < self.target_sample_size:
            # Fülle Reservoir auf
            self.reservoir_samples.append(entry)
        else:
            # Reservoir ist voll: Ersetze zufällig
            j = random.randint(0, self.total_entries_seen - 1)
            if j < self.target_sample_size:
                self.reservoir_samples[j] = entry
    
    def _update_rolling_stats(self, entry: Dict):
        """Aktualisiert Rolling Statistics inkrementell"""
        # Step-Bereich tracking - verwende globale Steps falls verfügbar
        step_val = None
        
        # Priorität: total_steps (globale Steps) > step (Environment-Steps)
        if 'total_steps' in entry:
            step_val = entry['total_steps']
        elif 'step' in entry:
            step_val = entry['step']
        else:
            # Fallback: Suche nach step-enthaltenden Keys
            step_cols = [col for col in entry.keys() if 'step' in col.lower()]
            if step_cols:
                step_val = entry.get(step_cols[0])
        
        if step_val is not None and isinstance(step_val, (int, float)):
            self.step_range[0] = min(self.step_range[0], step_val)
            self.step_range[1] = max(self.step_range[1], step_val)
        
        # Position-Heatmap aggregation - nach flattening sollten Keys wie 'position_x', 'position_y' existieren
        x_val, y_val = None, None
        
        # Variante 1: Geflattete Position-Keys
        if 'position_x' in entry and 'position_y' in entry:
            x_val = entry['position_x']
            y_val = entry['position_y']
        
        # Variante 2: Andere mögliche Formate nach flattening
        if x_val is None or y_val is None:
            for key in entry.keys():
                key_lower = key.lower()
                if ('position' in key_lower and 'x' in key_lower) and x_val is None:
                    x_val = entry.get(key)
                elif ('position' in key_lower and 'y' in key_lower) and y_val is None:
                    y_val = entry.get(key)
                
        # Variante 3: Direkte x/y Suche
        if x_val is None or y_val is None:
            if 'x' in entry and x_val is None:
                x_val = entry['x']
            if 'y' in entry and y_val is None:
                y_val = entry['y']
        
        # Speichere Position wenn beide Koordinaten verfügbar
        if x_val is not None and y_val is not None:
            try:
                x_int = int(x_val)
                y_int = int(y_val)
                self.position_counts[(x_int, y_int)] += 1
            except (ValueError, TypeError):
                pass  # Ignoriere ungültige Koordinaten
    
    def get_final_dataframe(self) -> pd.DataFrame:
        """Konvertiert Reservoir Samples zu DataFrame"""
        if not self.reservoir_samples:
            raise ValueError("Keine Samples gesammelt")
        
        df = pd.DataFrame(self.reservoir_samples)
        
        # Sortiere nach Steps - bevorzuge total_steps (globale Steps)
        step_col = None
        if 'total_steps' in df.columns:
            step_col = 'total_steps'
        else:
            # Fallback auf Environment Steps
            step_cols = [col for col in df.columns if 'step' in col.lower()]
            if step_cols:
                step_col = step_cols[0]
        
        if step_col:
            df = df.sort_values(step_col)
        else:
            raise ValueError("Keine Step-Spalte gefunden")
        
        return df
    
    def get_statistics(self) -> Dict:
        """Gibt gesammelte Statistiken zurück"""
        return {
            'total_entries_processed': self.total_entries_seen,
            'sample_size': len(self.reservoir_samples),
            'step_range': self.step_range if self.step_range[0] != float('inf') else [0, 0],
            'position_count': len(self.position_counts),
            'position_data': dict(self.position_counts),
            'sampling_ratio': len(self.reservoir_samples) / max(1, self.total_entries_seen) * 100
        }


def process_dataset_streaming(variant: str, max_files: Optional[int], verbose: bool, experiment_name: Optional[str] = None) -> Tuple[pd.DataFrame, Path, str, Dict]:
    """
    Streaming-basierte Verarbeitung des gesamten Datasets
    
    Returns:
        (dataframe, experiment_dir, timestamp, statistics)
    """
    if verbose:
        search_name = experiment_name or variant
        print(f"📁 Suche neuestes Experiment für {search_name}...")
    
    # Finde neuestes Experiment - verwende experiment_name falls angegeben
    if experiment_name:
        # Direkter Pfad zu spezifischem Experiment-Subdirectory
        base_path = Path("experiments") / experiment_name
        if not base_path.exists():
            raise FileNotFoundError(f"Experiment-Subdirectory nicht gefunden: {base_path}")
        
        # Finde neuestes Experiment in diesem Subdirectory
        experiment_dirs = [d for d in base_path.iterdir() if d.is_dir()]
        if not experiment_dirs:
            raise FileNotFoundError(f"Keine Experiment-Ordner in {base_path} gefunden")
        
        experiment_dir = max(experiment_dirs, key=lambda x: x.name)
    else:
        experiment_dir = find_latest_experiment_dir(variant)
    
    timestamp = experiment_dir.name
    
    if verbose:
        print(f"   Gefunden: {experiment_dir}")
    
    # Finde JSON-Dateien
    json_files = find_json_log_files(experiment_dir)
    
    if max_files:
        json_files = json_files[:max_files]
        print(f"⚠️  Verwende nur {max_files} von {len(json_files)} Dateien")
    
    print(f"🌊 Streaming-Verarbeitung von {len(json_files)} JSON-Dateien")
    
    # Initialisiere Streaming Processor
    processor = StreamingProcessor(target_sample_size=10000)
    
    total_entries = 0
    processed_files = 0
    
    # Verarbeite Dateien sequenziell mit Streaming
    for i, json_file in enumerate(json_files):
        if verbose or (i + 1) % 50 == 0:
            print(f"📄 Streaming Datei {i+1}/{len(json_files)}: {json_file.name}")
        
        # Debug nur für die erste Datei
        entries_count, _ = processor.process_file_streaming(json_file, debug_first_entry=(i == 0 and verbose))
        total_entries += entries_count
        processed_files += 1
        
        # Fortschritts-Update
        if (i + 1) % 100 == 0:
            stats = processor.get_statistics()
            print(f"   Fortschritt: {processed_files}/{len(json_files)} Dateien, "
                  f"{stats['total_entries_processed']:,} Einträge verarbeitet")
            print(f"   Aktueller Step-Bereich: {stats['step_range'][0]:,} bis {stats['step_range'][1]:,}")
            print(f"   Position-Daten bisher: {stats['position_count']:,} eindeutige Positionen")
    
    # Finale Statistiken
    stats = processor.get_statistics()
    stats['total_files'] = len(json_files)
    stats['total_entries'] = total_entries
    
    print(f"✅ Streaming abgeschlossen:")
    print(f"   📊 Dateien verarbeitet: {processed_files:,}")
    print(f"   📈 Gesamte Einträge: {total_entries:,}")
    print(f"   🎯 Reservoir Samples: {stats['sample_size']:,}")
    print(f"   📉 Sampling Ratio: {stats['sampling_ratio']:.2f}%")
    print(f"   🕐 Step-Bereich: {stats['step_range'][0]:,} bis {stats['step_range'][1]:,}")
    
    # Erstelle finalen DataFrame
    df = processor.get_final_dataframe()
    
    # Validiere Daten
    is_valid, missing = validate_dataframe(df)
    if not is_valid:
        print(f"⚠️  Datenvalidierung: Fehlend: {missing}")
    
    return df, experiment_dir, timestamp, stats


def plot_rewards_streaming(df: pd.DataFrame, variant: str, timestamp: str, 
                          output_dir: Path, smooth_window: int, output_format: str, 
                          stats: Dict, verbose: bool):
    """Streaming-optimierte Reward-Visualisierung"""
    
    if verbose:
        print("📈 Erstelle Streaming-optimierten Reward-Plot...")
    
    setup_plot_style()
    reward_mapping = get_reward_columns(df)
    
    if not reward_mapping:
        raise ValueError("Keine Reward-Spalten gefunden")
    
    # Bevorzuge total_steps (globale Steps) für Achsen-Beschriftung  
    step_col = None
    if 'total_steps' in df.columns:
        step_col = 'total_steps'
    else:
        # Fallback auf Environment Steps
        step_cols = [col for col in df.columns if 'step' in col.lower()]
        if step_cols:
            step_col = step_cols[0]
    
    if not step_col:
        raise ValueError("Keine Step-Spalte gefunden")
    
    # Erstelle Plot
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize'])
    
    # Sortiere nach Steps
    df_plot = df.sort_values(step_col)
    
    # Debug: Überprüfe Step-Bereich im DataFrame
    if verbose:
        print(f"🔍 Debug DataFrame Step-Bereich:")
        print(f"   Min Step im DataFrame: {df_plot[step_col].min():,}")
        print(f"   Max Step im DataFrame: {df_plot[step_col].max():,}")
        print(f"   Anzahl Datenpunkte: {len(df_plot):,}")
        print(f"   Step-Spalte: '{step_col}'")
    
    # Plotte jede Reward-Komponente
    for reward_type, col_name in reward_mapping.items():
        if col_name in df_plot.columns:
            # Adaptive Glättung basierend auf Datengröße
            adaptive_window = min(smooth_window, len(df_plot) // 20)
            smoothed_values = smooth_series(df_plot[col_name], adaptive_window)
            
            # Bestimme Farbe und Stil
            color = REWARD_COLORS.get(reward_type, '#888888')
            linestyle = '-' if reward_type == 'total' else '--'
            linewidth = 2.0 if reward_type == 'total' else 1.0
            alpha = 1.0 if reward_type == 'total' else 0.7
            
            ax.plot(
                df_plot[step_col], 
                smoothed_values, 
                label=f'{reward_type.title()} Reward',
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha
            )
    
    # Title mit korrekten Statistiken
    title = create_plot_title(variant, timestamp, 
                            f"Rewards (Reservoir Sample: {stats['sample_size']:,}/{stats['total_entries']:,})")
    apply_plot_styling(ax, title, "Training Steps", "Reward")
    
    # Explizite X-Achsen-Skalierung basierend auf tatsächlichen Daten
    actual_min = df_plot[step_col].min()
    actual_max = df_plot[step_col].max()
    ax.set_xlim(actual_min, actual_max)
    
    # Formatiere X-Achsen-Ticks für große Zahlen
    if actual_max > 1000000000:
        # Für Milliarden: zeige in B Format
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e9:.1f}B'))
    elif actual_max > 1000000:
        # Für Millionen: zeige in M Format
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    elif actual_max > 1000:
        # Für Tausende: zeige in K Format
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}K'))
    
    if verbose:
        print(f"   X-Achse gesetzt: {actual_min:,} bis {actual_max:,}")
    
    # Speichere Plot
    filename = f"{variant}_{timestamp}_rewards_streaming"
    save_plot(fig, output_dir, filename, [output_format])
    
    plt.close(fig)
    
    if verbose:
        print(f"   Streaming Reward-Plot erstellt: {filename}.{output_format}")


def plot_heatmap_streaming(stats: Dict, variant: str, timestamp: str, 
                          output_dir: Path, output_format: str, verbose: bool):
    """Streaming-optimierte Position-Heatmap aus aggregierten Daten"""
    
    if verbose:
        print("🗺️ Erstelle Streaming-optimierte Position-Heatmap...")
    
    position_data = stats.get('position_data', {})
    
    if not position_data:
        print("⚠️  Keine Position-Daten für Heatmap verfügbar")
        return
    
    setup_plot_style()
    
    # Konvertiere Position-Daten zu Listen
    positions = list(position_data.keys())
    counts = list(position_data.values())
    
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]
    
    # Erstelle Plot
    fig, ax = plt.subplots(figsize=PLOT_CONFIG['figsize'])
    
    # Logarithmische Farbskalierung für bessere Sichtbarkeit
    counts_log = np.log1p(counts)  # log1p(x) = log(1+x) für bessere Darstellung bei 0
    
    # Scatter plot mit Größe proportional zu Häufigkeit
    scatter = ax.scatter(
        x_coords, y_coords, 
        c=counts_log, 
        s=np.clip(counts_log * 20, 10, 200),  # Größe zwischen 10 und 200
        cmap='plasma',
        alpha=0.7,
        edgecolors='white',
        linewidth=0.5
    )
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Visit Frequency (log scale)', rotation=270, labelpad=20)
    
    # Title mit Statistiken
    total_visits = sum(counts)
    unique_positions = len(positions)
    title = create_plot_title(variant, timestamp, 
                            f"Position Heatmap ({unique_positions:,} unique positions, {total_visits:,} total visits)")
    
    apply_plot_styling(ax, title, "X Position", "Y Position")
    
    # Grid für bessere Orientierung
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Speichere Plot
    filename = f"{variant}_{timestamp}_heatmap_streaming"
    save_plot(fig, output_dir, filename, [output_format])
    
    plt.close(fig)
    
    if verbose:
        print(f"   Streaming Heatmap erstellt: {filename}.{output_format}")
        print(f"   Eindeutige Positionen: {unique_positions:,}")
        print(f"   Gesamtbesuche: {total_visits:,}")


def main():
    """
    Hauptfunktion für Streaming-basierte Verarbeitung
    
    Erstellt beide erforderlichen Visualisierungen:
    - Reward-Plot mit Reservoir-Sampling
    - Position-Heatmap mit inkrementeller Aggregation
    """
    parser = argparse.ArgumentParser(
        description="Streaming-basierte Visualisierung massiver Agent-Trainingsdaten (alle JSON-Dateien)"
    )
    parser.add_argument("--variant", choices=["v1", "v2", "v3", "v4"], required=True)
    parser.add_argument("--experiment-name", type=str, default=None, 
                       help="Name des Experiment-Subdirectories (z.B. 'v4_production' statt 'v4')")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default="png")
    parser.add_argument("--smooth-window", type=int, default=100)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    
    args = parser.parse_args()
    
    try:
        # Streaming-basierte Datenverarbeitung
        df, experiment_dir, timestamp, stats = process_dataset_streaming(
            args.variant, args.max_files, args.verbose, args.experiment_name
        )
        
        # Ausgabe-Verzeichnis
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = experiment_dir / "plots"
        
        # Erstelle Plots
        plot_rewards_streaming(
            df, args.variant, timestamp, output_dir, 
            args.smooth_window, args.format, stats, args.verbose
        )
        
        plot_heatmap_streaming(
            stats, args.variant, timestamp, output_dir, 
            args.format, args.verbose
        )
        
        print(f"✅ Streaming-Visualisierung abgeschlossen!")
        print(f"📊 Verarbeitet: {stats['total_entries']:,} Einträge aus {args.variant}")
        print(f"🎯 Vollständiger Step-Bereich: {stats['step_range'][0]:,} bis {stats['step_range'][1]:,}")
        print(f"🗺️ Position-Daten: {stats['position_count']:,} eindeutige Positionen")
        print(f"📁 Plots gespeichert in: {output_dir}")
        
    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
