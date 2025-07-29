#!/usr/bin/env python3
"""
debug_logs.py - Debugging-Skript für JSON-Logs der Agent-Trainings

Usage:
    python debug_logs.py --variant v4
    python debug_logs.py --variant v1 --verbose
"""

import argparse
import json
import pandas as pd
from pathlib import Path
import glob
from datetime import datetime
import sys

def find_latest_json_files(base_path: Path, pattern: str = "*/json_logs/stats_*.json") -> list:
    """Findet die neuesten JSON-Log-Dateien in einem Variant-Ordner"""
    search_pattern = str(base_path / pattern)
    json_files = glob.glob(search_pattern, recursive=True)
    json_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    return json_files

def analyze_json_structure(file_path: Path, verbose: bool = False):
    """Analysiert die Struktur einer JSON-Datei"""
    print(f"\n=== ANALYSIERE: {file_path.name} ===")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"Dateigröße: {file_path.stat().st_size / 1024:.1f} KB")
        print(f"Datentyp: {type(data)}")
        
        if isinstance(data, list):
            print(f"Anzahl Einträge: {len(data)}")
            
            if data:
                first_entry = data[0]
                print(f"Typ des ersten Eintrags: {type(first_entry)}")
                
                if isinstance(first_entry, dict):
                    print(f"Schlüssel im ersten Eintrag: {list(first_entry.keys())}")
                    
                    # Prüfe auf verschachtelte Struktur
                    nested_keys = []
                    flat_keys = []
                    
                    for key, value in first_entry.items():
                        if isinstance(value, dict):
                            nested_keys.append(key)
                            if verbose:
                                print(f"  {key} (nested): {list(value.keys())}")
                        else:
                            flat_keys.append(key)
                            if verbose:
                                print(f"  {key}: {type(value)} = {value}")
                    
                    print(f"Verschachtelte Schlüssel: {nested_keys}")
                    print(f"Flache Schlüssel: {flat_keys}")
                    
                    # Test DataFrame-Erstellung
                    return test_dataframe_creation(data, verbose)
                
        elif isinstance(data, dict):
            print(f"Dictionary mit Schlüsseln: {list(data.keys())}")
            
        return None
            
    except Exception as e:
        print(f"❌ Fehler beim Laden der Datei: {e}")
        return None

def test_dataframe_creation(data: list, verbose: bool = False):
    """Testet verschiedene Methoden zur DataFrame-Erstellung"""
    print(f"\n=== DATAFRAME-ERSTELLUNG TEST ===")
    
    # Test 1: Direkte Erstellung (für alte flache Struktur)
    try:
        df_direct = pd.DataFrame(data)
        print(f"✅ Direkte Erstellung: {df_direct.shape} - Spalten: {len(df_direct.columns)}")
        
        if verbose:
            print(f"Spalten: {list(df_direct.columns)}")
            print(f"Datentypen:\n{df_direct.dtypes}")
        
        # Prüfe auf verschachtelte Daten
        nested_cols = []
        for col in df_direct.columns:
            if df_direct[col].dtype == 'object':
                sample = df_direct[col].iloc[0] if len(df_direct) > 0 else None
                if isinstance(sample, dict):
                    nested_cols.append(col)
        
        if nested_cols:
            print(f"Verschachtelte Spalten gefunden: {nested_cols}")
            return test_nested_flattening(data, verbose)
        else:
            print("Keine verschachtelten Daten - DataFrame ist bereits flach")
            return df_direct
            
    except Exception as e:
        print(f"❌ Direkte Erstellung fehlgeschlagen: {e}")
        return test_nested_flattening(data, verbose)

def test_nested_flattening(data: list, verbose: bool = False):
    """Testet das Flatten von verschachtelten JSON-Strukturen"""
    print(f"\n=== FLATTENING-TEST ===")
    
    try:
        first_entry = data[0]
        
        # Erstelle flache Struktur mit rekursivem Flattening
        flattened_data = []
        for entry in data:
            flat_entry = flatten_dict(entry)
            flattened_data.append(flat_entry)
        
        df_flattened = pd.DataFrame(flattened_data)
        print(f"✅ Flattening erfolgreich: {df_flattened.shape}")
        
        if verbose:
            print(f"Neue Spalten: {list(df_flattened.columns)}")
            print(f"Erste 3 Zeilen:")
            print(df_flattened.head(3))
        
        # Prüfe spezifische Spalten
        check_columns = ['x', 'y', 'step', 'position_x', 'position_y']
        reward_cols = [col for col in df_flattened.columns if 'reward' in col.lower()]
        
        available_pos_cols = [col for col in check_columns if col in df_flattened.columns]
        print(f"Position/Step Spalten: {available_pos_cols}")
        print(f"Reward Spalten: {reward_cols}")
        
        # Prüfe Reward-Werte
        if reward_cols:
            try:
                non_zero_rewards = df_flattened[reward_cols].sum()
                print(f"Reward-Summen:")
                for reward_type, total in non_zero_rewards.items():
                    print(f"  {reward_type}: {total}")
            except Exception as sum_error:
                print(f"⚠️ Fehler beim Summieren der Rewards: {sum_error}")
                # Zeige Datentypen der Reward-Spalten
                print("Reward-Spalten Datentypen:")
                for col in reward_cols[:5]:  # Nur die ersten 5
                    dtype = df_flattened[col].dtype
                    sample = df_flattened[col].iloc[0] if len(df_flattened) > 0 else None
                    print(f"  {col}: {dtype} - Sample: {sample}")
        
        return df_flattened
        
    except Exception as e:
        print(f"❌ Flattening fehlgeschlagen: {e}")
        return None

def flatten_dict(d, parent_key='', sep='_'):
    """
    Rekursives Flattening eines verschachtelten Dictionaries
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def main():
    parser = argparse.ArgumentParser(description="Debug JSON-Logs der Agent-Trainings")
    parser.add_argument("--variant", choices=["v1", "v2", "v3", "v4"], required=True,
                       help="Agent-Variante (v1, v2, v3, v4)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Detaillierte Ausgabe")
    parser.add_argument("--max-files", type=int, default=3,
                       help="Maximale Anzahl zu analysierender Dateien")
    
    args = parser.parse_args()
    
    # Pfade
    base_path = Path("experiments") / args.variant
    
    if not base_path.exists():
        print(f"❌ Pfad nicht gefunden: {base_path}")
        sys.exit(1)
    
    print(f"🔍 Suche JSON-Dateien in: {base_path}")
    
    # Finde JSON-Dateien
    json_files = find_latest_json_files(base_path)
    
    if not json_files:
        print(f"❌ Keine JSON-Dateien gefunden in {base_path}")
        sys.exit(1)
    
    print(f"📁 Gefunden: {len(json_files)} JSON-Dateien")
    
    # Analysiere die neuesten Dateien
    files_to_analyze = json_files[:args.max_files]
    
    results = []
    for file_path in files_to_analyze:
        file_path = Path(file_path)
        df = analyze_json_structure(file_path, args.verbose)
        if df is not None:
            results.append((file_path.name, df))
    
    # Zusammenfassung
    print(f"\n=== ZUSAMMENFASSUNG ===")
    print(f"Erfolgreich analysierte Dateien: {len(results)}")
    
    if results:
        print("\n✅ ERFOLG: JSON-Logs können verarbeitet werden!")
        print("Die neuen strukturierten JSON-Logs funktionieren korrekt.")
        
        # Zeige Statistiken der neuesten Datei
        latest_name, latest_df = results[0]
        print(f"\nStatistiken der neuesten Datei ({latest_name}):")
        print(f"  Shape: {latest_df.shape}")
        print(f"  Spalten: {len(latest_df.columns)}")
        
        reward_cols = [col for col in latest_df.columns if 'reward' in col.lower()]
        if reward_cols:
            print(f"  Reward-Komponenten: {len(reward_cols)}")
            total_rewards = latest_df[reward_cols].sum().sum()
            print(f"  Gesamte Rewards: {total_rewards}")
        
        # Prüfe ob das ursprüngliche Problem behoben ist
        if reward_cols and total_rewards > 0:
            print("\n🎉 Das Reward-Problem scheint behoben zu sein!")
        elif reward_cols and total_rewards == 0:
            print("\n⚠️  Reward-Spalten vorhanden, aber alle Werte sind 0")
        else:
            print("\n❌ Keine Reward-Spalten gefunden")
    
    else:
        print("❌ Keine Dateien konnten erfolgreich verarbeitet werden")

if __name__ == "__main__":
    main()
