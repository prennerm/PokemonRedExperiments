# Visualization Guidelines for Agent Training Analysis

## ⚠️ KRITISCHE REGEL: Vollständige Datenanalyse
**ALLE vorhandenen JSON-Dateien eines Training-Runs MÜSSEN verarbeitet werden!**
- Niemals JSON-Dateien auslassen oder ignorieren
- Jeder Trainings-Run kann Millionen von Steps enthalten
- Fehlende Daten verschleiern kritische Trainingsphasen
- Bei Performance-Problemen: **Streaming & Intelligent Sampling**, nicht Daten weglassen

## 🌊 Streaming-Architecture für massive Datasets (500+ Dateien, >27GB)

### **Streaming mit Reservoir Sampling**
1. **File-by-File Processing**: Sequenzielle Verarbeitung mit sofortiger Aggregation
2. **Reservoir Sampling**: Mathematisch garantierte Gleichverteilung über GESAMTEN Zeitraum
3. **Inkrementelle Aggregation**: Rolling Statistics ohne Full-DataFrame
4. **Memory-Bounded**: Nie mehr als 10k Samples + Rolling Stats im RAM

### **Konzept: StreamingProcessor Class**
- **Reservoir Buffer**: Maximal 10.000 repräsentative Samples im Speicher
- **Rolling Statistics**: Step-Range und Position-Counts werden live getrackt
- **File-Level Sampling**: Pro Datei nur 100 zeitlich gleichverteilte Einträge
- **Garbage Collection**: Explizite Speicherfreigabe nach jeder Datei

### **Konzept: Reservoir Sampling Algorithm**
- **Gleichverteilung garantiert**: Mathematisch korrekte Repräsentation über gesamten Zeitraum
- **Memory-konstant**: Buffer-Größe bleibt immer bei 10.000 Einträgen
- **Zufälliger Ersatz**: Neue Einträge ersetzen zufällig alte bei vollem Buffer

### **Konzept: Inkrementelle Aggregation**
- **Step-Range Tracking**: Min/Max Steps werden live aktualisiert ohne DataFrame
- **Position-Heatmap**: Koordinaten-Counts werden direkt aggregiert
- **Memory-Effizienz**: Nur Statistiken gespeichert, keine Rohdaten

## Environment Setup

### 0. Python Environment
```bash
# Create and activate environment
python -m venv poke_viz
poke_viz\Scripts\activate

# Install required packages
pip install pandas matplotlib seaborn numpy
```

### 0.1 Required Libraries
```
pandas==2.2.3          # DataFrame operations (nur für finale 10k Samples)
matplotlib==3.9.2       # Plotting
seaborn==0.13.2         # Styling
numpy==2.0.2            # Numerical operations
```

## Requirements

### 1. Core Architecture: StreamingProcessor Class
**Konzept**: Zentrale Klasse für memory-effiziente Verarbeitung massiver Datasets
- **Reservoir Buffer**: Hält maximal 10.000 repräsentative Samples
- **Rolling Statistics**: Trackt Step-Range und Position-Counts ohne DataFrame
- **File Processing**: Verarbeitet JSON-Dateien sequenziell mit sofortiger Aggregation
- **Memory Management**: Explizite Garbage Collection nach jeder Datei

### 2. Streaming Pipeline Konzept
**Ablauf**: Sequenzielle Verarbeitung aller JSON-Dateien ohne Memory-Explosion
- **File Discovery**: Finde alle JSON-Dateien im Experiment-Verzeichnis
- **StreamingProcessor Init**: Initialisiere Reservoir Buffer und Statistics
- **Sequential Processing**: Verarbeite jede Datei einzeln mit File-Level Sampling
- **Final Extraction**: Konvertiere Reservoir Samples zu DataFrame für Visualisierung

### 3. Required Visualizations

#### 3.1 Reward Plot (Streaming-optimiert)
**Konzept**: Memory-effiziente Reward-Visualisierung mit repräsentativen Samples
- **Datenquelle**: 10.000 Reservoir Samples über GESAMTEN Trainingszeitraum
- **X-Achse**: Globale Training Steps (`total_steps` falls verfügbar, sonst `step`)
- **Y-Achse**: Reward-Komponenten (total, event, level, heal, badge, explore, dead, stuck)
- **Adaptive Glättung**: Window-Größe basierend auf tatsächlicher Sample-Anzahl
- **Titel-Info**: Zeigt Sampling-Ratio für Transparenz

#### 3.2 Position Heatmap (Inkrementell aggregiert)
**Konzept**: Memory-effiziente Heatmap aus aggregierten Position-Counts
- **Datenquelle**: Inkrementell aufgebaute Position-Counts (Dictionary)
- **Aggregation**: Koordinaten → Visit-Count Mapping ohne Rohdaten-Speicherung
- **Visualisierung**: Scatter-Plot mit logarithmischer Farbskalierung
- **Memory-Effizienz**: Nur eindeutige Koordinaten und Counts gespeichert

### 4. Command-Line Usage
```bash
# Basic usage
python visualize_training.py --variant v4

# With additional options
python visualize_training.py --variant v1 --output-dir plots/ --format png
python visualize_training.py --variant v2 --smooth-window 50 --verbose
```

### 5. Output Requirements
- **File naming**: `{variant}_{timestamp}_rewards.png` and `{variant}_{timestamp}_heatmap.png`
- **Output directory**: Create and save plots in a `plots/` subdirectory within the used training run (same level as `checkpoints/`, `json_logs/`, `tensorboard/` subdirectories)
- **Multiple formats**: Support PNG, PDF, SVG
- **High resolution**: Suitable for publications (300 DPI minimum)

## Implementation Guidelines

### 6. Code Structure
**Architektur**: Streaming-basiertes Design mit klarer Modularisierung
- **Haupt-Skript**: StreamingProcessor Integration mit CLI
- **StreamingProcessor**: Reservoir Sampling und inkrementelle Aggregation
- **Pipeline-Funktion**: Orchestrierung der sequenziellen Datei-Verarbeitung
- **Plotting-Funktionen**: Optimierte Visualisierung für Streaming-Daten
- **Utils-Module**: Hilfsfunktionen für File-Discovery und Styling

### 7. Main Function
**CLI-Integration**: Standard Command-Line Interface mit Streaming-Unterstützung
- **Argument-Parsing**: Variant, Output-Dir, Format, Smooth-Window, Verbose
- **Streaming-Pipeline**: Aufruf der dataset-weiten Streaming-Verarbeitung
- **Output-Management**: Automatische Plot-Generierung im Experiment-Verzeichnis
- **Error-Handling**: Vollständige Traceback-Ausgabe für Debugging
- **Success-Feedback**: Anzeige von verarbeiteten Einträgen und Step-Bereich

### 8. Memory Management & Performance
**Memory-Bounded Processing**: Konstante Memory-Nutzung unabhängig von Dataset-Größe
- **Garbage Collection**: Explizite Speicherfreigabe nach jeder JSON-Datei
- **Reservoir Sampling Limits**: Buffer nie größer als 10.000 Einträge
- **File-Level Sampling**: Maximal 100 Samples pro Datei für gleichmäßige Verteilung
- **Adaptive Glättung**: Window-Größe basierend auf verfügbaren Datenpunkten

### 9. Expected Output Behavior
**Transparente Fortschritts-Anzeige**: Vollständige Übersicht über Verarbeitungsfortschritt
- **File-Progress**: Anzeige aktueller Datei und Gesamtfortschritt
- **Intermediate Updates**: Regelmäßige Status-Updates bei vielen Dateien
- **Final Statistics**: Vollständige Übersicht über verarbeitete Daten und Sampling-Ratio
- **Step-Range Validation**: Anzeige des tatsächlich abgedeckten Trainingszeitraums

### 10. Validation & Quality Control
**Qualitätssicherung**: Mehrschichtige Validierung der Streaming-Verarbeitung
- **Step-Range Tracking**: Kontinuierliche Überwachung des abgedeckten Zeitraums
- **Sampling-Ratio Monitoring**: Transparenz über Verhältnis Sample/Gesamt-Daten
- **Memory Usage Tracking**: Optional verfügbare Speicherverbrauchs-Überwachung
- **Data Validation**: Prüfung auf erforderliche Spalten und Datenqualität

## Example Output Structure
```
experiments/v4/20250726_130714/
├── checkpoints/
├── json_logs/
├── tensorboard/
└── plots/
    ├── v4_20250726_130714_rewards.png      # Generated reward plot
    └── v4_20250726_130714_heatmap.png      # Generated heatmap plot
```

## Quality Standards
- **Code quality**: Follow PEP 8 style guidelines
- **Documentation**: Comprehensive docstrings for all functions
- **Testing**: Include sample data loading tests
- **Performance**: Handle large datasets efficiently
- **Compatibility**: Work with both old and new JSON log formats

## Integration with Existing Tools
- **Consistency**: Use the same data loading logic as existing utilities
- **JSON Format**: Supports nested JSON structure with `total_steps` (global) and `step` (environment)
- **Reusability**: Design functions to be importable by other scripts
- **Configuration**: Follow standard command-line patterns

## Future Extensions
- **Interactive plots**: Option to generate interactive Plotly visualizations
- **Comparison mode**: Side-by-side comparison of multiple variants
- **Animation**: Time-lapse visualization of agent movement
- **Advanced metrics**: Additional statistical analysis plots
- **Report generation**: Automatic PDF report with all visualizations
- **LD logging & transparency**: 
  1. Roh-Delta (λ-discrepancy) pro Update loggen (Term-Wert, Anteil am Gesamt-Loss).  
  2. Rolling Metriken (z.B. gleitender Mittelwert, Varianz) speichern.  
  3. LD-Verteilung (Histogramm / Heatmap) über Zeitfenster sammeln.  
  4. Korrelation zwischen LD und Reward-/Badge-Zuwächsen berechnen (Hinweis, ob LD beim Fortschritt hilft).  
  5. Warn-Flags im Log setzen, sobald LD definierte Schwellen überschreitet (Marker im Plot).  
  6. LD-Kennzahlen in den finalen Report integrieren (Mittelwert, Max, Top-LD-Zonen etc.).

### Overlay-Prototype (Heatmap + World Map) – Status Oktober 2025
- *Prototyp*: `notebooks/overlay_heatmap_prototype.py`  
  Lädt das aktuellste Sampling (nutzt `--max-data-points`) und legt eine große Welt-Map unter die Heatmap. Output unter `experiments/<run>/plots/overlay_prototype/`.
- *Asset*: `data/pokemon_red_global_map.png` (7200 × 7200). Enthält Overworld + Interiors, aber die Offsets stimmen nicht automatisch mit `MAP_DATA` überein.
- *Beobachtungen*:  
  - Overworld-Bereiche lassen sich grob treffen, Interior-Maps (Player’s House, Oak’s Lab …) landen an falschen Stellen.  
  - Transparenz (50 %) funktioniert; Schwarz verschwindet, Karte scheint durch.  
  - Doppelter Timestamp-Ordner entfernt.
- *Todo / Ideen*:  
  1. Kalibrierung der großen Karte (einheitlicher Maßstab + Offsets).  
  2. Ggf. getrennte Sprite-Sammlung für Innenräume anlegen.  
  3. Kleine Tabelle `map_id -> (pixel_x, pixel_y, width, height)` pflegen oder automatisch generieren.  
  4. Danach den Prototyp in das Paket (z.B. `plot_position_heatmap(..., overlay=True)`) überführen.  
  5. Optional: Falls keine global passende Map vorhanden ist, Outdoor-Overlay aktiv lassen, Interiors ausblenden.

## Success Criteria
1. Script successfully loads data from any variant (v1-v4)
2. Generates publication-quality reward and heatmap visualizations
3. Handles JSON log formats with proper step tracking (`total_steps` priority)
4. Provides clear error messages for troubleshooting
5. Completes analysis of typical training run within 30 seconds
6. Outputs are consistent and reproducible across different runs
