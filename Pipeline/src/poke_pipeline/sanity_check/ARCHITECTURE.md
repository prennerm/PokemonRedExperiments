# Blueprint: Multi-Agent Benchmark Suite - Konsolidierter Blueprint

## Projekt-Übersicht
Das **Multi-Agent Benchmark Suite** ist ein modulares Framework zum systematischen Vergleich von drei verschiedenen Reinforcement Learning-Agenten auf Standard-Gym-Environments.  
Das System trainiert jeden Agenten auf jedem Environment, extrahiert Learning Curves aus TensorBoard-Logs, generiert Vergleichsplots und erstellt umfassende Reports.  
Ziel ist es, die Leistung verschiedener RL-Algorithmen wissenschaftlich fundiert zu vergleichen, ohne Reward-Shaping oder andere Modifikationen der ursprünglichen Environments.

## Wichtige Libraries und Abhängigkeiten

### Kern-Libraries
- **`gymnasium`** (≥0.28.0): Moderne Gym-Interface für RL-Environments
  - Standardisierte Environment-API (make, reset, step, render)
  - Observation/Action-Space-Definitionen
  - Wrapper-System für Environment-Modifikationen
  - Beeinflusst: `core/environment.py`, `utils/wrappers.py`

- **`stable-baselines3`** (≥2.0.0): State-of-the-Art RL-Algorithmen
  - PPO, RecurrentPPO Implementierungen
  - Standardisierte Training/Evaluation-Pipeline
  - VecEnv-Support für paralleles Training
  - Beeinflusst: `agents/`, `core/evaluator.py`, `core/runner.py`

- **`sb3-contrib`** (≥2.0.0): Zusätzliche SB3-Algorithmen
  - RecurrentPPO für LSTM-basierte Policies
  - Erweiterte Policy-Typen (MultiInputLstmPolicy)
  - Beeinflusst: `agents/lstm.py`

### Monitoring & Logging
- **`tensorboard`** (≥2.12.0): Training-Monitoring und Visualisierung
  - Scalar-Logging für Learning Curves
  - Event-File-Format für Metriken-Speicherung
  - Beeinflusst: `utils/tensorboard.py`, `utils/callbacks.py`

- **`wandb`** (optional): Erweiterte Experiment-Tracking
  - Cloud-basiertes Logging und Vergleiche
  - Hyperparameter-Sweeps (zukünftige Erweiterung)

### Visualisierung
- **`matplotlib`** (≥3.5.0): Plot-Generierung
  - Learning Curves und Vergleichsplots
  - Publikationsreife Visualisierungen
  - Beeinflusst: `visualization/plotter.py`

- **`seaborn`** (≥0.11.0): Statistische Plots
  - Erweiterte Styling-Optionen
  - Heatmaps für Agent-Vergleiche
  - Beeinflusst: `visualization/plotter.py`

### Datenverarbeitung
- **`numpy`** (≥1.21.0): Numerische Berechnungen
  - Array-Operationen für Observations
  - Statistik-Berechnungen für Evaluation
  - Beeinflusst: Alle Module

- **`pandas`** (≥1.3.0): Datenanalyse und -export
  - CSV-Export für Reports
  - Aggregation von Benchmark-Ergebnissen
  - Beeinflusst: `core/reporter.py`

### Auswirkungen auf die Pipeline-Architektur

1. **Gymnasium-Kompatibilität**: Alle Environments müssen das neue `gymnasium.Env` Interface verwenden
2. **SB3-Integration**: Agent-Implementierungen folgen der SB3-Architektur (Policy, VecEnv, Callbacks)
3. **TensorBoard-Integration**: Standardisierte Metric-Namen für konsistente Extraktion
4. **Modularität**: Library-spezifische Funktionalität ist in separaten Modulen isoliert
5. **Hardcodierte Konfigurationen**: Environment- und Agent-Konfigurationen sind direkt im Code definiert für einfache Wartung

## Relevante Test-Environments

Das Sanity Check Suite verwendet eine sorgfältig ausgewählte Sammlung von Standard-Gymnasium-Environments, die verschiedene Aspekte von Reinforcement Learning-Algorithmen testen. Diese Environments sind direkt im Code definiert (`src/core/environment.py`):

### Core Test-Environments

#### **CartPole-v1**
- **Typ:** Klassisches Kontrolltask
- **Observation Space:** Kontinuierlich (4D: Position, Velocity, Angle, Angular Velocity)
- **Action Space:** Diskret (2 Aktionen: Links, Rechts)
- **Reward:** Dichte Belohnung (+1 pro Schritt)
- **Challenge:** Balancieren einer Stange auf einem Wagen
- **Testet:** Kontinuierliche Observations, dichte Rewards, Kontrollaufgaben

#### **Acrobot-v1**
- **Typ:** Underactuated Double-Pendulum Control
- **Observation Space:** Kontinuierlich (6D: cos(θ₁), sin(θ₁), cos(θ₂), sin(θ₂), θ̇₁, θ̇₂)
- **Action Space:** Diskret (3 Aktionen: −1, 0, +1 Torque am unteren Gelenk)
- **Reward:** −1 pro Zeitschritt bis zum Erreichen der Zielhöhe (State[0] > 1.0)
- **Challenge:** Erfordert Momentum-Aufbau in unteraktuiertem System
- **Testet:** Continuous Dynamics, Sparse Rewards, Momentum Control

#### **LunarLander-v3**
- **Typ:** Komplexe Dynamics mit Shaped Rewards
- **Observation Space:** Kontinuierlich (8D: Position, Velocity, Angle, etc.)
- **Action Space:** Diskret (4 Aktionen: Nothing, Left, Main, Right)
- **Reward:** Shaped Rewards für Landung, Fuel-Verbrauch, Crashes
- **Challenge:** Kontrollierte Landung eines Lunar Landers
- **Testet:** Komplexe Dynamics, Shaped Rewards, Multi-Objective-Optimierung

#### **FrozenLake-v1**
- **Typ:** Stochastische Gridworld
- **Observation Space:** Diskret (16 States in 4x4 Grid)
- **Action Space:** Diskret (4 Aktionen: Up, Down, Left, Right)
- **Reward:** Sparse (0 für normale Schritte, +1 für Ziel)
- **Challenge:** Navigation durch rutschiges Terrain zum Ziel
- **Testet:** Discrete Observations, Stochastic Transitions, Sparse Rewards

#### **Acrobot-v1**
- **Typ:** Underactuated Double-Pendulum Control
- **Observation Space:** Kontinuierlich (6D: cos(θ₁), sin(θ₁), cos(θ₂), sin(θ₂), θ̇₁, θ̇₂)
- **Action Space:** Diskret (3 Aktionen: −1, 0, +1 Torque am unteren Gelenk)
- **Reward:** −1 pro Zeitschritt bis zum Erreichen der Zielhöhe (State[0] > 1.0)
- **Challenge:** Erfordert Momentum-Aufbau in unteraktuiertem System
- **Testet:** Continuous Dynamics, Sparse Rewards, Momentum Control

#### **Acrobot-v1 (Partial Observability)**
- **Typ:** Wie Acrobot-v1, aber ohne direkte Geschwindigkeitsinformation
- **Observation Space:** Kontinuierlich (4D: cos(θ₁), sin(θ₁), cos(θ₂), sin(θ₂)) — **Gelenkgeschwindigkeiten (θ̇₁, θ̇₂) verborgen**
- **Action Space:** Diskret (3 Aktionen: −1, 0, +1 Torque am unteren Gelenk)
- **Reward:** Wie oben (−1 pro Schritt bis Ziel)
- **Challenge:** Partielle Beobachtung erschwert Zustandsinferenz → benötigt Memory (LSTM) und λ-Diskrepanz
- **Testet:** Fähigkeit von LSTM und λ-Diskrepanz, fehlende Informationen zu kompensieren

### Environment-spezifische Konfigurationen

Jedes Environment hat optimierte Hyperparameter für die verschiedenen Agenten:

- **Standard PPO**: Environment-spezifische Hyperparameter in `get_environment_specific_hyperparams()`
- **Advanced Sanity Check**: Erweiterte Konfigurationen mit Success-Thresholds und Wrapper-Definitionen
- **Observation Wrapper**: Konvertierung zu Dictionary-Observations für LSTM-Kompatibilität

### Environment-Modifikationen

Zur besseren Evaluation der verschiedenen Agenten werden gezielt minimale Modifikationen angebracht:

#### **Acrobot-v1 (Partial Observability) Wrapper**
- **Methode:** Observation Space Reduktion durch Wrapper
- **Implementierung:** `PartialObservabilityWrapper` in `utils/wrappers.py`
- **Modifikation:** Entfernung der Geschwindigkeitskomponenten (θ̇₁, θ̇₂) aus dem Observation Space
- **Transformation:** 6D → 4D (cos(θ₁), sin(θ₁), cos(θ₂), sin(θ₂))
- **Zweck:** Test der Memory-Fähigkeiten von LSTM und λ-Diskrepanz Agenten
- **Status:** Nur für Acrobot-v1 (Partial) aktiv

### Bewertungskriterien

- **Success Thresholds**: Minimale Performance-Schwellenwerte für jeden Agent
- **Baseline Performance**: Random Policy Performance als Referenz
- **Training Steps**: Environment-spezifische Trainingszeiten
- **Evaluation Metrics**: Standardisierte Metriken für konsistente Vergleiche

Diese Environment-Auswahl gewährleistet umfassende Tests verschiedener RL-Herausforderungen:
- Discrete vs. Continuous Observations
- Dense vs. Sparse Rewards
- Deterministic vs. Stochastic Dynamics
- Simple vs. Complex State Spaces
- Full vs. Partial Observability (Memory Requirements)

### Installation
```bash
pip install gymnasium>=0.28.0 stable-baselines3>=2.0.0 sb3-contrib>=2.0.0
pip install tensorboard>=2.12.0 matplotlib>=3.5.0 seaborn>=0.11.0
pip install numpy>=1.21.0 pandas>=1.3.0
```

### Struktur
```plaintext
src/
├── main.py                  # CLI-Entry Point
├── core/                    # Kern-Funktionalität
│   ├── __init__.py          # Exports der Core-Module
│   ├── environment.py       # Environment-Factory und -Management
│   ├── evaluator.py         # Policy-Evaluation und Metriken
│   ├── reporter.py          # Text/JSON/CSV-Report-Generierung
│   └── runner.py            # Hauptlogik für Training und Vergleiche
├── visualization/           # Plot-Generierung
│   ├── __init__.py          # Exports der Visualization-Module
│   └── plotter.py           # Learning Curves und Vergleichsplots
├── agents/                  # Agent-Implementierungen
│   ├── __init__.py          # Exports der Agent-Module
│   ├── base.py              # BaseAgent Interface
│   ├── standard_ppo.py      # Standard PPO (Baseline)
│   ├── lstm.py              # LSTM-erweiterte Version (v2)
│   └── ld.py                # Lambda-Discrepancy Version (v3)
└── utils/                   # Hilfsfunktionen
    ├── __init__.py          # Exports der Utility-Module
    ├── paths.py             # Pfad-Management und Session-Verzeichnisse
    ├── callbacks.py         # Training-Callbacks für Monitoring
    ├── tensorboard.py       # TensorBoard-Datenextraktion
    └── wrappers.py          # Environment-Wrapper für Standardisierung
```
### Detaillierte Datei-Beschreibungen

#### `src/main.py`  
**Zweck:** CLI-Interface für das gesamte System  
- `main()` – Hauptfunktion mit `argparse`  
- Argument-Parsing für verschiedene Ausführungsmodi (`--all`, `--env`, `--agents`, `--compare`, `--debug`)  
- Initialisierung und Aufruf von `SanityCheckRunner`

#### `src/core/__init__.py`  
**Zweck:** Exports der Core-Module  
- Exports: `SanityCheckRunner`, `EnvironmentFactory`, `PolicyEvaluator`, `ComparisonReporter`

#### `src/core/environment.py`  
**Zweck:** Environment-Factory und -Management  
- **Klasse:** `EnvironmentFactory`  
  - `make_env()` – Einzelnes Environment mit Wrappern erstellen  
  - `make_vec_env()` – Vektorisiertes Environment erstellen  
  - `get_hyperparameters()` – Environment-spezifische Hyperparameter laden  
  - `get_available_environments()` – Liste verfügbarer Environments zurückgeben

#### `src/core/evaluator.py`  
**Zweck:** Policy-Evaluation und Metriken-Berechnung  
- **Klasse:** `PolicyEvaluator`  
  - `evaluate_trained_policy()` – Evaluierung trainierter Modelle  
  - `evaluate_random_policy()` – Baseline-Evaluation mit Random-Policy  
  - `detailed_evaluation()` – Umfassende Evaluation mit Modell-Informationen  
  - `compare_agents()` – Vergleich mehrerer Agenten auf einem Environment

#### `src/core/reporter.py`  
**Zweck:** Text/JSON/CSV-Report-Generierung  
- **Klassen:**  
  - `ComparisonReporter` – Generierung von Vergleichsreports  
  - `MarkdownReporter` – Markdown-Report-Generierung  
- **Methoden:**  
  - `generate_comparison_report()` – Detaillierter Textreport  
  - `generate_json_report()` – JSON-Export  
  - `generate_summary_table()` – CSV-Tabelle  
  - `generate_markdown_report()` – Markdown-Report mit Plot-Referenzen

#### `src/core/runner.py`  
**Zweck:** Hauptlogik für Training und Vergleiche  
- **Klasse:** `SanityCheckRunner`  
- **Methoden:**  
  - `run_single()` – Einzelner Agent auf einzelnem Environment  
  - `run_comparison()` – Vergleich mehrerer Agenten auf einem Environment  
  - `run_comprehensive()` – Vollständiger Benchmark über alle Kombinationen  
  - `_load_agent()` – Dynamisches Laden von Agent-Implementierungen  
  - `_save_results()` – Speicherung aller Ergebnisse und Reports

#### `src/visualization/__init__.py`  
**Zweck:** Exports der Visualization-Module  
- Exports: `LearningCurvePlotter`, `ComparisonPlotter`

#### `src/visualization/plotter.py`  
**Zweck:** Learning Curves und Vergleichsplots  
- **Klassen:**  
  - `LearningCurvePlotter` – Generierung von Learning Curve Plots  
  - `ComparisonPlotter` – Vergleichsplots zwischen Agenten  
- **Methoden:**  
  - `plot_learning_curves()` – Einzelnes Environment, alle Agenten  
  - `plot_comparison_matrix()` – Vergleichsmatrix über alle Environments  
  - `create_publication_plots()` – Publication-ready Plots  
  - `save_plots()` – Speicherung in verschiedenen Formaten

#### `src/agents/__init__.py`  
**Zweck:** Exports der Agent-Module  
- Exports: `BaseAgent`, `StandardPPOAgent`, `LSTMAgent`, `LDAgent`

#### `src/agents/base.py`  
**Zweck:** BaseAgent Interface und gemeinsame Funktionalität  
- **Klasse:** `BaseAgent`  
- **Methoden:**  
  - `create_model()` – Modell-Erstellung (abstract)  
  - `train()` – Training-Interface  
  - `evaluate()` – Evaluation-Interface  
  - `save_model()` – Modell-Speicherung  
  - `load_model()` – Modell-Laden

#### `src/agents/standard_ppo.py`  
**Zweck:** Standard PPO Agent (Baseline)  
- **Klasse:** `StandardPPOAgent`  
- **Methoden:**  
  - `create_model()` – PPO-Modell mit Standard-Hyperparametern

#### `src/agents/lstm.py`  
**Zweck:** LSTM-erweiterte Version (v2)  
- **Klasse:** `LSTMAgent`  
- **Methoden:**  
  - `create_model()` – LSTM-verstärktes Modell mit RecurrentPPO


#### `src/agents/ld.py`  
**Zweck:** Lambda-Discrepancy Version (v3)  
- **Klasse:** `LDAgent`  
- **Methoden:**  
  - `create_model()` – Modell mit Lambda-Discrepancy-Regularisierung und LSTM


#### `src/utils/__init__.py`  
**Zweck:** Exports der Utility-Module  
- Exports: `create_session_directory`, `EnhancedStatsCallback`, `extract_tensorboard_data`, `DictObsWrapper`

#### `src/utils/paths.py`  
**Zweck:** Pfad-Management und Session-Verzeichnisse  
- **Funktionen:**  
  - `create_session_directory()` – Timestamped Session-Verzeichnis erstellen  
  - `get_project_root()` – Projekt-Root-Verzeichnis finden

#### `src/utils/callbacks.py`  
**Zweck:** Training-Callbacks für Monitoring  
- **Klassen & Methoden:**  
  - `EnhancedStatsCallback` – Erweiterte Statistiken während des Trainings  
  - `ProgressCallback` – Fortschrittsanzeige für Konsole  
    - `_on_step()` – Callback bei jedem Trainingsschritt  
    - `_on_training_start()` – Callback bei Trainingsstart

#### `src/utils/tensorboard.py`  
**Zweck:** TensorBoard-Datenextraktion  
- **Funktionen:**  
  - `extract_tensorboard_data()` – Extraktion von Learning Curves aus TB-Logs  
  - `extract_all_tensorboard_data()` – Extraktion aller verfügbaren Metriken  
  - `find_tensorboard_logs()` – Suche nach TB-Log-Verzeichnissen

#### `src/utils/wrappers.py`  
**Zweck:** Environment-Wrapper für Standardisierung und Modifikationen  
- **Klassen & Methoden:**  
  - `DictObsWrapper` – Konvertierung zu Dictionary-Observations  
  - `MountainCarRewardShapingWrapper` – Potenzial-basiertes Reward Shaping für MountainCar  
  - `NoRewardShapingWrapper` – Dokumentation der No-Reward-Shaping-Regel  
    - `observation()` – Observation-Transformation  
    - `_convert_discrete_observation()` – Discrete-zu-Dict-Konvertierung  
    - `_convert_continuous_observation()` – Continuous-zu-Dict-Konvertierung  
    - `_get_potential()` – Potenzial-Funktion für Reward Shaping
