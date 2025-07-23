# Multi-Agent Benchmark Suite

Ein systematisches Framework zum Vergleich von Reinforcement Learning-Agenten auf Standard-Gymnasium-Environments.

## Überblick

Dieses System trainiert und vergleicht **drei verschiedene RL-Agenten** auf vier standardisierten Test-Environments. Es generiert automatisch Learning Curves, Vergleichsplots und detaillierte Reports für wissenschaftliche Analysen.

### Verfügbare Agenten

| Agent | Beschreibung | Besonderheiten |
|-------|-------------|----------------|
| **Standard PPO** | Baseline-Implementation | Stable-Baselines3, MLP-Policy |
| **LSTM Agent** | Memory-Enhanced PPO | RecurrentPPO, LSTM-Layers |
| **LD Agent** | Lambda-Discrepancy PPO | LSTM + λ-Discrepancy Loss |

### Test-Environments

| Environment | Typ | Challenge |
|-------------|-----|-----------|
| **CartPole-v1** | Klassische Kontrolle | Balance, dichte Rewards |
| **Acrobot-v1** | Underactuated Control | Momentum-basierte Dynamics |
| **LunarLander-v3** | Komplexe Dynamics | Multi-Objective, Shaped Rewards |
| **FrozenLake-v1** | Stochastische Navigation | Discrete State, Sparse Rewards |
| **Acrobot-v1 (Partial)** | Partial Observability | Memory Requirements, LSTM/LD-Test |

## Schnellstart

### Installation

```bash
# Abhängigkeiten installieren
pip install gymnasium>=0.28.0 stable-baselines3>=2.0.0 sb3-contrib>=2.0.0
pip install tensorboard>=2.12.0 matplotlib>=3.5.0 seaborn>=0.11.0
pip install numpy>=1.21.0 pandas>=1.3.0

# Für LunarLander-v3 (Box2D Physics)
pip install swig
pip install gymnasium[box2d]

# Ins Projekt-Verzeichnis wechseln
cd src/
```

### Grundlegende Verwendung

```bash
# Verfügbare Optionen anzeigen
python main.py --list

# Einzelnen Agent trainieren
python main.py --agent standard_ppo --env CartPole-v1

# Alle Agenten auf einem Environment vergleichen
python main.py --compare --env LunarLander-v3

# Vollständigen Benchmark durchführen
python main.py --comprehensive
```

## Verwendungsszenarien

### 1. Einzelner Agent-Test
Testet einen spezifischen Agent auf einem Environment:

```bash
python main.py --agent lstm --env Acrobot-v1 --timesteps 100000
```

**Output:**
- Trainiertes Modell
- TensorBoard-Logs
- Evaluation-Ergebnisse
- Learning Curve Plot

### 2. Agent-Vergleich
Vergleicht alle drei Agenten auf einem Environment:

```bash
python main.py --compare --env FrozenLake-v1 --eval-episodes 50
```

**Output:**
- Drei trainierte Modelle
- Vergleichende Learning Curves
- Performance-Tabelle
- Detaillierter Vergleichsreport

### 3. Vollständiger Benchmark
Führt systematischen Vergleich über alle Kombinationen durch:

```bash
python main.py --comprehensive --timesteps 200000
```

**Output:**
- 15 trainierte Modelle (3 Agenten × 5 Environments)
- Umfassende Vergleichsmatrix
- Wissenschaftlicher Report
- Publikationsreife Plots

## Konfiguration

### Anpassung der Trainingsparameter
Das System verwendet vorkonfigurierte, optimierte Hyperparameter für jede Agent-Environment-Kombination, die direkt im Code definiert sind. Für die meisten Anwendungsfälle sind keine Änderungen nötig.

**Erweiterte Konfiguration:**
Hyperparameter können direkt in den entsprechenden Agent-Klassen angepasst werden:
- `src/agents/standard_ppo.py` - Standard PPO Hyperparameter
- `src/agents/lstm.py` - LSTM-spezifische Einstellungen
- `src/agents/ld.py` - Lambda-Discrepancy-Koeffizienten
- `src/core/environment.py` - Environment-Konfigurationen

### Kommandozeilen-Optionen

| Option | Beschreibung | Beispiel |
|--------|-------------|----------|
| `--output` | Output-Verzeichnis | `--output ./results` |
| `--timesteps` | Training-Schritte | `--timesteps 100000` |
| `--seed` | Random Seed | `--seed 123` |
| `--eval-episodes` | Evaluation-Episodes | `--eval-episodes 100` |
| `--no-plots` | Plots überspringen | `--no-plots` |
| `--no-reports` | Reports überspringen | `--no-reports` |
| `--debug` | Debug-Modus | `--debug` |
| `--dry-run` | Konfiguration anzeigen | `--dry-run` |

## Output-Struktur

Nach dem Training erstellt das System folgende Struktur:

```
results/
└── 20250717_143022/          # Timestamped Session
    ├── models/               # Trainierte Modelle
    │   ├── standard_ppo_CartPole-v1.zip
    │   ├── lstm_CartPole-v1.zip
    │   └── ld_CartPole-v1.zip
    ├── tensorboard/          # TensorBoard-Logs
    │   ├── standard_ppo/
    │   ├── lstm/
    │   └── ld/
    ├── plots/               # Generierte Plots
    │   ├── learning_curves_CartPole-v1.png
    │   ├── agent_comparison.png
    │   └── comprehensive_matrix.png
    ├── reports/             # Text-Reports
    │   ├── comparison_report.md
    │   ├── results_summary.csv
    │   └── detailed_results.json
    └── evaluation/          # Evaluation-Daten
        ├── standard_ppo_eval.json
        ├── lstm_eval.json
        └── ld_eval.json
```

## Wissenschaftliche Verwendung

### Lambda-Discrepancy Methode
Der LD Agent implementiert die λ-Discrepancy-Methode aus dem Paper:
*"Mitigating Partial Observability in Sequential Decision Processes via the Lambda Discrepancy"*

**Mathematik:**
- D_t = TD(0)_t - V(s_t)
- TD(0)_t = r_t + γ·V(s_{t+1})
- Loss = PPO-Loss + λ·|LD_prediction - D_t|

### Hyperparameter-Optimierung
Jeder Agent hat environment-spezifische Hyperparameter:

- **CartPole-v1**: Schnelle Konvergenz, kleine Batch-Sizes
- **Acrobot-v1**: Momentum-orientierte Parameter für Underactuated Control
- **LunarLander-v3**: Balancierte Parameter für komplexe Dynamics
- **FrozenLake-v1**: Hohe LD-Koeffizienten für Stochastizität
- **Acrobot-v1 (Partial)**: Verstärkte LSTM/Memory-Parameter für partielle Beobachtbarkeit

## Fehlerbehebung

### Häufige Probleme

**Import-Fehler:**
```bash
# Stelle sicher, dass du im src/ Verzeichnis bist
cd src/
python main.py --list
```

**CUDA-Probleme:**
```bash
# CPU-Only Training erzwingen
export CUDA_VISIBLE_DEVICES=""
python main.py --agent standard_ppo --env CartPole-v1
```

**Speicher-Probleme:**
```bash
# Kleinere Batch-Sizes verwenden
python main.py --agent lstm --env CartPole-v1 --timesteps 10000
```

### Debug-Modus
Für detaillierte Informationen:

```bash
python main.py --debug --dry-run --agent lstm --env CartPole-v1
```

## Weiterführende Informationen

- **ARCHITECTURE.md** - Technische Dokumentation und Systemarchitektur
- **legacy/** - Experimentelle und Referenz-Implementierungen

## Beitragen

Das System ist modular aufgebaut. Neue Agenten können einfach hinzugefügt werden:

1. Neue Agent-Klasse in `src/agents/` erstellen
2. Agent in `src/agents/__init__.py` registrieren
3. Environment-spezifische Hyperparameter hinzufügen

## Lizenz

Dieses Projekt ist Teil einer Masterarbeit über Reinforcement Learning und Lambda-Discrepancy-Methoden.
