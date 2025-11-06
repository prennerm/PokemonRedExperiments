# Pokemon Red RL Visualization Guide - Data Processing & Best Practices

**Last Updated:** 2025-09-30
**Purpose:** Guide for creating effective visualizations from Pokemon Red RL training data
**Context:** Lambda Discrepancy paper research - comparing v1-v4 agent variants

## Project Context

This is a Pokemon Red reinforcement learning project investigating Lambda Discrepancy (LD) as auxiliary loss for improving LSTM-based agents in partially observable environments. Four agent variants are being compared:

- **v1:** PPO with Frame-Stacking (3 frames) - context-free agent
- **v2:** PPO with Single-Frame - minimal baseline
- **v3:** Recurrent PPO with LSTM - memory through recurrent representation
- **v4:** Recurrent PPO with LSTM + Lambda Discrepancy Loss - main contribution

## Data Format Overview

JSON training logs contain rich nested structures with temporal, spatial, and performance data.

## JSON Data Structure

Training logs contain comprehensive agent state information:

```json
{
  "step": 0,
  "total_steps": 32768,
  "position": {"x": 3, "y": 6, "map": 38},
  "rewards": {
    "total": 0.0,
    "components": {"event": 0.0, "level": 0.0, "heal": 0.0, "badge": 0.0, "explore": 0.0, "dead": -0.0, "stuck": -0.0}
  },
  "player_status": {"health": 0.0, "levels": [0,0,0,0,0,0], "badges": 0, "pokemon_count": 0},
  "actions": {"last_action": 5},
  "statistics": {"deaths": 0, "exploration_coords": 0}
}
```

## Large Dataset Processing

Critical techniques for handling massive training datasets (100k+ samples):

1. **Reservoir Sampling Algorithm**
   - Maintains fixed memory footprint (e.g., 25k samples) regardless of dataset size
   - Provides statistically representative sample from massive datasets
   - Algorithm: For sample N, if reservoir full, replace random existing sample with probability k/N

2. **Intelligent File Sampling**
   - Process subset of files (e.g., every 10th file) for faster iteration
   - Within-file sampling: Take every N-th entry for temporal distribution
   - Configurable limits: `max_files`, `max_samples_per_file`, `target_sample_size`

3. **Streaming Processing**
   - Process files sequentially, one at a time
   - `gc.collect()` after each file to prevent memory accumulation
   - Never load entire dataset into memory simultaneously

4. **Memory Management**
   - Constant memory usage regardless of dataset size
   - Progress tracking with `tqdm`
   - Graceful error handling for corrupted files

## Available Data for Analysis

### Rich Behavioral Data
- **Reward decomposition:** Individual components (event, level, heal, badge, explore, dead, stuck)
- **Player progression:** Health, levels, badges, pokemon count/types
- **Spatial dynamics:** Position tracking across maps with global coordinate system
- **Action patterns:** Action sequences, decision consistency
- **Learning metrics:** Death rates, exploration efficiency, behavioral stability

### Dataset Scale
- **Typical size:** 3,000+ JSON files per experiment
- **Entry density:** ~200-650k individual timesteps per training run
- **Processing capability:** ~2.6 files/second with streaming
- **Memory efficiency:** Constant ~50MB footprint regardless of dataset size

## Research Visualization Requirements

### Paper-Focused Analysis Needs
- **Static plots** for academic papers (PNG/PDF format)
- **Comparative analysis** between agent variants (v1-v4)
- **Lambda Discrepancy effectiveness** demonstration
- **POMDP-specific metrics** showing memory necessity
- **Processing speed** <10 seconds for rapid iteration

### Key Insights
- **Purpose-built tools** outperform adapted legacy scripts
- **Simple analytical plots** more valuable than complex animations
- **Research focus** should drive visualization design, not inherited complexity

## Valuable Lessons Learned

### 1. **Reservoir Sampling is Extremely Powerful**
- **Best practice:** Use for any dataset >100k samples
- **Implementation:** `StreamingProcessor` class from `visualize_training.py`
- **Key insight:** Maintains statistical properties while enabling interactive analysis

### 2. **Streaming Architecture Principles**
From `notebooks/visualize_training.py`:
```python
# Process files one by one
for json_file in json_files:
    entries = process_file_streaming(json_file)
    update_reservoir_sampling(entries)
    gc.collect()  # Critical for memory management
```

### 3. **Intelligent Sampling Strategies**
- **Temporal sampling:** Take every N-th entry within files for time distribution
- **File sampling:** Process subset of files for faster iteration during development
- **Adaptive sampling:** Adjust sample size based on file size and available resources

### 4. **Memory Management Best Practices**
- Always call `gc.collect()` after processing large files
- Use generators/iterators instead of loading full datasets
- Monitor memory usage during development with appropriate limits

## Implementation Guidelines

### For New Visualization Tools
1. **Start with purpose-built scripts** rather than adapting existing tools
2. **Use streaming + reservoir sampling** for datasets >10k samples
3. **Template:** Base new tools on `visualize_training.py` `StreamingProcessor` class
4. **Focus on research questions** - Lambda Discrepancy effectiveness, POMDP learning
5. **Optimize for iteration speed** - aim for <10 second processing times

### Environment Setup
- **Training environment:** `conda activate poke_env`
- **Visualization environment:** `conda activate poke_viz_extended`
- **Key command:** `python notebooks/enhanced_heatmap_visualization.py --variant v4 --max-files 20 --max-samples 5000`

### Data Locations
- **Experiments:** `experiments/{variant}/{timestamp}/json_logs/`
- **Configs:** `configs/v{1-4}*.yaml`
- **Global coordinates:** `src/poke_pipeline/global_map.py`
- **Map metadata:** `src/poke_pipeline/data/map_data.json`

## Research Focus Areas

### Core Research Question
"Does LSTM + Lambda Discrepancy lead to more stable and effective policies in complex POMDP environments compared to classical PPO approaches?"

### Key Comparisons Needed
1. **v3 vs v4:** Isolate Lambda Discrepancy contribution
2. **v2 vs v3/v4:** Demonstrate memory necessity
3. **v1 vs v3/v4:** Explicit vs implicit temporal information
4. **v4 ablation:** Different λ coefficients in auxiliary loss

**Key Insight:** Use `visualize_training.py` streaming architecture as foundation for all large-dataset visualization tools.