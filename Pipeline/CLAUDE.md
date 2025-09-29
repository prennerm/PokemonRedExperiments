# Claude Code Configuration for Pokemon Red RL Pipeline

## Project Context
This is a Pokemon Red reinforcement learning training pipeline with multiple environment variants (v1-v4) and λ-discrepancy PPO implementation. The project focuses on training RL agents to play Pokemon Red using PyBoy emulator.

## Environment Setup
```bash
# Primary environment for training and development
conda activate poke_env

# Visualization and analysis environment
conda activate poke_viz_extended
```

## Key Commands
```bash
# Training variants
python -m poke_pipeline.train --variant v4 --config configs/v4_ld_025.yaml

# Interactive agent execution
python -m poke_pipeline.run

# Enhanced heatmap visualization
python notebooks/enhanced_heatmap_visualization.py --variant v4 --max-files 20 --max-samples 5000

# Linting and type checking (when available)
# npm run lint
# npm run typecheck
```

## Project Structure
- `src/poke_pipeline/` - Core training pipeline
  - `train.py` - Main training script
  - `red_gym_env_*.py` - Environment variants (v1-v4)
  - `ppo_lambda_discrepancy.py` - Core λ-discrepancy implementation
  - `global_map.py` - Pokemon Red coordinate system
  - `data/map_data.json` - Map metadata
- `configs/` - Training configurations
- `experiments/` - Training runs and results
  - `{variant}/{timestamp}/plots/` - Visualization outputs
- `notebooks/` - Analysis and visualization tools
- `visualization/` - Legacy visualization code

## Current Priority Issues
1. **Training Stability** - SubprocVecEnv deadlocks with staggered resets
2. **Coordinate System Mapping** - Global coordinates vs Pokemon Red map image alignment
3. **Code Organization** - Duplicate code between environment versions
4. **Performance Optimization** - Memory usage and GPU utilization

## Development Notes
- Use `poke_viz_extended` environment for all visualization tasks
- All plots should be saved in experiment-specific `plots/` directories
- JSON logs contain position data with (x, y, map_id) coordinates
- Global coordinate system: 484x476 grid with map-specific offsets
- Pokemon Red map image: 4000x4000 pixels (coordinate mapping needs verification)

## Common Workflows
1. **Start new training**: Choose variant + config, verify GPU availability
2. **Analyze existing run**: Use enhanced heatmap visualization with appropriate sampling
3. **Debug training issues**: Check JSON logs, memory usage, environment resets
4. **Compare runs**: Use run comparison framework (pending implementation)

## Best Practices
- Always use `--max-files` parameter for large training runs to avoid timeout
- Verify coordinate ranges make sense for the expected Pokemon Red locations
- Keep experiments organized by timestamp in dedicated directories
- Use verbose output for debugging, quiet mode for production runs