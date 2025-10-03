# Visualization Suggestions for Lambda Discrepancy Paper

**Date:** 2025-09-30
**Purpose:** Specific visualization ideas for demonstrating Lambda Discrepancy effectiveness in Pokemon Red RL
**Context:** v4 (LSTM + LD) is the main contribution, compare against v1-v3 baselines

## Core Research Question

"Does LSTM + Lambda Discrepancy lead to more stable and effective policies in complex POMDP environments compared to classical PPO approaches?"

## Visualization Categories

### 1. **Core Contribution Analysis: Lambda Discrepancy Effect**

**Purpose:** Isolate the specific contribution of Lambda Discrepancy auxiliary loss

**Key Comparison:** v3 (LSTM only) vs v4 (LSTM + LD)

**Suggested Plots:**
- **Learning curves with confidence intervals** - Show mean ± std reward over training steps
- **Policy stability metrics** - Variance in episodic rewards over rolling windows
- **Convergence speed analysis** - Time to reach performance thresholds
- **Loss component tracking** - PPO loss vs LD auxiliary loss over time

**Why This Matters:** Directly addresses your research question by isolating the LD contribution from the LSTM memory effect.

**Implementation Notes:**
- Use reward curves from tensorboard logs
- Calculate rolling statistics (mean, variance) over episode windows
- Highlight periods where v4 shows improved stability over v3

---

### 2. **Memory vs No-Memory Necessity**

**Purpose:** Demonstrate that memory is necessary for this POMDP environment

**Key Comparisons:**
- v2 (single frame) vs v3/v4 (LSTM variants)
- v1 (frame stacking) vs v3/v4 (explicit vs implicit memory)

**Suggested Plots:**
- **Performance gap visualization** - Bar charts showing final performance differences
- **Learning efficiency comparison** - Steps required to reach performance milestones
- **Exploration effectiveness** - New coordinates discovered per episode over time
- **Decision consistency analysis** - How similarly agents behave in repeated scenarios

**Why This Matters:** Establishes the baseline need for memory in Pokemon Red, justifying the LSTM approach before demonstrating LD improvements.

**Implementation Notes:**
- Focus on metrics where memory provides clear advantages
- Show both learning speed and final performance differences
- Use Pokemon Red-specific metrics (badges, levels, exploration)

---

### 3. **Lambda Discrepancy Auxiliary Loss Dynamics**

**Purpose:** Show how the LD auxiliary loss evolves and contributes to learning

**Suggested Plots:**
- **LD loss trajectory** - Lambda Discrepancy values decreasing over training (should trend toward 0 as representations become more Markovian)
- **LD-Performance correlation** - Correlation between LD reduction and policy improvement
- **Auxiliary loss weighting study** - Performance with different α values (LD loss coefficient)
- **TD(λ) discrepancy visualization** - Show Q^λ=0 vs Q^λ=1 differences over time

**Why This Matters:** Provides mechanistic insight into how Lambda Discrepancy works as auxiliary loss, supporting the theoretical foundation.

**Implementation Notes:**
- Extract LD loss values from training logs (if logged)
- Create correlation plots with reward improvements
- Consider ablation study with α ∈ [0.1, 0.25, 0.5] if multiple runs available

---

### 4. **POMDP-Specific Behavioral Analysis**

**Purpose:** Show qualitative differences in agent behavior that demonstrate better memory utilization

**Suggested Plots:**
- **Information retention analysis** - How well agents remember past events (e.g., NPC locations, item pickups)
- **Decision consistency metrics** - Behavior similarity in repeated but temporally separated situations
- **Exploration pattern analysis** - Systematic vs random exploration behaviors
- **Action sequence analysis** - Coherent multi-step action plans vs reactive behavior

**Why This Matters:** Demonstrates that LD doesn't just improve numbers, but leads to more intelligent, memory-aware behavior.

**Implementation Notes:**
- Design metrics that capture memory-dependent decisions
- Use action sequence analysis from JSON logs
- Create behavioral fingerprints for each variant

---

### 5. **Pokemon Red Game-Specific Progress**

**Purpose:** Show concrete progress in the actual game environment

**Suggested Plots:**
- **Game milestone progression** - Badges obtained, gym leaders defeated over time
- **Team development analysis** - Pokemon levels, team composition evolution
- **Exploration efficiency** - Map coverage, new areas discovered per episode
- **Game state advancement** - Story progress, key items collected

**Why This Matters:** Grounds the technical improvements in actual game performance that readers can understand intuitively.

**Implementation Notes:**
- Extract game-specific metrics from JSON logs
- Use cumulative progress curves
- Show both speed of progress and final achievements

---

## Implementation Priority

### High Priority (Core Paper Results)
1. **Lambda Discrepancy Effect (v3 vs v4)**
2. **Memory Necessity (v2 vs v3/v4)**
3. **LD Auxiliary Loss Dynamics**

### Medium Priority (Supporting Evidence)
4. **POMDP Behavioral Analysis**
5. **Pokemon Red Progress Metrics**

### Low Priority (If Time/Space Permits)
- Ablation studies with different α values
- Detailed action sequence analysis
- Cross-variant exploration pattern comparisons

## Technical Implementation Notes

### Data Processing
- Use `visualize_training.py` streaming architecture as base
- Apply reservoir sampling for large datasets (>100k samples)
- Target <10 second processing times for rapid iteration

### Plot Requirements
- **Format:** PNG/PDF for paper inclusion
- **Style:** Clean, publication-ready matplotlib figures
- **Statistics:** Include error bars, confidence intervals where appropriate
- **Comparison:** Side-by-side or overlay plots for variant comparisons

### Code Organization
- Create modular plotting functions for each visualization type
- Use consistent color schemes across all plots (v1=blue, v2=green, v3=orange, v4=red)
- Include data validation and error handling for missing/corrupted logs

## Expected Outcomes

If Lambda Discrepancy is effective, we should see:
- **v4 > v3** in learning stability and final performance
- **v3/v4 >> v2** demonstrating memory necessity
- **LD loss decreasing** over training as representations become more Markovian
- **Behavioral improvements** in decision consistency and exploration efficiency

These visualizations will provide both quantitative evidence and qualitative insights supporting your Lambda Discrepancy contribution to POMDP reinforcement learning.