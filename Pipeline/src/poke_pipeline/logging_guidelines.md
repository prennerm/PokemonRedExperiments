# Logging Guidelines for Agent Training

## Objective
The primary goal of this document is to establish clear and consistent guidelines for logging during the training of agents (v1 to v4). The logging process must ensure that the generated JSON files are:

1. **Consistent**: All four agent variants must follow the same logging structure unless a unique feature of a specific agent variant (e.g., v4 - Lambda Discrepancy) necessitates a deviation.
2. **Readable**: The JSON files should be structured in a way that is logical and easy to interpret.
3. **Comprehensive**: All relevant data required for analysis and debugging must be included.

## Rules

### 1. Consistency Across Agents
- **Uniform Logging**: Any changes to the logging process must apply to all four agent variants unless there is a compelling reason tied to a unique feature of a specific agent.
- **Prohibited Adjustments**: It is strictly forbidden to make logging changes that affect only one agent variant without justification.
- **Allowed Adjustments**: Logging adjustments specific to an agent variant are allowed only if they are directly related to a unique feature of that variant (e.g., v4's Lambda Discrepancy).

### 2. JSON Structure
- **Clarity**: The JSON files must be structured in a way that is intuitive and easy to navigate.
- **Key-Value Pairs**: Use descriptive keys for all logged data to ensure clarity.
- **Nested Structure**: Where appropriate, use nested structures to group related data logically.

### 3. Data to Log
- **Common Data**: Ensure that all agents log the following data consistently:
  - Step count
  - Position (x, y, map)
  - Rewards (total, step, and components)
  - Player status (health, levels, badges, etc.)
  - Actions (last action, action history)
  - Statistics (e.g., deaths, progress, exploration metrics)
- **Agent-Specific Data**: Log additional data only if it pertains to a unique feature of the agent variant.

### 4. Review and Validation
- **Code Review**: All changes to the logging process must undergo a thorough review to ensure compliance with these guidelines.
- **Validation**: Test the logging process to confirm that the JSON files are correctly structured and include all necessary data.

## Implementation Notes
- The `StatsCallback` class in `callbacks.py` is the central point for logging during training. Any changes to the logging process should be implemented here.
- The `agent_stats` collected from the environments (`red_gym_env...`) must be consistent across all variants to ensure uniform logging.
- Use the `verbose` flag in `StatsCallback` to print debug information during development and testing.

## Example JSON Structure
```json
{
  "step": 100,
  "position": {
    "x": 10,
    "y": 20,
    "map": 3
  },
  "rewards": {
    "total": 150,
    "step": 10,
    "components": {
      "event": 50,
      "level": 30,
      "heal": 20
    }
  },
  "player_status": {
    "health": 0.8,
    "levels": [5, 10, 15],
    "badges": 3
  },
  "actions": {
    "last_action": "move_up",
    "history": ["move_left", "move_down", "move_up"]
  },
  "statistics": {
    "deaths": 2,
    "exploration": 50
  }
}
```

## Conclusion
By adhering to these guidelines, we ensure that the logging process is robust, consistent, and provides valuable insights for analysis and debugging. Any deviations from these rules must be justified and documented.
