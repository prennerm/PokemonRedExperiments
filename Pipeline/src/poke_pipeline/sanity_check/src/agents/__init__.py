"""
Agent Module for Multi-Agent Benchmark Suite.

This module provides three different reinforcement learning agents for
systematic comparison on standard Gym environments:

1. StandardPPOAgent - Baseline PPO implementation
2. LSTMAgent - LSTM-enhanced PPO with memory capabilities  
3. LDAgent - Lambda-Discrepancy PPO for partial observability

All agents follow the BaseAgent interface for consistent usage across
the benchmark suite.
"""

from .base import BaseAgent
from .standard_ppo import StandardPPOAgent
from .lstm import LSTMAgent
from .ld import LDAgent

# Export all agent classes
__all__ = [
    'BaseAgent',
    'StandardPPOAgent', 
    'LSTMAgent',
    'LDAgent'
]

# Agent registry for dynamic loading
AGENT_REGISTRY = {
    'standard_ppo': StandardPPOAgent,
    'ppo': StandardPPOAgent,  # Alias
    'baseline': StandardPPOAgent,  # Alias
    'lstm': LSTMAgent,
    'recurrent': LSTMAgent,  # Alias
    'memory': LSTMAgent,  # Alias
    'ld': LDAgent,
    'lambda_discrepancy': LDAgent,  # Alias
    'partial_obs': LDAgent,  # Alias
}

def get_agent_class(agent_name: str):
    """
    Get agent class by name.
    
    Args:
        agent_name: Name of the agent (case-insensitive)
        
    Returns:
        Agent class
        
    Raises:
        ValueError: If agent name is not found
    """
    agent_name = agent_name.lower().strip()
    
    if agent_name not in AGENT_REGISTRY:
        available_agents = list(AGENT_REGISTRY.keys())
        raise ValueError(
            f"Unknown agent '{agent_name}'. "
            f"Available agents: {available_agents}"
        )
    
    return AGENT_REGISTRY[agent_name]

def list_available_agents():
    """
    List all available agent names.
    
    Returns:
        List of available agent names
    """
    return list(AGENT_REGISTRY.keys())

def get_agent_info():
    """
    Get information about all available agents.
    
    Returns:
        Dictionary with agent information
    """
    info = {}
    
    for name, agent_class in AGENT_REGISTRY.items():
        # Skip aliases for main info
        if name in ['standard_ppo', 'lstm', 'ld']:
            try:
                # Create dummy config to get agent info
                dummy_config = {'name': name}
                agent = agent_class(dummy_config)
                info[name] = agent.get_model_info()
            except Exception:
                # Fallback if agent creation fails
                info[name] = {
                    'algorithm': 'Unknown',
                    'description': f'Agent class: {agent_class.__name__}'
                }
    
    return info
