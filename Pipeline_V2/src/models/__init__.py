"""
Models module - Custom RL models and policies for Pokemon Red experiments.

Contains:
- RecurrentPPOLD: PPO with Lambda Discrepancy auxiliary loss
- MultiInputLstmPolicyLD: Policy with dual value heads (MC + LD)
"""
from .recurrent_ppo_ld import MultiInputLstmPolicyLD, RecurrentPPOLD

__all__ = ["RecurrentPPOLD", "MultiInputLstmPolicyLD"]
