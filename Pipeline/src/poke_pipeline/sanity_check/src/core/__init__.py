"""
Core module for Multi-Agent Benchmark Suite.

This module provides the fundamental components for running 
multi-agent benchmarks on standard Gym environments.
"""

from .environment import EnvironmentFactory, make_env, load_env_configs
from .evaluator import evaluate_random_policy, evaluate_trained_policy, detailed_evaluation
from .runner import SanityCheckRunner
from .reporter import ComparisonReporter, MarkdownReporter

__all__ = [
    'EnvironmentFactory',
    'make_env', 
    'load_env_configs',
    'evaluate_random_policy',
    'evaluate_trained_policy', 
    'detailed_evaluation',
    'SanityCheckRunner',
    'ComparisonReporter',
    'MarkdownReporter'
]