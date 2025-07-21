"""
Evaluation utilities for trained agents.
"""

import numpy as np
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv
import gymnasium as gym

from .environment import EnvironmentFactory


class PolicyEvaluator:
    """Evaluator for trained policies."""
    
    def __init__(self, env_factory: EnvironmentFactory):
        self.env_factory = env_factory
    
    def evaluate_random_policy(self, env_name: str, n_episodes: int = 20, 
                              seed: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate random policy performance."""
        env = self.env_factory.make_env(env_name, seed=seed)
        config = self.env_factory.get_env_config(env_name)
        
        episode_rewards = []
        episode_lengths = []
        successes = []
        
        success_threshold = config.get('success_threshold', float('inf'))
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Check success based on environment threshold
            successes.append(episode_reward >= success_threshold)
        
        env.close()
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'success_rate': np.mean(successes) if successes else 0.0,
            'n_episodes': n_episodes
        }
    
    def evaluate_trained_policy(self, model, env_name: str, n_episodes: int = 20, 
                               seed: Optional[int] = None, deterministic: bool = True) -> Dict[str, Any]:
        """Evaluate trained policy performance."""
        env = self.env_factory.make_env(env_name, seed=seed)
        config = self.env_factory.get_env_config(env_name)
        
        start_time = time.time()
        
        # Use SB3's evaluate_policy for consistency
        episode_rewards, episode_lengths = evaluate_policy(
            model, 
            env, 
            n_eval_episodes=n_episodes, 
            deterministic=deterministic,
            return_episode_rewards=True
        )
        
        evaluation_time = time.time() - start_time
        
        # Calculate success rate
        success_threshold = config.get('success_threshold', float('inf'))
        successes = []
        
        for reward in episode_rewards:
            # Use consistent success logic based on environment thresholds
            successes.append(reward >= success_threshold)
        
        env.close()
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'episode_rewards': list(episode_rewards),
            'episode_lengths': list(episode_lengths),
            'success_rate': np.mean(successes) if successes else 0.0,
            'n_episodes': n_episodes,
            'evaluation_time': evaluation_time,
            'deterministic': deterministic
        }
    
    def detailed_evaluation(self, model, env_name: str, agent_name: str, 
                           n_episodes: int = 20, seed: Optional[int] = None) -> Dict[str, Any]:
        """Perform detailed evaluation including model information."""
        config = self.env_factory.get_env_config(env_name)
        
        # Standard evaluation
        eval_results = self.evaluate_trained_policy(
            model, env_name, n_episodes=n_episodes, seed=seed
        )
        
        # Add model information
        model_info = {
            'agent_name': agent_name,
            'env_name': env_name,
            'model_type': type(model).__name__,
            'policy_type': type(model.policy).__name__ if hasattr(model, 'policy') else 'unknown',
            'model_size': self._estimate_model_size(model),
            'expected_random_reward': config.get('expected_random_reward', 0),
            'success_threshold': config.get('success_threshold', float('inf')),
            'timesteps_trained': config.get('timesteps', 0)
        }
        
        # Combine results
        detailed_results = {**eval_results, **model_info}
        
        # Add performance metrics
        detailed_results['improvement_over_random'] = (
            detailed_results['mean_reward'] - model_info['expected_random_reward']
        )
        
        detailed_results['baseline_passed'] = (
            detailed_results['mean_reward'] >= model_info['success_threshold']
        )
        
        return detailed_results
    
    def _estimate_model_size(self, model) -> int:
        """Estimate model size in parameters."""
        try:
            if hasattr(model, 'policy'):
                total_params = sum(p.numel() for p in model.policy.parameters())
                return total_params
            else:
                return 0
        except:
            return 0
    
    def compare_agents(self, models: Dict[str, Any], env_name: str, 
                      n_episodes: int = 20, seed: Optional[int] = None) -> Dict[str, Any]:
        """Compare multiple agents on the same environment."""
        results = {}
        
        for agent_name, model in models.items():
            print(f"  Evaluating {agent_name}...")
            results[agent_name] = self.detailed_evaluation(
                model, env_name, agent_name, n_episodes=n_episodes, seed=seed
            )
        
        # Add comparison metrics
        mean_rewards = {name: result['mean_reward'] for name, result in results.items()}
        best_agent = max(mean_rewards, key=mean_rewards.get)
        worst_agent = min(mean_rewards, key=mean_rewards.get)
        
        comparison_summary = {
            'best_agent': best_agent,
            'worst_agent': worst_agent,
            'best_reward': mean_rewards[best_agent],
            'worst_reward': mean_rewards[worst_agent],
            'performance_gap': mean_rewards[best_agent] - mean_rewards[worst_agent],
            'agent_ranking': sorted(mean_rewards.items(), key=lambda x: x[1], reverse=True)
        }
        
        return {
            'individual_results': results,
            'comparison_summary': comparison_summary,
            'env_name': env_name,
            'n_episodes': n_episodes
        }


# Convenience functions for backward compatibility
def evaluate_random_policy(env_name: str, env_factory: EnvironmentFactory, 
                          n_episodes: int = 20, seed: Optional[int] = None) -> Dict[str, Any]:
    """Evaluate random policy performance."""
    evaluator = PolicyEvaluator(env_factory)
    return evaluator.evaluate_random_policy(env_name, n_episodes, seed)


def evaluate_trained_policy(model, env_name: str, env_factory: EnvironmentFactory,
                           n_episodes: int = 20, seed: Optional[int] = None) -> Dict[str, Any]:
    """Evaluate trained policy performance."""
    evaluator = PolicyEvaluator(env_factory)
    return evaluator.evaluate_trained_policy(model, env_name, n_episodes, seed)


def detailed_evaluation(model, env_name: str, agent_name: str, env_factory: EnvironmentFactory,
                       n_episodes: int = 20, seed: Optional[int] = None) -> Dict[str, Any]:
    """Perform detailed evaluation including model information."""
    evaluator = PolicyEvaluator(env_factory)
    return evaluator.detailed_evaluation(model, env_name, agent_name, n_episodes, seed)