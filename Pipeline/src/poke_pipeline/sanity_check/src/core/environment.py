"""
Environment factory and wrapper utilities.
"""

import gymnasium as gym
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from utils.wrappers import DictObsWrapper, PartialObservabilityWrapper


class EnvironmentFactory:
    """Factory for creating standardized environments."""
    
    def __init__(self):
        """Initialize with hardcoded environment configurations."""
        self.env_configs = self._get_default_env_configs()
    
    def _get_default_env_configs(self) -> Dict[str, Any]:
        """Get default environment configurations based on ARCHITECTURE.md."""
        return {
            'CartPole-v1': {
                'env_id': 'CartPole-v1',
                'env_kwargs': {},
                'wrapper_type': 'continuous_dict',
                'description': 'Classic control task - balancing pole on cart',
                'timesteps': 100000,
                'success_threshold': 195.0  # Gymnasium's solve threshold for CartPole-v1
            },
            'LunarLander-v3': {
                'env_id': 'LunarLander-v3',
                'env_kwargs': {},
                'wrapper_type': 'continuous_dict', 
                'description': 'Complex dynamics with shaped rewards',
                'timesteps': 300000,
                'success_threshold': 200.0
            },
            'FrozenLake-v1': {
                'env_id': 'FrozenLake-v1',
                'env_kwargs': {'is_slippery': True},
                'wrapper_type': 'discrete_dict',
                'description': 'Stochastic gridworld with sparse rewards',
                'timesteps': 150000,
                'success_threshold': 0.7
            },
            'Acrobot-v1': {
                'env_id': 'Acrobot-v1',
                'env_kwargs': {},
                'wrapper_type': 'continuous_dict',
                'description': 'Underactuated double-pendulum control',
                'timesteps': 200000,
                'success_threshold': -100.0  # Episodic reward (negative time to reach goal)
            },
            'Acrobot-v1-Partial': {
                'env_id': 'Acrobot-v1',
                'env_kwargs': {},
                'wrapper_type': 'continuous_dict_partial',
                'description': 'Acrobot with partial observability (no velocity info)',
                'timesteps': 250000,
                'success_threshold': -150.0  # Harder due to partial observability
            }
        }
    
    def make_env(self, env_name: str, seed: Optional[int] = None, monitor_path: Optional[Path] = None, 
                 enable_reward_shaping: bool = True) -> gym.Env:
        """Create a single environment with appropriate wrappers."""
        if env_name not in self.env_configs:
            raise ValueError(f"Environment '{env_name}' not found in config. Available: {list(self.env_configs.keys())}")
        
        config = self.env_configs[env_name]
        
        # Create base environment
        env = gym.make(config['env_id'], **config.get('env_kwargs', {}))
        
        # Set seed if provided
        if seed is not None:
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        
        
        # Add Monitor wrapper if path provided
        if monitor_path is not None:
            env = Monitor(env, str(monitor_path))
        
        # Apply partial observability wrapper if needed
        wrapper_type = config.get('wrapper_type', 'continuous_dict')
        if wrapper_type == 'continuous_dict_partial':
            # Apply partial observability wrapper for Acrobot-v1-Partial
            env = PartialObservabilityWrapper(env)
        
        # Add DictObsWrapper based on wrapper_type
        if wrapper_type == 'discrete_dict':
            env = DictObsWrapper(env, discrete=True)
        elif wrapper_type in ['continuous_dict', 'continuous_dict_partial']:
            env = DictObsWrapper(env, discrete=False)
        else:
            raise ValueError(f"Unknown wrapper_type: {wrapper_type}")
        
        return env
    
    def make_vec_env(self, env_name: str, n_envs: int = 1, seed: Optional[int] = None, 
                     monitor_dir: Optional[Path] = None, enable_reward_shaping: bool = True) -> DummyVecEnv:
        """Create vectorized environment."""
        def _make_env(rank: int):
            def _init():
                monitor_path = monitor_dir / f"env_{rank}" if monitor_dir else None
                env_seed = seed + rank if seed is not None else None
                return self.make_env(env_name, seed=env_seed, monitor_path=monitor_path, 
                                   enable_reward_shaping=enable_reward_shaping)
            return _init
        
        vec_env = DummyVecEnv([_make_env(i) for i in range(n_envs)])
        
        # Add VecNormalize for CartPole (based on advanced_sanity_check.py)
        if env_name == 'CartPole-v1':
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
        
        return vec_env
    
    def get_env_config(self, env_name: str) -> Dict[str, Any]:
        """Get configuration for specific environment."""
        if env_name not in self.env_configs:
            raise ValueError(f"Environment '{env_name}' not found in config")
        return self.env_configs[env_name]
    
    def evaluate_random_policy(self, env_name: str, n_episodes: int = 20, seed: Optional[int] = None) -> Tuple[float, float]:
        """Evaluate random policy performance."""
        env = self.make_env(env_name, seed=seed)
        
        episode_rewards = []
        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
        
        env.close()
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        return mean_reward, std_reward
    
    def get_available_environments(self) -> list:
        """Get list of available environment names."""
        return list(self.env_configs.keys())


# Convenience functions for backward compatibility
def load_env_configs() -> Dict[str, Any]:
    """Get environment configurations."""
    factory = EnvironmentFactory()
    return factory.env_configs


def make_env(env_name: str, seed: Optional[int] = None, 
             monitor_path: Optional[Path] = None, enable_reward_shaping: bool = True) -> gym.Env:
    """Create a single environment with appropriate wrappers."""
    factory = EnvironmentFactory()
    return factory.make_env(env_name, seed=seed, monitor_path=monitor_path, 
                           enable_reward_shaping=enable_reward_shaping)