"""
Environment wrappers for standardization.

This module provides wrappers to standardize different Gym environments
for consistent use with the Multi-Agent Benchmark Suite agents.
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Any, Tuple, Optional
from gymnasium.spaces import Box, Discrete, Dict as DictSpace


class DictObsWrapper(gym.ObservationWrapper):
    """
    Wrapper to convert environment observations to dictionary format.
    
    This is required for RecurrentPPO and LSTM agents which expect
    dictionary observations, while standard Gym environments often
    provide Box or Discrete observations.
    """
    
    def __init__(self, env: gym.Env, discrete: bool = False):
        """
        Initialize the wrapper.
        
        Args:
            env: The environment to wrap
            discrete: Whether the environment has discrete observations
        """
        super().__init__(env)
        self.discrete = discrete
        
        # Convert observation space to dictionary format
        if discrete:
            # For discrete environments like FrozenLake
            if isinstance(env.observation_space, Discrete):
                obs_dim = env.observation_space.n
                self.observation_space = DictSpace({
                    "state": Box(0, 1, shape=(obs_dim,), dtype=np.float32)
                })
            else:
                raise ValueError(f"Expected Discrete space for discrete=True, got {type(env.observation_space)}")
        else:
            # For continuous environments like CartPole, MountainCar, LunarLander
            if isinstance(env.observation_space, Box):
                self.observation_space = DictSpace({
                    "state": env.observation_space
                })
            else:
                raise ValueError(f"Expected Box space for discrete=False, got {type(env.observation_space)}")
    
    def observation(self, obs: Any) -> Dict[str, Any]:
        """
        Convert observation to dictionary format.
        
        Args:
            obs: Original observation from environment
            
        Returns:
            Dictionary observation with 'state' key
        """
        if self.discrete:
            # Convert discrete state to one-hot encoding
            if isinstance(self.env.observation_space, Discrete):
                one_hot = np.zeros(self.env.observation_space.n, dtype=np.float32)
                one_hot[obs] = 1.0
                return {"state": one_hot}
            else:
                raise ValueError("Discrete wrapper expects discrete observation space")
        else:
            # Wrap continuous observation in dictionary
            return {"state": obs}


class NoRewardShapingWrapper(gym.Wrapper):
    """
    Wrapper that explicitly documents no reward shaping is applied.
    
    This wrapper serves as documentation that the environment rewards
    are used as-is without any modifications, which is important for
    scientific validity of the benchmark comparisons.
    """
    
    def __init__(self, env: gym.Env):
        """
        Initialize the wrapper.
        
        Args:
            env: The environment to wrap
        """
        super().__init__(env)
        self.original_reward_range = env.reward_range
        
        # Log that no reward shaping is applied
        print(f"🔬 NoRewardShapingWrapper: {env.spec.id} rewards used as-is")
        print(f"   Reward range: {self.original_reward_range}")
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Step function that passes through rewards unchanged.
        
        Args:
            action: Action to take
            
        Returns:
            Standard Gym step return tuple with original rewards
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Document that reward is unchanged
        if 'original_reward' not in info:
            info['original_reward'] = reward
        
        return obs, reward, terminated, truncated, info


class MonitorWrapper(gym.Wrapper):
    """
    Simple monitoring wrapper for episode statistics.
    
    Tracks episode rewards and lengths for basic monitoring
    without modifying the environment behavior.
    """
    
    def __init__(self, env: gym.Env):
        """
        Initialize the wrapper.
        
        Args:
            env: The environment to wrap
        """
        super().__init__(env)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
    
    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment and episode tracking.
        
        Returns:
            Standard Gym reset return tuple
        """
        self.current_episode_reward = 0
        self.current_episode_length = 0
        return self.env.reset(**kwargs)
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Step function that tracks episode statistics.
        
        Args:
            action: Action to take
            
        Returns:
            Standard Gym step return tuple with episode info
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        # Add episode info when episode ends
        if terminated or truncated:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            info['episode'] = {
                'r': self.current_episode_reward,
                'l': self.current_episode_length
            }
        
        return obs, reward, terminated, truncated, info
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """
        Get episode statistics.
        
        Returns:
            Dictionary with episode statistics
        """
        if not self.episode_rewards:
            return {'episodes': 0}
        
        return {
            'episodes': len(self.episode_rewards),
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'mean_length': np.mean(self.episode_lengths),
            'std_length': np.std(self.episode_lengths)
        }


class MountainCarRewardShapingWrapper(gym.Wrapper):
    """
    Reward shaping wrapper for MountainCar-v0 using potential-based shaping.
    
    This wrapper adds potential-based reward shaping to help agents learn
    more efficiently in the sparse reward environment of MountainCar.
    The shaping uses the potential function based on height and velocity.
    
    The original sparse reward structure is preserved while providing
    additional learning signals. This is scientifically sound as it's
    potential-based and doesn't change the optimal policy.
    """
    
    def __init__(self, env: gym.Env, shaping_weight: float = 1.0):
        """
        Initialize the wrapper.
        
        Args:
            env: The MountainCar environment to wrap
            shaping_weight: Weight for the potential-based shaping reward
        """
        super().__init__(env)
        
        if env.spec.id != 'MountainCar-v0':
            raise ValueError(f"MountainCarRewardShapingWrapper only supports MountainCar-v0, got {env.spec.id}")
        
        self.shaping_weight = shaping_weight
        self.previous_potential = None
        
        # MountainCar constants
        self.min_position = -1.2
        self.max_position = 0.6
        self.goal_position = 0.5
        self.max_speed = 0.07
        
        print(f"🚗 MountainCarRewardShapingWrapper: Applied with weight {shaping_weight}")
        print(f"   Improved potential-based shaping: normalized position + velocity components")
        print(f"   Expected shaping reward range: [{-shaping_weight:.2f}, {shaping_weight:.2f}] per step")
    
    def _compute_potential(self, position: float, velocity: float) -> float:
        """
        Compute improved potential function based on position and velocity.
        
        The potential function encourages:
        1. Moving towards the goal (higher position is better)
        2. Having velocity that helps reach the goal
        3. Scaled to be comparable with original reward (-1 per step)
        
        Args:
            position: Car position [-1.2, 0.6]
            velocity: Car velocity [-0.07, 0.07]
            
        Returns:
            Potential value in range approximately [-1, 1]
        """
        # 1. Position component: normalize to [0, 1] where 1 is best (at goal)
        normalized_pos = (position - self.min_position) / (self.max_position - self.min_position)
        position_potential = normalized_pos  # [0, 1], higher is better
        
        # 2. Velocity component: encourage helpful velocity
        # Normalize velocity to [-1, 1]
        normalized_vel = velocity / self.max_speed
        
        # Velocity is helpful if it supports reaching the goal:
        # - Rightward velocity is always good (helps reach goal)
        # - Leftward velocity is only good for momentum building far from goal
        if position < -0.5:  # Far left: need momentum, slight preference for rightward
            velocity_potential = 0.2 * normalized_vel + 0.1 * abs(normalized_vel)
        elif position < 0.0:  # Valley center: strongly prefer rightward velocity
            velocity_potential = 0.4 * max(0, normalized_vel) + 0.1 * abs(normalized_vel)
        else:  # Close to goal: only rightward velocity helps
            velocity_potential = 0.5 * max(0, normalized_vel)
        
        # Combine components with position being more important
        total_potential = 0.7 * position_potential + 0.3 * velocity_potential
        
        # Scale to approximately [-0.5, 0.5] for reasonable shaping rewards
        return total_potential - 0.5
    
    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment and potential tracking.
        
        Returns:
            Standard Gym reset return tuple
        """
        obs, info = self.env.reset(**kwargs)
        
        # Extract position and velocity from observation
        position, velocity = obs['state'] if isinstance(obs, dict) else obs
        self.previous_potential = self._compute_potential(position, velocity)
        
        return obs, info
    
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Step function with potential-based reward shaping.
        
        Args:
            action: Action to take
            
        Returns:
            Standard Gym step return tuple with shaped rewards
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Extract position and velocity from observation
        position, velocity = obs['state'] if isinstance(obs, dict) else obs
        
        # Compute current potential
        current_potential = self._compute_potential(position, velocity)
        
        # Potential-based shaping: F(s') - F(s)
        if self.previous_potential is not None:
            shaping_reward = self.shaping_weight * (current_potential - self.previous_potential)
        else:
            shaping_reward = 0.0
        
        # Update previous potential
        self.previous_potential = current_potential
        
        # Combine original reward with shaping
        shaped_reward = reward + shaping_reward
        
        # Add debug info
        info['original_reward'] = reward
        info['shaping_reward'] = shaping_reward
        info['potential'] = current_potential
        info['position'] = position
        info['velocity'] = velocity
        
        return obs, shaped_reward, terminated, truncated, info


def create_standardized_env(env_id: str, 
                          env_kwargs: Optional[Dict[str, Any]] = None,
                          wrapper_type: str = 'continuous_dict',
                          seed: Optional[int] = None,
                          enable_reward_shaping: bool = True) -> gym.Env:
    """
    Create a standardized environment with appropriate wrappers.
    
    Args:
        env_id: Gym environment ID
        env_kwargs: Additional keyword arguments for environment creation
        wrapper_type: Type of wrapper to apply ('continuous_dict' or 'discrete_dict')
        seed: Random seed for reproducibility
        enable_reward_shaping: Whether to apply reward shaping for suitable environments
        
    Returns:
        Wrapped environment ready for agent training
    """
    if env_kwargs is None:
        env_kwargs = {}
    
    # Create base environment
    env = gym.make(env_id, **env_kwargs)
    
    # Set seed if provided
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    
    # Apply environment-specific reward shaping if enabled
    if enable_reward_shaping and env_id == 'MountainCar-v0':
        env = MountainCarRewardShapingWrapper(env, shaping_weight=1.0)
    else:
        # Apply no-reward-shaping wrapper for documentation
        env = NoRewardShapingWrapper(env)
    
    # Apply monitoring wrapper
    env = MonitorWrapper(env)
    
    # Apply dictionary observation wrapper
    if wrapper_type == 'discrete_dict':
        env = DictObsWrapper(env, discrete=True)
    elif wrapper_type == 'continuous_dict':
        env = DictObsWrapper(env, discrete=False)
    else:
        raise ValueError(f"Unknown wrapper_type: {wrapper_type}")
    
    return env


def get_wrapper_info(env: gym.Env) -> Dict[str, Any]:
    """
    Get information about applied wrappers.
    
    Args:
        env: Wrapped environment
        
    Returns:
        Dictionary with wrapper information
    """
    wrapper_info = {
        'wrappers': [],
        'original_env': None,
        'observation_space': str(env.observation_space),
        'action_space': str(env.action_space)
    }
    
    # Walk through wrapper chain
    current_env = env
    while hasattr(current_env, 'env'):
        wrapper_info['wrappers'].append(type(current_env).__name__)
        current_env = current_env.env
    
    wrapper_info['original_env'] = type(current_env).__name__
    
    return wrapper_info